from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from dotenv import load_dotenv

from agents.ai_matcher import AIMatcher
from agents.ats_discovery import discover_and_verify_companies
from agents.job_prefilter import filter_jobs_for_ai
from agents.job_title_enricher import JobTitleEnricher, needs_title_enrichment, title_from_reasoning
from agents.location_matcher import LocationMatcher
from agents.role_matcher import RoleMatcher
from agents.salary_researcher import SalaryResearcher
from agents.search_providers.router import JobSearchRouter
from agents.big_tech_hunter import BigTechHunter
from agents.startup_discoverer import StartupDiscoverer
from agents.target_hunter import TargetHunter
from models.job import JobPosting, MatchResult, ScanResult
from models.user_profile import UserProfile, read_uses_web_search
from storage.discovered_companies import DiscoveredCompaniesStore
from storage.memory import JobMemory
from storage.scan_history import ScanHistoryStore

logger = logging.getLogger(__name__)

DEFAULT_SCAN_RESULTS_PATH = Path("data/scan_results.json")
ProgressCallback = Callable[[str, dict[str, Any]], None]


def _coerce_job_posting(job: JobPosting) -> JobPosting:
    if isinstance(job, JobPosting):
        return job
    if hasattr(job, "model_dump"):
        return JobPosting.model_validate(job.model_dump(mode="json"))
    return JobPosting.model_validate(job)


class JobHunterOrchestrator:
    def __init__(
        self,
        memory: JobMemory | None = None,
        target_hunter: TargetHunter | None = None,
        startup_discoverer: StartupDiscoverer | None = None,
        big_tech_hunter: BigTechHunter | None = None,
        ai_matcher: AIMatcher | None = None,
        search_router: JobSearchRouter | None = None,
        location_matcher: LocationMatcher | None = None,
        role_matcher: RoleMatcher | None = None,
        score_threshold: float | None = None,
        scan_results_path: Path | str = DEFAULT_SCAN_RESULTS_PATH,
        scan_history_path: Path | str | None = None,
        discovered_companies_path: Path | str | None = None,
    ) -> None:
        load_dotenv()
        self.memory = memory or JobMemory()
        self.search_router = search_router or JobSearchRouter()
        self.target_hunter = target_hunter or TargetHunter()
        self.startup_discoverer = startup_discoverer or StartupDiscoverer(
            search_router=self.search_router,
        )
        self.big_tech_hunter = big_tech_hunter or BigTechHunter(
            search_router=self.search_router,
        )
        self.ai_matcher = ai_matcher or AIMatcher(
            salary_researcher=SalaryResearcher(search_router=self.search_router),
        )
        self.location_matcher = location_matcher or LocationMatcher()
        self.role_matcher = role_matcher or RoleMatcher()
        self.score_threshold = score_threshold or float(os.getenv("MATCH_SCORE_THRESHOLD", "7"))
        self.scan_results_path = Path(scan_results_path)
        self.scan_history_path = Path(scan_history_path) if scan_history_path else None
        self.discovered_companies_path = (
            Path(discovered_companies_path) if discovered_companies_path else None
        )

    def _emit(self, callback: ProgressCallback | None, event: str, payload: dict[str, Any]) -> None:
        if callback:
            callback(event, payload)

    def _all_known_companies(self, profile: UserProfile) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        if self.discovered_companies_path:
            discovered = DiscoveredCompaniesStore(self.discovered_companies_path).load()
        return self.target_hunter._load_companies(profile, discovered_companies=discovered)

    async def run_scan(
        self,
        profile: UserProfile,
        on_progress: ProgressCallback | None = None,
    ) -> ScanResult:
        logger.info("[Orchestrator] Starting scan for roles: %s", ", ".join(profile.target_roles))
        self.search_router.reset_usage_stats()
        self._emit(
            on_progress,
            "status",
            {"message": "Avvio raccolta annunci dagli agenti..."},
        )

        discovered_store = (
            DiscoveredCompaniesStore(self.discovered_companies_path)
            if self.discovered_companies_path
            else None
        )
        persisted_discovered = discovered_store.load() if discovered_store else []

        async def run_startup_discoverer() -> list[JobPosting]:
            if not read_uses_web_search(profile):
                self._emit(
                    on_progress,
                    "agent_done",
                    {"agent": "Startup Discoverer", "count": 0, "skipped": True},
                )
                return []

            self._emit(
                on_progress,
                "status",
                {"message": "Startup Discoverer in esecuzione (ricerche web)..."},
            )
            jobs = await self.startup_discoverer.safe_run(profile, on_progress=on_progress)
            self._emit(
                on_progress,
                "agent_done",
                {"agent": "Startup Discoverer", "count": len(jobs)},
            )
            return jobs

        async def run_big_tech_hunter() -> list[JobPosting]:
            if not read_uses_web_search(profile):
                self._emit(
                    on_progress,
                    "agent_done",
                    {"agent": "Big Tech Hunter", "count": 0, "skipped": True},
                )
                return []

            self._emit(
                on_progress,
                "status",
                {"message": "Big Tech Hunter in esecuzione (Google, IBM, Microsoft, Meta, Amazon...)..."},
            )
            jobs = await self.big_tech_hunter.safe_run(profile, on_progress=on_progress)
            self._emit(
                on_progress,
                "agent_done",
                {"agent": "Big Tech Hunter", "count": len(jobs)},
            )
            return jobs

        async def run_target_hunter() -> list[JobPosting]:
            self._emit(
                on_progress,
                "status",
                {"message": "Target Hunter in esecuzione..."},
            )
            try:
                jobs = await self.target_hunter.run(
                    profile,
                    discovered_companies=persisted_discovered,
                )
                self._emit(
                    on_progress,
                    "agent_done",
                    {"agent": "Target Hunter", "count": len(jobs)},
                )
                return jobs
            except Exception as exc:
                logger.exception("[Target Hunter] Agent failed but pipeline continues: %s", exc)
                return []

        startup_jobs, target_jobs, bigtech_jobs = await asyncio.gather(
            run_startup_discoverer(),
            run_target_hunter(),
            run_big_tech_hunter(),
        )
        self._emit(
            on_progress,
            "phase_done",
            {
                "phase": "collection",
                "message": (
                    f"Raccolta iniziale finita: {len(target_jobs)} da Target Hunter, "
                    f"{len(startup_jobs)} da Startup Discoverer, "
                    f"{len(bigtech_jobs)} da Big Tech Hunter."
                ),
            },
        )

        if read_uses_web_search(profile):
            self._emit(
                on_progress,
                "status",
                {"message": "Verifica aziende ATS dinamiche dai risultati web..."},
            )
            self._emit(
                on_progress,
                "search_providers",
                {
                    "phase": "discovery",
                    "stats": self.search_router.get_usage_stats(),
                },
            )

            known_companies = self._all_known_companies(profile)
            newly_verified = await discover_and_verify_companies(
                startup_jobs + bigtech_jobs,
                profile,
                known_companies,
                self.target_hunter,
            )
            self._emit(
                on_progress,
                "phase_done",
                {
                    "phase": "ats_discovery",
                    "message": (
                        f"Discovery ATS completata: {len(newly_verified)} nuove aziende verificate."
                    ),
                },
            )
        else:
            self._emit(
                on_progress,
                "status",
                {"message": "Modalità senza ricerche web: salto discovery ATS dinamica."},
            )
            newly_verified = []

        dynamic_jobs: list[JobPosting] = []
        if newly_verified:
            if discovered_store:
                discovered_store.merge_new(newly_verified)
            names = ", ".join(company["name"] for company in newly_verified)
            self._emit(
                on_progress,
                "status",
                {"message": f"Nuove aziende ATS verificate: {names}"},
            )
            self._emit(
                on_progress,
                "companies_discovered",
                {"companies": newly_verified},
            )
            dynamic_jobs = await self.target_hunter.fetch_companies(profile, newly_verified)
            self._emit(
                on_progress,
                "agent_done",
                {"agent": "Target Hunter (ATS dinamico)", "count": len(dynamic_jobs)},
            )
            self._emit(
                on_progress,
                "phase_done",
                {
                    "phase": "dynamic_target_hunter",
                    "message": (
                        f"Fetch ATS dinamico completato: {len(dynamic_jobs)} annunci aggiunti."
                    ),
                },
            )

        merged_jobs = self._deduplicate_jobs(
            target_jobs + dynamic_jobs + startup_jobs + bigtech_jobs
        )
        new_jobs = self.memory.get_new_jobs(merged_jobs)
        if new_jobs:
            self._emit(
                on_progress,
                "status",
                {"message": f"Arricchimento titoli annunci ({len(new_jobs)} nuovi)..."},
            )
            title_enricher = JobTitleEnricher()
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                new_jobs = await title_enricher.enrich_many(client, new_jobs)
            self._emit(
                on_progress,
                "phase_done",
                {
                    "phase": "title_enrichment",
                    "message": f"Arricchimento titoli completato su {len(new_jobs)} nuovi annunci.",
                },
            )
        else:
            self._emit(
                on_progress,
                "phase_done",
                {
                    "phase": "title_enrichment",
                    "message": "Nessun nuovo annuncio da arricchire.",
                },
            )
        eligible_jobs, prefilter_skipped = await filter_jobs_for_ai(
            new_jobs,
            profile,
            location_matcher=self.location_matcher,
            role_matcher=self.role_matcher,
            on_progress=on_progress,
        )
        self._emit(
            on_progress,
            "phase_done",
            {
                "phase": "prefilter",
                "message": (
                    f"Pre-filtro completato: {len(eligible_jobs)} idonei, "
                    f"{prefilter_skipped} scartati prima dell'AI."
                ),
            },
        )
        self._emit(
            on_progress,
            "summary",
            {
                "total_found": len(merged_jobs),
                "new_jobs": len(new_jobs),
                "eligible_jobs": len(eligible_jobs),
                "prefilter_skipped": prefilter_skipped,
                "skipped_seen": len(merged_jobs) - len(new_jobs),
                "dynamic_companies": len(newly_verified),
            },
        )
        logger.info(
            "[Orchestrator] Found %s jobs, %s new, %s eligible after prefilter (%s skipped).",
            len(merged_jobs),
            len(new_jobs),
            len(eligible_jobs),
            prefilter_skipped,
        )

        if not eligible_jobs:
            self._emit(
                on_progress,
                "status",
                {"message": "Nessun annuncio idoneo dopo i criteri fondamentali."},
            )
        else:
            self._emit(
                on_progress,
                "status",
                {"message": f"Analisi AI su {len(eligible_jobs)} annunci (pre-filtrati)..."},
            )

        match_results: list[MatchResult] = []
        promoted: list[MatchResult] = []
        total_jobs = len(eligible_jobs)

        for index, job in enumerate(eligible_jobs, start=1):
            self._emit(
                on_progress,
                "analyzing",
                {
                    "current": index,
                    "total": total_jobs,
                    "title": job.title,
                    "company": job.company,
                },
            )
            result = await self.ai_matcher.match(job, profile)
            match_results.append(result)
            self.memory.mark_seen(result.job.dedup_key)
            self.memory.save()

            is_promoted = result.approved and result.match_score >= self.score_threshold
            self._emit(
                on_progress,
                "match",
                {
                    "current": index,
                    "result": result.model_dump(mode="json"),
                    "promoted": is_promoted,
                },
            )

            if is_promoted:
                promoted.append(result)
                self.memory.save_match(result)
                self.memory.save()
                ScanResult(
                    matches=promoted.copy(),
                    total_found=len(merged_jobs),
                    total_analyzed=len(match_results),
                    total_promoted=len(promoted),
                    total_prefilter_skipped=prefilter_skipped,
                ).save(self.scan_results_path)
                self._emit(
                    on_progress,
                    "promoted",
                    {"result": result.model_dump(mode="json")},
                )

        promoted.sort(key=lambda item: item.match_score, reverse=True)
        self.memory.save()
        self._emit(
            on_progress,
            "phase_done",
            {
                "phase": "ai_analysis",
                "message": (
                    f"Analisi AI completata: {len(match_results)} analizzati, "
                    f"{len(promoted)} promossi."
                ),
            },
        )

        scan_result = ScanResult(
            matches=promoted,
            total_found=len(merged_jobs),
            total_analyzed=len(match_results),
            total_promoted=len(promoted),
            total_prefilter_skipped=prefilter_skipped,
        )
        scan_result.save(self.scan_results_path)
        if self.scan_history_path and scan_result.matches:
            ScanHistoryStore(self.scan_history_path).append(scan_result)
        self._emit(
            on_progress,
            "complete",
            {
                "scan_result": scan_result.model_dump(mode="json"),
                "provider_stats": self.search_router.get_usage_stats(),
            },
        )
        logger.info(
            "[Orchestrator] Scan complete. Promoted %s/%s analyzed jobs.",
            scan_result.total_promoted,
            scan_result.total_analyzed,
        )
        return scan_result

    def _job_richness(self, job: JobPosting) -> int:
        score = len(job.description or "")
        if job.source in {"lever", "greenhouse"}:
            score += 10_000
        return score

    def _deduplicate_jobs(self, jobs: list[JobPosting]) -> list[JobPosting]:
        unique: dict[str, JobPosting] = {}
        for job in jobs:
            existing = unique.get(job.dedup_key)
            if existing is None or self._job_richness(job) > self._job_richness(existing):
                unique[job.dedup_key] = job
        return list(unique.values())

    async def run_accurate_match(
        self,
        job: JobPosting,
        profile: UserProfile,
    ) -> tuple[MatchResult, str]:
        """Fetch full job description from the posting page, then re-run AI matching."""
        job = _coerce_job_posting(job)
        previous_len = len((job.description or "").strip())
        enricher = JobTitleEnricher()
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            enriched_job, description_enriched, page_fetched = (
                await enricher.enrich_for_accurate_match(client, job)
            )
        enriched_job = _coerce_job_posting(enriched_job)

        result = await self.ai_matcher.match(enriched_job, profile)
        if description_enriched:
            result = result.model_copy(update={"matched_with_full_description": True})

        new_len = len((enriched_job.description or "").strip())
        score_text = f"{result.match_score:.1f}/10"
        if not page_fetched:
            status = (
                f"Impossibile leggere la pagina dell'annuncio; "
                f"match su estratto disponibile: {score_text}."
            )
        elif description_enriched:
            status = (
                f"Descrizione arricchita ({previous_len} → {new_len} caratteri), "
                f"match ricalcolato: {score_text}."
            )
        elif new_len > previous_len:
            status = (
                f"Descrizione leggermente ampliata ({previous_len} → {new_len} caratteri), "
                f"match ricalcolato: {score_text}."
            )
        else:
            status = (
                f"Descrizione già completa ({new_len} caratteri), "
                f"match ricalcolato: {score_text}."
            )

        return result, status


def run_accurate_match_sync(
    job: JobPosting,
    profile: UserProfile,
    *,
    memory_path: Path | str | None = None,
    scan_results_path: Path | str | None = None,
    discovered_companies_path: Path | str | None = None,
) -> tuple[MatchResult, str]:
    memory = JobMemory(memory_path) if memory_path else JobMemory()
    orchestrator = JobHunterOrchestrator(
        memory=memory,
        scan_results_path=scan_results_path or DEFAULT_SCAN_RESULTS_PATH,
        discovered_companies_path=discovered_companies_path,
    )
    return asyncio.run(orchestrator.run_accurate_match(job, profile))


def run_scan_sync(
    profile: UserProfile,
    on_progress: ProgressCallback | None = None,
    *,
    memory_path: Path | str | None = None,
    scan_results_path: Path | str | None = None,
    scan_history_path: Path | str | None = None,
    discovered_companies_path: Path | str | None = None,
) -> ScanResult:
    memory = JobMemory(memory_path) if memory_path else JobMemory()
    orchestrator = JobHunterOrchestrator(
        memory=memory,
        scan_results_path=scan_results_path or DEFAULT_SCAN_RESULTS_PATH,
        scan_history_path=scan_history_path,
        discovered_companies_path=discovered_companies_path,
    )
    return asyncio.run(orchestrator.run_scan(profile, on_progress=on_progress))
