from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.ai_matcher import AIMatcher
from agents.job_prefilter import filter_jobs_for_ai
from agents.location_matcher import LocationMatcher
from agents.role_matcher import RoleMatcher
from agents.startup_discoverer import StartupDiscoverer
from agents.target_hunter import TargetHunter
from models.job import JobPosting, MatchResult, ScanResult
from models.user_profile import UserProfile
from storage.memory import JobMemory

logger = logging.getLogger(__name__)

DEFAULT_SCAN_RESULTS_PATH = Path("data/scan_results.json")
ProgressCallback = Callable[[str, dict[str, Any]], None]


class JobHunterOrchestrator:
    def __init__(
        self,
        memory: JobMemory | None = None,
        target_hunter: TargetHunter | None = None,
        startup_discoverer: StartupDiscoverer | None = None,
        ai_matcher: AIMatcher | None = None,
        location_matcher: LocationMatcher | None = None,
        role_matcher: RoleMatcher | None = None,
        score_threshold: float | None = None,
        scan_results_path: Path | str = DEFAULT_SCAN_RESULTS_PATH,
    ) -> None:
        load_dotenv()
        self.memory = memory or JobMemory()
        self.target_hunter = target_hunter or TargetHunter()
        self.startup_discoverer = startup_discoverer or StartupDiscoverer()
        self.ai_matcher = ai_matcher or AIMatcher()
        self.location_matcher = location_matcher or LocationMatcher()
        self.role_matcher = role_matcher or RoleMatcher()
        self.score_threshold = score_threshold or float(os.getenv("MATCH_SCORE_THRESHOLD", "7"))
        self.scan_results_path = Path(scan_results_path)

    def _emit(self, callback: ProgressCallback | None, event: str, payload: dict[str, Any]) -> None:
        if callback:
            callback(event, payload)

    async def run_scan(
        self,
        profile: UserProfile,
        on_progress: ProgressCallback | None = None,
    ) -> ScanResult:
        logger.info("[Orchestrator] Starting scan for roles: %s", ", ".join(profile.target_roles))
        self._emit(
            on_progress,
            "status",
            {"message": "Avvio raccolta annunci dagli agenti..."},
        )

        async def run_named(agent_name: str, coro: Any) -> tuple[str, list[JobPosting]]:
            self._emit(
                on_progress,
                "status",
                {"message": f"{agent_name} in esecuzione..."},
            )
            jobs = await coro
            self._emit(
                on_progress,
                "agent_done",
                {"agent": agent_name, "count": len(jobs)},
            )
            return agent_name, jobs

        agent_results = await asyncio.gather(
            run_named("Target Hunter", self.target_hunter.safe_run(profile)),
            run_named("Startup Discoverer", self.startup_discoverer.safe_run(profile)),
        )
        target_jobs = next(jobs for name, jobs in agent_results if name == "Target Hunter")
        startup_jobs = next(jobs for name, jobs in agent_results if name == "Startup Discoverer")

        merged_jobs = self._deduplicate_jobs(target_jobs + startup_jobs)
        new_jobs = self.memory.get_new_jobs(merged_jobs)
        eligible_jobs, prefilter_skipped = await filter_jobs_for_ai(
            new_jobs,
            profile,
            location_matcher=self.location_matcher,
            role_matcher=self.role_matcher,
            on_progress=on_progress,
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
                self._emit(
                    on_progress,
                    "promoted",
                    {"result": result.model_dump(mode="json")},
                )

        promoted.sort(key=lambda item: item.match_score, reverse=True)
        self.memory.save()

        scan_result = ScanResult(
            matches=promoted,
            total_found=len(merged_jobs),
            total_analyzed=len(match_results),
            total_promoted=len(promoted),
            total_prefilter_skipped=prefilter_skipped,
        )
        scan_result.save(self.scan_results_path)
        self._emit(
            on_progress,
            "complete",
            {"scan_result": scan_result.model_dump(mode="json")},
        )
        logger.info(
            "[Orchestrator] Scan complete. Promoted %s/%s analyzed jobs.",
            scan_result.total_promoted,
            scan_result.total_analyzed,
        )
        return scan_result

    def _deduplicate_jobs(self, jobs: list[JobPosting]) -> list[JobPosting]:
        unique: dict[str, JobPosting] = {}
        for job in jobs:
            unique.setdefault(job.dedup_key, job)
        return list(unique.values())


def run_scan_sync(
    profile: UserProfile,
    on_progress: ProgressCallback | None = None,
) -> ScanResult:
    orchestrator = JobHunterOrchestrator()
    return asyncio.run(orchestrator.run_scan(profile, on_progress=on_progress))
