from __future__ import annotations

import json
import logging
import os
import re

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents.job_prefilter import _extract_salary_range
from agents.search_providers.router import JobSearchRouter
from models.job import JobPosting
from models.user_profile import UserProfile, read_uses_web_search

logger = logging.getLogger(__name__)

SALARY_QUERY_TEMPLATES = (
    "RAL stipendio {title} {company} {location}",
    "{title} salary {company} site:glassdoor.com",
    "{title} {company} salary site:levels.fyi",
    "stipendio medio {title} {location} EUR",
)
LEVEL_TERMS: dict[str, tuple[str, ...]] = {
    "internship": ("intern", "internship", "stage", "trainee"),
    "graduate": ("graduate", "new grad", "entry", "junior"),
    "entry": ("entry", "junior", "associate"),
    "mid": ("mid", "intermediate", "ii"),
    "senior": ("senior", "sr", "staff", "principal"),
    "manager": ("manager", "lead", "head", "director", "vp"),
}

SALARY_SYNTHESIS_SYSTEM = """Sei un analista compensazioni. Ti vengono forniti estratti da ricerca web reale (Glassdoor, Levels.fyi, Indeed, ecc.).

Regole rigide:
- NON inventare cifre. Usa SOLO importi esplicitamente presenti negli estratti o nel campo extracted_salary_from_snippets.
- Se non ci sono cifre affidabili negli estratti, imposta estimated_salary_eur a null.
- Spiega in italiano quali fonti hai usato e quanto sono affidabili.

Rispondi SOLO con json valido:
{
  "estimated_salary_eur": "42.000-55.000 EUR",
  "research_summary": "Breve spiegazione in italiano con fonti e affidabilità.",
  "confidence": "low|medium|high",
  "sources": ["url fonte"]
}
"""


class SalaryResearchResult(BaseModel):
    estimated_salary_eur: str | None = None
    research_summary: str | None = None
    confidence: str = "low"
    sources: list[str] = Field(default_factory=list)


class SalaryResearchResponse(BaseModel):
    estimated_salary_eur: str | None = None
    research_summary: str | None = None
    confidence: str = "low"
    sources: list[str] = Field(default_factory=list)

    @field_validator("estimated_salary_eur", mode="before")
    @classmethod
    def coerce_estimated_salary(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in {"null", "none", "n/a", ""}:
                return None
            return cleaned
        if isinstance(value, (int, float)):
            amount = int(value)
            formatted = f"{amount:,}".replace(",", ".")
            return f"{formatted} EUR"
        return str(value)


class SalaryResearcher:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 90.0,
        search_router: JobSearchRouter | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout_seconds = timeout_seconds
        self.search_router = search_router or JobSearchRouter()
        self._client: AsyncOpenAI | None = None
        self._cache: dict[str, SalaryResearchResult] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "your_deepseek_api_key_here"

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def research(self, job: JobPosting, profile: UserProfile) -> SalaryResearchResult | None:
        if not read_uses_web_search(profile):
            result = SalaryResearchResult(
                research_summary=(
                    "Modalità senza ricerche web: RAL non cercata online. "
                    "Usa la RAL indicata nell'annuncio se presente."
                ),
                confidence="low",
            )
            return result

        cache_key = f"{job.company.lower()}|{job.title.lower()}|{job.location.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        snippets, provider, source_urls = await self._gather_web_evidence(job, profile)
        if not snippets:
            result = SalaryResearchResult(
                research_summary=(
                    "RAL non indicata nell'annuncio. Ricerca web (SerpApi/Serper/DataForSEO/Apify/ScraperAPI) "
                    "senza risultati utili per questo ruolo e azienda."
                ),
                confidence="low",
            )
            self._cache[cache_key] = result
            return result

        extracted_salary, extracted_sources = _extract_salary_from_snippets(
            snippets,
            job=job,
            profile=profile,
        )
        all_sources = list(dict.fromkeys([*extracted_sources, *source_urls]))[:5]

        if extracted_salary:
            summary = await self._summarize_evidence(
                job,
                profile,
                snippets,
                provider,
                extracted_salary,
                confidence="high" if len(extracted_sources) >= 2 else "medium",
            )
            result = SalaryResearchResult(
                estimated_salary_eur=extracted_salary,
                research_summary=summary or (
                    f"RAL non indicata nell'annuncio. Range ricavato da fonti web "
                    f"({provider or 'ricerca'}): {extracted_salary}."
                ),
                confidence="high" if len(extracted_sources) >= 2 else "medium",
                sources=all_sources,
            )
        elif self.is_configured:
            synthesized = await self._synthesize_from_snippets(
                job,
                profile,
                snippets,
                all_sources,
                provider,
            )
            result = SalaryResearchResult(
                estimated_salary_eur=synthesized.estimated_salary_eur,
                research_summary=synthesized.research_summary,
                confidence=synthesized.confidence,
                sources=synthesized.sources or all_sources,
            )
        else:
            result = SalaryResearchResult(
                research_summary=(
                    f"RAL non indicata nell'annuncio. Trovati {len(snippets)} risultati web "
                    f"({provider or 'n/d'}) ma nessuna cifra leggibile negli estratti."
                ),
                confidence="low",
                sources=all_sources,
            )

        self._cache[cache_key] = result
        return result

    async def _gather_web_evidence(
        self,
        job: JobPosting,
        profile: UserProfile,
    ) -> tuple[list[dict[str, str]], str | None, list[str]]:
        location_targets = profile.search_location_targets()
        seen_links: set[str] = set()
        snippets: list[dict[str, str]] = []
        provider_used: str | None = None

        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for location_target in location_targets:
                for template in SALARY_QUERY_TEMPLATES:
                    query = template.format(
                        title=job.title,
                        company=job.company,
                        location=location_target,
                    ).strip()
                    results, provider = await self.search_router.search(
                        client,
                        "google",
                        query,
                        location_target,
                        exclude_providers={"deepseek"},
                    )
                    if provider and not provider_used:
                        provider_used = provider

                    for item in results:
                        link = _item_link(item)
                        if not link or link in seen_links:
                            continue
                        seen_links.add(link)
                        snippets.append(
                            {
                                "title": str(item.get("title", "")),
                                "link": link,
                                "snippet": str(item.get("description", "") or item.get("snippet", "")),
                            },
                        )
                        if len(snippets) >= 12:
                            break
                    if len(snippets) >= 12:
                        break
                if len(snippets) >= 12:
                    break

        source_urls = [item["link"] for item in snippets if item.get("link")]
        if snippets:
            logger.info(
                "[SalaryResearcher] %s snippet raccolti via %s per %s @ %s",
                len(snippets),
                provider_used or "unknown",
                job.title,
                job.company,
            )
        return snippets, provider_used, source_urls

    async def _summarize_evidence(
        self,
        job: JobPosting,
        profile: UserProfile,
        snippets: list[dict[str, str]],
        provider: str | None,
        extracted_salary: str,
        *,
        confidence: str,
    ) -> str | None:
        if not self.is_configured:
            return None

        evidence_block = "\n".join(
            f"- {item['title']}: {item['snippet'][:200]}" for item in snippets[:5]
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=400,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Scrivi 2-3 frasi in italiano che spiegano la stima RAL trovata su fonti web. "
                            "NON cambiare il range numerico fornito. Cita le fonti."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Ruolo: {job.title} @ {job.company}\n"
                            f"Range da estratti web: {extracted_salary}\n"
                            f"Provider ricerca: {provider or 'n/d'}\n"
                            f"Affidabilità: {confidence}\n\n"
                            f"Estratti:\n{evidence_block}"
                        ),
                    },
                ],
            )
            return (completion.choices[0].message.content or "").strip() or None
        except Exception as exc:
            logger.warning("[SalaryResearcher] Summary failed: %s", exc)
            return None

    async def _synthesize_from_snippets(
        self,
        job: JobPosting,
        profile: UserProfile,
        snippets: list[dict[str, str]],
        source_urls: list[str],
        provider: str | None,
    ) -> SalaryResearchResponse:
        location = job.location or profile.search_location_query()
        evidence_block = "\n\n".join(
            f"- Titolo: {item['title']}\n  URL: {item['link']}\n  Estratto: {item['snippet']}"
            for item in snippets[:10]
        )
        user_prompt = (
            f"Analizza SOLO gli estratti web per:\n"
            f"- Ruolo: {job.title}\n"
            f"- Ruoli target utente: {', '.join(profile.target_roles)}\n"
            f"- Azienda: {job.company}\n"
            f"- Località: {location}\n"
            f"- Livello: {profile.experience_level}\n\n"
            f"Provider ricerca: {provider or 'n/d'}\n"
            f"Estratti:\n{evidence_block}\n\n"
            "Considera solo cifre coerenti con ruolo/livello richiesti; ignora seniority troppo diverse "
            "(es. architect/staff se il ruolo è junior/entry). "
            "Se non trovi cifre esplicite e coerenti negli estratti, estimated_salary_eur deve essere null."
        )

        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SALARY_SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = _parse_salary_json(content)

        if not parsed.estimated_salary_eur:
            parsed.confidence = "low"
            if not parsed.research_summary:
                parsed.research_summary = (
                    "RAL non indicata nell'annuncio. Fonti web consultate ma nessuna cifra "
                    "affidabile negli estratti."
                )

        if parsed.research_summary and "non indicata" not in parsed.research_summary.lower():
            parsed.research_summary = f"RAL non indicata nell'annuncio. {parsed.research_summary}"
        if source_urls and not parsed.sources:
            parsed.sources = source_urls[:5]
        return parsed


def _extract_salary_from_snippets(
    snippets: list[dict[str, str]],
    *,
    job: JobPosting,
    profile: UserProfile,
) -> tuple[str | None, list[str]]:
    scored_ranges: list[tuple[int, tuple[int, int], str]] = []
    sources: list[str] = []
    job_tokens = _tokenize(f"{job.title} {' '.join(profile.target_roles)}")
    level_terms = set(LEVEL_TERMS.get(profile.experience_level, ()))

    for item in snippets:
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", ""))
        text = f"{title} {snippet}"
        parsed = _extract_salary_range(text)
        if not parsed:
            continue
        context_tokens = _tokenize(text)
        overlap = len(job_tokens.intersection(context_tokens))
        level_boost = 1 if (level_terms and any(term in text.lower() for term in level_terms)) else 0
        source_boost = 1 if any(site in (item.get("link", "") or "") for site in ("glassdoor", "levels.fyi", "indeed")) else 0
        score = overlap + level_boost + source_boost
        link = str(item.get("link", ""))
        scored_ranges.append((score, parsed, link))
        link = item.get("link")
        if link:
            sources.append(str(link))

    if not scored_ranges:
        return None, []

    scored_ranges.sort(key=lambda item: item[0], reverse=True)
    top_score = scored_ranges[0][0]
    if top_score > 0:
        selected = [item for item in scored_ranges if item[0] >= max(1, top_score - 1)]
    else:
        selected = scored_ranges[:4]

    ranges = [item[1] for item in selected]
    lows = sorted(item[0] for item in ranges)
    highs = sorted(item[1] for item in ranges)

    # Remove outliers when enough evidence exists.
    if len(lows) >= 4:
        lows = lows[1:-1]
    if len(highs) >= 4:
        highs = highs[1:-1]

    low = min(lows)
    high = max(highs)
    if high > low * 3 and len(highs) > 1:
        high = highs[len(highs) // 2]

    selected_sources = [item[2] for item in selected if item[2]]
    deduped_sources = list(dict.fromkeys(selected_sources + sources))
    return _format_salary_range(low, high), deduped_sources


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
        if token not in {"salary", "stipendio", "eur", "euro", "jobs", "job"}
    }


def _format_salary_range(low: int, high: int) -> str:
    def fmt(value: int) -> str:
        return f"{value:,}".replace(",", ".")

    if low == high:
        return f"{fmt(low)} EUR"
    return f"{fmt(low)}-{fmt(high)} EUR"


def _item_link(item: dict) -> str:
    link = item.get("link")
    if link:
        return str(link)
    apply_options = item.get("apply_options", [])
    if apply_options and isinstance(apply_options[0], dict):
        return str(apply_options[0].get("link", ""))
    return ""


def _parse_salary_json(text: str) -> SalaryResearchResponse:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty salary research response")

    try:
        return SalaryResearchResponse.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return SalaryResearchResponse.model_validate(json.loads(match.group(0)))
