from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.deepseek_web_search import deepseek_web_search
from agents.duckduckgo_search import search_web as ddg_search_web

logger = logging.getLogger(__name__)

JOB_SEARCH_SYSTEM = """Sei un assistente per la ricerca di offerte di lavoro online.
Usa la ricerca web per trovare annunci REALI e attuali.

Regole:
- Cerca su LinkedIn Jobs, Indeed, StepStone, Lever, Greenhouse, Workday e siti careers.
- Includi solo URL che sembrano pagine di singole offerte o listing affidabili.
- Rispondi SOLO con json valido:
{
  "results": [
    {
      "title": "Software Engineer",
      "company": "Acme",
      "url": "https://...",
      "description": "breve estratto",
      "location": "Milano"
    }
  ]
}
"""


class JobSearchHit(BaseModel):
    title: str
    company: str = ""
    url: str
    description: str = ""
    location: str = ""


class JobSearchBatch(BaseModel):
    results: list[JobSearchHit] = Field(default_factory=list)


def hits_to_serpapi_shape(hits: list[JobSearchHit], source_engine: str) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for hit in hits:
        shaped.append(
            {
                "source_engine": source_engine,
                "title": hit.title,
                "company_name": hit.company,
                "location": hit.location,
                "description": hit.description,
                "apply_options": [{"link": hit.url}],
                "job_id": hit.url,
                "link": hit.url,
            },
        )
    return shaped


def organic_to_serpapi_shape(items: list[dict[str, str]], source_engine: str) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for item in items:
        link = item.get("link", "")
        if not link:
            continue
        shaped.append(
            {
                "source_engine": source_engine,
                "title": item.get("title", ""),
                "company_name": "",
                "location": "",
                "description": item.get("snippet", ""),
                "apply_options": [{"link": link}],
                "job_id": link,
                "link": link,
            },
        )
    return shaped


async def search_jobs_duckduckgo(query: str) -> list[dict[str, Any]]:
    organic = await ddg_search_web(f"{query} jobs hiring", max_results=10)
    return organic_to_serpapi_shape(organic, "duckduckgo")


async def search_jobs_deepseek(query: str, location: str) -> list[dict[str, Any]]:
    user_prompt = (
        f"Trova annunci di lavoro attuali per questa ricerca:\n"
        f"Query: {query}\n"
        f"Area: {location}\n"
        "Restituisci fino a 8 risultati pertinenti."
    )
    try:
        text, _sources = await deepseek_web_search(
            system_prompt=JOB_SEARCH_SYSTEM,
            user_prompt=user_prompt,
            max_uses=3,
        )
        batch = _parse_job_batch(text)
        return hits_to_serpapi_shape(batch.results, "deepseek_web")
    except Exception as exc:
        logger.warning("[JobSearchFallback] DeepSeek web search failed: %s", exc)
        return []


def _parse_job_batch(text: str) -> JobSearchBatch:
    cleaned = text.strip()
    if not cleaned:
        return JobSearchBatch()
    try:
        return JobSearchBatch.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return JobSearchBatch()
        return JobSearchBatch.model_validate(json.loads(match.group(0)))
