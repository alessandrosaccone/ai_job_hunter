from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agents.job_title_enricher import (
    JobTitleEnricher,
    needs_company_enrichment,
    needs_title_enrichment,
    refine_job_metadata,
    title_from_reasoning,
)
from models.job import ScanResult

load_dotenv()


async def _repair_scan(
    scan: ScanResult,
    enricher: JobTitleEnricher,
    client: httpx.AsyncClient,
) -> tuple[int, int]:
    updated = 0
    total = len(scan.matches)
    for index, match in enumerate(scan.matches):
        job = refine_job_metadata(match.job)
        if needs_title_enrichment(job.title) or needs_company_enrichment(job.company):
            job = await enricher.enrich(client, job)
        if needs_title_enrichment(job.title):
            fallback = title_from_reasoning(match.reasoning)
            if fallback:
                job = job.model_copy(update={"title": fallback})
        if job.title != match.job.title or job.company != match.job.company:
            updated += 1
        scan.matches[index] = match.model_copy(update={"job": job})
    return updated, total


async def repair_scan_results(path: Path) -> None:
    scan = ScanResult.load(path)
    if not scan:
        print(f"No scan results at {path}")
        return

    enricher = JobTitleEnricher()
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
        updated, total = await _repair_scan(scan, enricher, client)

    scan.save(path)
    print(f"Updated {updated}/{total} matches in {path}")


async def repair_scan_history(path: Path) -> None:
    if not path.exists():
        print(f"No scan history at {path}")
        return

    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    scans = payload.get("scans", [])
    if not scans:
        print(f"No scans to repair in {path}")
        return

    enricher = JobTitleEnricher()
    total_updated = 0
    total_matches = 0

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
        for index, item in enumerate(scans):
            scan = ScanResult.model_validate(item)
            updated, total = await _repair_scan(scan, enricher, client)
            total_updated += updated
            total_matches += total
            scans[index] = scan.model_dump(mode="json")

    payload["scans"] = scans
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {total_updated}/{total_matches} matches in {path}")


if __name__ == "__main__":
    asyncio.run(repair_scan_results(Path("data/profiles/io/scan_results.json")))
    asyncio.run(repair_scan_history(Path("data/profiles/io/scan_history.json")))
