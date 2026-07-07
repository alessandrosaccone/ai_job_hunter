from __future__ import annotations

import json
import logging
from pathlib import Path

from models.job import JobPosting, MatchResult

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("data/memory.json")


class JobMemory:
    def __init__(self, path: Path | str = DEFAULT_MEMORY_PATH) -> None:
        self.path = Path(path)
        self.seen_urls: set[str] = set()
        self.notified_matches: list[MatchResult] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[JobMemory] Failed to load memory file: %s", exc)
            return

        self.seen_urls = {url.lower() for url in data.get("seen_urls", [])}
        self.notified_matches = [
            MatchResult.model_validate(item) for item in data.get("notified_matches", [])
        ]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seen_urls": sorted(self.seen_urls),
            "notified_matches": [match.model_dump(mode="json") for match in self.notified_matches],
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def is_seen(self, url: str) -> bool:
        return url.lower() in self.seen_urls

    def mark_seen(self, url: str) -> None:
        self.seen_urls.add(url.lower())

    def get_new_jobs(self, jobs: list[JobPosting]) -> list[JobPosting]:
        return [job for job in jobs if not self.is_seen(job.dedup_key)]

    def save_match(self, result: MatchResult) -> None:
        self.notified_matches.append(result)
        self.mark_seen(result.job.dedup_key)
