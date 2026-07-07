from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from models.job import MatchResult
from storage.memory import JobMemory

logger = logging.getLogger(__name__)

DEFAULT_SAVED_PATH = Path("data/saved_jobs.json")


class SavedApplication(BaseModel):
    match: MatchResult
    saved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SavedJobsStore:
    def __init__(
        self,
        path: Path | str = DEFAULT_SAVED_PATH,
        memory: JobMemory | None = None,
    ) -> None:
        self.path = Path(path)
        self.memory = memory or JobMemory()
        self.applications: list[SavedApplication] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            self.applications = [
                SavedApplication.model_validate(item) for item in data.get("applications", [])
            ]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[SavedJobsStore] Failed to load saved jobs: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "applications": [app.model_dump(mode="json") for app in self.applications],
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def is_saved(self, dedup_key: str) -> bool:
        key = dedup_key.lower()
        return any(app.match.job.dedup_key.lower() == key for app in self.applications)

    def add(self, result: MatchResult) -> bool:
        if self.is_saved(result.job.dedup_key):
            return False
        self.applications.append(SavedApplication(match=result))
        self.memory.mark_seen(result.job.dedup_key)
        self.memory.save()
        self.save()
        return True

    def remove(self, dedup_key: str) -> bool:
        key = dedup_key.lower()
        before = len(self.applications)
        self.applications = [
            app for app in self.applications if app.match.job.dedup_key.lower() != key
        ]
        if len(self.applications) == before:
            return False
        self.save()
        return True

    def list_sorted(self) -> list[SavedApplication]:
        return sorted(self.applications, key=lambda app: app.saved_at, reverse=True)
