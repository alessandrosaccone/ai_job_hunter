from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_OVERRIDES_PATH = Path("data/salary_overrides.json")


class SalaryOverride(BaseModel):
    salary_eur: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SalaryOverrideStore:
    def __init__(self, path: Path | str = DEFAULT_OVERRIDES_PATH) -> None:
        self.path = Path(path)
        self._overrides: dict[str, SalaryOverride] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            raw = data.get("overrides", {})
            self._overrides = {
                key.lower(): SalaryOverride.model_validate(value)
                for key, value in raw.items()
            }
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("[SalaryOverrideStore] Failed to load overrides: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "overrides": {
                key: entry.model_dump(mode="json")
                for key, entry in sorted(self._overrides.items())
            },
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def get(self, dedup_key: str) -> str | None:
        entry = self._overrides.get(dedup_key.lower())
        return entry.salary_eur if entry else None

    def set(self, dedup_key: str, salary_eur: str) -> None:
        cleaned = salary_eur.strip()
        if not cleaned:
            self.remove(dedup_key)
            return
        self._overrides[dedup_key.lower()] = SalaryOverride(salary_eur=cleaned)
        self.save()

    def remove(self, dedup_key: str) -> bool:
        key = dedup_key.lower()
        if key not in self._overrides:
            return False
        del self._overrides[key]
        self.save()
        return True

    def as_dict(self) -> dict[str, str]:
        return {key: entry.salary_eur for key, entry in self._overrides.items()}
