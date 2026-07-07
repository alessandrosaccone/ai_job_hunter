from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator

JobSource = Literal["lever", "greenhouse", "workday", "serpapi"]
DEFAULT_SCAN_RESULTS_PATH = Path("data/scan_results.json")


class JobPosting(BaseModel):
    id: str
    title: str
    company: str
    url: str
    source: JobSource
    location: str = ""
    description: str = ""
    salary_hint: str | None = None
    work_mode_hint: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        normalized = parsed._replace(fragment="", query="")
        return urlunparse(normalized).rstrip("/")

    @property
    def dedup_key(self) -> str:
        return self.url.lower()


class MatchResult(BaseModel):
    job: JobPosting
    match_score: float = Field(ge=0, le=10)
    approved: bool
    reasoning: str


class ScanResult(BaseModel):
    matches: list[MatchResult] = Field(default_factory=list)
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_found: int = 0
    total_analyzed: int = 0
    total_promoted: int = 0
    total_prefilter_skipped: int = 0

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SCAN_RESULTS_PATH) -> ScanResult | None:
        results_path = Path(path)
        if not results_path.exists():
            return None
        with results_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.model_validate(data)

    def save(self, path: Path | str = DEFAULT_SCAN_RESULTS_PATH) -> None:
        results_path = Path(path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(self.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)
