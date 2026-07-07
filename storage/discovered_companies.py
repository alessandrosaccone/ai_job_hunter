from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _company_key(company: dict[str, Any]) -> str:
    return f"{company.get('ats', '').lower()}:{company.get('slug', '').lower()}"


class DiscoveredCompaniesStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[DiscoveredCompanies] Failed to load %s: %s", self.path, exc)
            return []
        companies = data.get("companies", [])
        return companies if isinstance(companies, list) else []

    def save(self, companies: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"companies": companies}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def merge_new(self, discovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not discovered:
            return self.load()

        existing = self.load()
        by_key = {_company_key(company): company for company in existing}
        now = datetime.now(timezone.utc).isoformat()

        for company in discovered:
            key = _company_key(company)
            entry = {
                **company,
                "discovered_at": company.get("discovered_at", now),
                "source": company.get("source", "startup_discoverer"),
            }
            by_key[key] = entry

        merged = list(by_key.values())
        self.save(merged)
        return merged
