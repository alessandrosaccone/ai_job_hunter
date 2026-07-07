from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from models.job import ScanResult
from storage.memory import JobMemory

logger = logging.getLogger(__name__)

DEFAULT_SCAN_HISTORY_PATH = Path("data/scan_history.json")
ROME = ZoneInfo("Europe/Rome")

ITALIAN_WEEKDAYS = [
    "lunedì",
    "martedì",
    "mercoledì",
    "giovedì",
    "venerdì",
    "sabato",
    "domenica",
]
ITALIAN_MONTHS = [
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
]


def format_italian_date(day: date) -> str:
    weekday = ITALIAN_WEEKDAYS[day.weekday()]
    month = ITALIAN_MONTHS[day.month - 1]
    return f"{weekday} {day.day} {month} {day.year}"


class ScanHistoryStore:
    def __init__(self, path: Path | str = DEFAULT_SCAN_HISTORY_PATH) -> None:
        self.path = Path(path)
        self.scans: list[ScanResult] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[ScanHistoryStore] Failed to load history: %s", exc)
            return

        self.scans = [ScanResult.model_validate(item) for item in data.get("scans", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self.scans, key=lambda scan: scan.scanned_at, reverse=True)
        payload = {"scans": [scan.model_dump(mode="json") for scan in ordered]}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def append(self, scan: ScanResult) -> None:
        if not scan.matches:
            return
        self.scans.append(scan)
        self.save()

    def total_matches(self) -> int:
        return sum(len(scan.matches) for scan in self.scans)

    def group_by_day(self, tz: ZoneInfo = ROME) -> list[tuple[date, list[ScanResult]]]:
        by_day: dict[date, list[ScanResult]] = defaultdict(list)
        for scan in self.scans:
            if not scan.matches:
                continue
            day = scan.scanned_at.astimezone(tz).date()
            by_day[day].append(scan)

        grouped: list[tuple[date, list[ScanResult]]] = []
        for day, scans in by_day.items():
            ordered_scans = sorted(scans, key=lambda item: item.scanned_at, reverse=True)
            grouped.append((day, ordered_scans))
        grouped.sort(key=lambda item: item[0], reverse=True)
        return grouped

    def migrate_if_needed(
        self,
        memory_path: Path | str,
        latest_scan_path: Path | str,
    ) -> None:
        if self.scans:
            return

        migrated = False

        latest_scan = ScanResult.load(latest_scan_path)
        if latest_scan and latest_scan.matches:
            self.scans.append(latest_scan)
            migrated = True

        memory = JobMemory(memory_path)
        if memory.notified_matches:
            known_keys = {
                match.job.dedup_key.lower()
                for scan in self.scans
                for match in scan.matches
            }
            legacy_matches = [
                match
                for match in memory.notified_matches
                if match.job.dedup_key.lower() not in known_keys
            ]
            if legacy_matches:
                scanned_at = _legacy_timestamp(memory_path)
                self.scans.append(
                    ScanResult(
                        matches=legacy_matches,
                        scanned_at=scanned_at,
                        total_found=0,
                        total_analyzed=len(legacy_matches),
                        total_promoted=len(legacy_matches),
                        total_prefilter_skipped=0,
                    ),
                )
                migrated = True

        if migrated:
            self.save()


def _legacy_timestamp(memory_path: Path | str) -> datetime:
    path = Path(memory_path)
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc)
