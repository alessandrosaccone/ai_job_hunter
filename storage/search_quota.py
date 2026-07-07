from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

QUOTA_PATH = Path("data/search_quota.json")
LEGACY_SERPAPI_QUOTA_PATH = Path("data/serpapi_quota.json")


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _load_data() -> dict:
    if QUOTA_PATH.exists():
        try:
            with QUOTA_PATH.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {"providers": {}}

    if LEGACY_SERPAPI_QUOTA_PATH.exists():
        try:
            with LEGACY_SERPAPI_QUOTA_PATH.open(encoding="utf-8") as handle:
                legacy = json.load(handle)
            if legacy.get("exhausted") and legacy.get("month") == _current_month():
                return {
                    "providers": {
                        "serpapi": {"month": legacy["month"], "exhausted": True},
                    },
                }
        except (json.JSONDecodeError, OSError):
            pass

    return {"providers": {}}


def _save_data(data: dict) -> None:
    QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUOTA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def is_provider_exhausted(provider: str) -> bool:
    data = _load_data()
    entry = data.get("providers", {}).get(provider, {})
    return bool(entry.get("exhausted")) and entry.get("month") == _current_month()


def mark_provider_exhausted(provider: str) -> None:
    data = _load_data()
    providers = data.setdefault("providers", {})
    providers[provider] = {"month": _current_month(), "exhausted": True}
    _save_data(data)


def clear_provider_exhausted(provider: str | None = None) -> None:
    if provider is None:
        if QUOTA_PATH.exists():
            QUOTA_PATH.unlink()
        if LEGACY_SERPAPI_QUOTA_PATH.exists():
            LEGACY_SERPAPI_QUOTA_PATH.unlink()
        return

    data = _load_data()
    providers = data.get("providers", {})
    providers.pop(provider, None)
    data["providers"] = providers
    _save_data(data)


def exhausted_providers() -> list[str]:
    data = _load_data()
    month = _current_month()
    return [
        name
        for name, entry in data.get("providers", {}).items()
        if entry.get("month") == month and entry.get("exhausted")
    ]


# Backward compatibility
def is_serpapi_marked_exhausted() -> bool:
    return is_provider_exhausted("serpapi")


def mark_serpapi_exhausted() -> None:
    mark_provider_exhausted("serpapi")


def clear_serpapi_exhausted() -> None:
    clear_provider_exhausted("serpapi")
