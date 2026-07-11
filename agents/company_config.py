from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WORKDAY_COMPANIES_PATH = Path("config/workday_companies.json")


def _company_matches_field(company: dict[str, Any], career_field: str | None) -> bool:
    if not career_field:
        return True
    return career_field in company.get("fields", ["tech"])


def load_workday_companies(*, career_field: str | None = None) -> list[dict[str, Any]]:
    if not WORKDAY_COMPANIES_PATH.exists():
        return []
    with WORKDAY_COMPANIES_PATH.open(encoding="utf-8") as handle:
        companies = json.load(handle)
    if not isinstance(companies, list):
        return []
    return [company for company in companies if _company_matches_field(company, career_field)]


def company_storage_key(company: dict[str, Any]) -> str:
    ats = str(company.get("ats", "")).lower()
    slug = str(company.get("slug", "")).lower()
    if ats == "workday":
        tenant = str(company.get("tenant", "")).lower()
        return f"workday:{tenant}:{slug}"
    return f"{ats}:{slug}"


def load_target_companies(
    companies_path: Path | str,
    *,
    career_field: str | None = None,
) -> list[dict[str, Any]]:
    base_path = Path(companies_path)
    field_path = (
        base_path.parent / "target_companies" / f"{career_field}.json"
        if career_field
        else None
    )

    companies: list[dict[str, Any]] = []

    # Keep the historical target_companies.json as the tech/engineering list.
    if career_field and career_field != "tech" and field_path and field_path.exists():
        with field_path.open(encoding="utf-8") as handle:
            companies.extend(json.load(handle))
    else:
        if base_path.exists():
            with base_path.open(encoding="utf-8") as handle:
                companies.extend(json.load(handle))

        if field_path and field_path.exists():
            with field_path.open(encoding="utf-8") as handle:
                companies.extend(json.load(handle))

    merged: dict[str, dict[str, Any]] = {
        company_storage_key(company): company for company in companies
    }
    for company in load_workday_companies(career_field=career_field):
        merged.setdefault(company_storage_key(company), company)

    return list(merged.values())
