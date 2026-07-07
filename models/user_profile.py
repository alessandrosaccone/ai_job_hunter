from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

WorkMode = Literal["Remote", "Hybrid", "Full-time in office"]
CareerField = Literal[
    "tech",
    "management",
    "sales",
    "design",
    "operations",
    "finance",
    "import_export",
]
ExperienceLevel = Literal[
    "graduate",
    "internship",
    "entry",
    "mid",
    "senior",
    "manager",
]
ExperienceLevelMode = Literal["exact", "or_higher", "all_lower", "or_lower"]
SearchMode = Literal["full", "no_search"]

LEVEL_ORDER: list[ExperienceLevel] = [
    "internship",
    "graduate",
    "entry",
    "mid",
    "senior",
    "manager",
]

DEFAULT_PROFILE_PATH = Path("config/user_profile.json")


class FundamentalCriteria(BaseModel):
    location: bool = False
    target_role: bool = False
    salary: bool = False
    work_mode: bool = False
    experience_level: bool = False


class ExperienceLevelRule(BaseModel):
    mode: ExperienceLevelMode = "or_higher"
    offset: int = Field(default=1, ge=0, le=5)


def allowed_experience_levels(
    user_level: ExperienceLevel,
    rule: ExperienceLevelRule,
) -> set[ExperienceLevel]:
    idx = LEVEL_ORDER.index(user_level)
    last = len(LEVEL_ORDER) - 1

    if rule.mode == "exact":
        indices = {idx}
    elif rule.mode == "all_lower":
        indices = set(range(0, idx + 1))
    elif rule.mode == "or_higher":
        indices = set(range(idx, min(last, idx + rule.offset) + 1))
    elif rule.mode == "or_lower":
        indices = set(range(max(0, idx - rule.offset), idx + 1))
    else:
        indices = {idx}

    return {LEVEL_ORDER[i] for i in indices}


def read_search_mode(profile: UserProfile) -> SearchMode:
    mode = profile.model_dump().get("search_mode", "full")
    return mode if mode in {"full", "no_search"} else "full"


def read_uses_web_search(profile: UserProfile) -> bool:
    return read_search_mode(profile) == "full"


class UserProfile(BaseModel):
    career_field: CareerField = "tech"
    experience_level: ExperienceLevel = "mid"
    education: str = ""
    passions: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    desired_salary_eur: int | None = None
    location: str = ""
    work_mode: WorkMode = "Remote"
    free_text_preferences: str = ""
    fundamental_criteria: FundamentalCriteria = Field(default_factory=FundamentalCriteria)
    experience_level_rule: ExperienceLevelRule = Field(default_factory=ExperienceLevelRule)
    search_mode: SearchMode = "full"

    def uses_web_search(self) -> bool:
        return read_uses_web_search(self)

    def location_places(self) -> list[str]:
        return [place.strip() for place in self.location.split(",") if place.strip()]

    def search_location_targets(self) -> list[str]:
        """Each city/country as its own search target (e.g. Milano, Italy, Spain → 3 queries)."""
        places = self.location_places()
        return places if places else ["Italy"]

    def search_location_query(self) -> str:
        """First search target only — prefer search_location_targets() for queries."""
        return self.search_location_targets()[0]

    def allowed_experience_levels(self) -> set[ExperienceLevel]:
        return allowed_experience_levels(self.experience_level, self.experience_level_rule)

    @field_validator("target_roles")
    @classmethod
    def validate_target_roles(cls, value: list[str]) -> list[str]:
        cleaned = [role.strip() for role in value if role.strip()]
        if not cleaned:
            raise ValueError("At least one target role keyword is required.")
        return cleaned

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE_PATH) -> UserProfile | None:
        profile_path = Path(path)
        if not profile_path.exists():
            return None
        with profile_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if "search_mode" not in data:
            data["search_mode"] = "full"
        return cls.model_validate(data)

    def save(self, path: Path | str = DEFAULT_PROFILE_PATH) -> None:
        profile_path = Path(path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as handle:
            json.dump(self.model_dump(), handle, indent=2, ensure_ascii=False)
