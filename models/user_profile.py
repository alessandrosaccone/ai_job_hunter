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

DEFAULT_PROFILE_PATH = Path("config/user_profile.json")


class FundamentalCriteria(BaseModel):
    location: bool = False
    target_role: bool = False
    salary: bool = False
    work_mode: bool = False
    experience_level: bool = False


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

    def location_places(self) -> list[str]:
        return [place.strip() for place in self.location.split(",") if place.strip()]

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
        return cls.model_validate(data)

    def save(self, path: Path | str = DEFAULT_PROFILE_PATH) -> None:
        profile_path = Path(path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as handle:
            json.dump(self.model_dump(), handle, indent=2, ensure_ascii=False)
