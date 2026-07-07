from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REGISTRY_PATH = Path("data/profiles/registry.json")
PROFILES_ROOT = Path("data/profiles")

LEGACY_PROFILE_PATH = Path("config/user_profile.json")
LEGACY_MEMORY_PATH = Path("data/memory.json")
LEGACY_SAVED_PATH = Path("data/saved_jobs.json")
LEGACY_SCAN_PATH = Path("data/scan_results.json")


@dataclass(frozen=True)
class ProfilePaths:
    slug: str
    display_name: str
    profile_path: Path
    memory_path: Path
    saved_jobs_path: Path
    scan_results_path: Path
    scan_history_path: Path
    discovered_companies_path: Path


def slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name.strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "profilo"


def _profile_dir(slug: str) -> Path:
    return PROFILES_ROOT / slug


def _paths_for(slug: str, display_name: str) -> ProfilePaths:
    root = _profile_dir(slug)
    return ProfilePaths(
        slug=slug,
        display_name=display_name,
        profile_path=root / "profile.json",
        memory_path=root / "memory.json",
        saved_jobs_path=root / "saved_jobs.json",
        scan_results_path=root / "scan_results.json",
        scan_history_path=root / "scan_history.json",
        discovered_companies_path=root / "discovered_companies.json",
    )


def _load_registry_data() -> dict:
    if not REGISTRY_PATH.exists():
        return {"profiles": [], "last_active": None}
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_registry_data(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def list_profiles() -> list[ProfilePaths]:
    ensure_profiles_ready()
    data = _load_registry_data()
    return [
        _paths_for(item["slug"], item["display_name"])
        for item in data.get("profiles", [])
    ]


def get_profile_paths(slug: str) -> ProfilePaths | None:
    for profile in list_profiles():
        if profile.slug == slug:
            return profile
    return None


def get_last_active_slug() -> str | None:
    ensure_profiles_ready()
    return _load_registry_data().get("last_active")


def set_last_active(slug: str) -> None:
    data = _load_registry_data()
    data["last_active"] = slug
    _save_registry_data(data)


def _unique_slug(base: str) -> str:
    existing = {profile.slug for profile in list_profiles()}
    if base not in existing:
        return base
    index = 2
    while f"{base}-{index}" in existing:
        index += 1
    return f"{base}-{index}"


def _register_profile(slug: str, display_name: str) -> ProfilePaths:
    paths = _paths_for(slug, display_name)
    paths.profile_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_registry_data()
    profiles = data.setdefault("profiles", [])
    if not any(item["slug"] == slug for item in profiles):
        profiles.append({"slug": slug, "display_name": display_name})
    data["last_active"] = slug
    _save_registry_data(data)
    return paths


def create_profile(display_name: str) -> ProfilePaths:
    ensure_profiles_ready()
    cleaned_name = display_name.strip()
    if not cleaned_name:
        raise ValueError("Il nome profilo non può essere vuoto.")

    slug = _unique_slug(slugify(cleaned_name))
    return _register_profile(slug, cleaned_name)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _migrate_legacy_profile() -> None:
    if REGISTRY_PATH.exists():
        return

    default_slug = "predefinito"
    paths = _paths_for(default_slug, "Predefinito")
    paths.profile_path.parent.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(LEGACY_PROFILE_PATH, paths.profile_path)
    _copy_if_exists(LEGACY_MEMORY_PATH, paths.memory_path)
    _copy_if_exists(LEGACY_SAVED_PATH, paths.saved_jobs_path)
    _copy_if_exists(LEGACY_SCAN_PATH, paths.scan_results_path)

    profiles: list[dict[str, str]] = []
    if paths.profile_path.exists() or paths.memory_path.exists():
        profiles.append({"slug": default_slug, "display_name": "Predefinito"})

    _save_registry_data({"profiles": profiles, "last_active": default_slug if profiles else None})


def ensure_profiles_ready() -> None:
    PROFILES_ROOT.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_profile()

    data = _load_registry_data()
    if not data.get("profiles"):
        _register_profile("predefinito", "Predefinito")


def delete_profile(slug: str) -> ProfilePaths | None:
    """Remove profile from registry and delete its data directory."""
    ensure_profiles_ready()
    data = _load_registry_data()
    profiles = data.get("profiles", [])
    if len(profiles) <= 1:
        raise ValueError("Deve restare almeno un profilo.")

    remaining = [item for item in profiles if item["slug"] != slug]
    if len(remaining) == len(profiles):
        return None

    profile_dir = _profile_dir(slug)
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    data["profiles"] = remaining
    if data.get("last_active") == slug:
        data["last_active"] = remaining[0]["slug"]
    _save_registry_data(data)

    return _paths_for(remaining[0]["slug"], remaining[0]["display_name"])


def resolve_active_profile(slug: str | None = None) -> ProfilePaths:
    ensure_profiles_ready()
    profiles = list_profiles()
    if not profiles:
        return create_profile("Predefinito")

    if slug:
        match = next((profile for profile in profiles if profile.slug == slug), None)
        if match:
            set_last_active(match.slug)
            return match

    last_active = get_last_active_slug()
    if last_active:
        match = get_profile_paths(last_active)
        if match:
            return match

    first = profiles[0]
    set_last_active(first.slug)
    return first
