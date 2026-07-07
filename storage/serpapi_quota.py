"""Backward-compatible wrappers around storage.search_quota."""

from __future__ import annotations

from storage.search_quota import (
    clear_provider_exhausted as clear_serpapi_exhausted,
    is_provider_exhausted as is_serpapi_marked_exhausted,
    mark_provider_exhausted as mark_serpapi_exhausted,
)

__all__ = [
    "clear_serpapi_exhausted",
    "is_serpapi_marked_exhausted",
    "mark_serpapi_exhausted",
]
