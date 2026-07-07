from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from models.job import JobPosting
from models.user_profile import UserProfile

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "BaseAgent"

    @abstractmethod
    async def run(self, profile: UserProfile) -> list[JobPosting]:
        raise NotImplementedError

    async def safe_run(self, profile: UserProfile) -> list[JobPosting]:
        try:
            results = await self.run(profile)
            logger.info("[%s] Returned %s job postings.", self.name, len(results))
            return results
        except Exception as exc:
            logger.exception("[%s] Agent failed but pipeline continues: %s", self.name, exc)
            return []
