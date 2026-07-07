from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You generate job search keywords for a candidate.
Return ONLY valid json with this schema:
{
  "keywords": ["keyword 1", "keyword 2"]
}

Rules:
- Produce 6 to 10 concise search keywords/phrases in English or Italian.
- Mix explicit role titles and related synonyms.
- Reflect career field, experience level, passions, and location context.
- Do not include company names.
"""


class KeywordExpansion(BaseModel):
    keywords: list[str] = Field(min_length=1)


class KeywordExpander:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def expand(self, profile: UserProfile) -> list[str]:
        base_keywords = list(dict.fromkeys(profile.target_roles))
        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            return self._fallback_keywords(profile, base_keywords)

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Generate search keywords as json for this profile:\n"
                            f"{json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)}"
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = KeywordExpansion.model_validate(json.loads(content))
            merged = list(dict.fromkeys([*base_keywords, *parsed.keywords]))
            return merged[:12]
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            logger.warning("[KeywordExpander] Failed, using fallback keywords: %s", exc)
            return self._fallback_keywords(profile, base_keywords)

    def _fallback_keywords(self, profile: UserProfile, base_keywords: list[str]) -> list[str]:
        level_aliases = {
            "graduate": "graduate",
            "internship": "intern",
            "entry": "junior",
            "mid": "",
            "senior": "senior",
            "manager": "manager",
        }
        level_word = level_aliases.get(profile.experience_level, "")
        expanded = list(base_keywords)
        for role in profile.target_roles:
            if level_word:
                expanded.append(f"{level_word} {role}".strip())
            for place in profile.search_location_targets():
                expanded.append(f"{role} {place}".strip())
        return list(dict.fromkeys(keyword for keyword in expanded if keyword))[:10]
