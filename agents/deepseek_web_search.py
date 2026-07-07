from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEEPSEEK_ANTHROPIC_URL = "https://api.deepseek.com/anthropic/v1/messages"


async def deepseek_web_search(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    model: str | None = None,
    max_uses: int = 3,
    max_tokens: int = 2500,
    timeout_seconds: float = 90.0,
) -> tuple[str, list[str]]:
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not key or key == "your_deepseek_api_key_here":
        raise RuntimeError("DEEPSEEK_API_KEY missing")

    selected_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    payload = {
        "model": selected_model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_uses,
            },
        ],
        "tool_choice": {"type": "auto"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
    }

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            DEEPSEEK_ANTHROPIC_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return extract_anthropic_text_and_sources(data)


def extract_anthropic_text_and_sources(payload: dict[str, Any]) -> tuple[str, list[str]]:
    texts: list[str] = []
    sources: list[str] = []

    for block in payload.get("content", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                texts.append(str(text))
        elif block_type == "web_search_tool_result":
            for item in block.get("content", []) or []:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if url:
                    sources.append(str(url))

    return "\n".join(texts), sources
