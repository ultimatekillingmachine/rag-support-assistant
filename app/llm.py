"""LLM providers.

* ``mock`` — offline extractive stub, used by tests and CI (no keys needed);
* ``openai_compatible`` — any OpenAI-compatible /chat/completions endpoint
  (OpenAI, OpenRouter, vLLM, ...);
* ``ollama`` — local models, also via its OpenAI-compatible endpoint
  (http://localhost:11434/v1).
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.config import Settings

CONTEXT_MARKER = "ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ:"


class LLMError(RuntimeError):
    """Raised when the LLM endpoint fails or returns an unexpected shape."""


class LLM(Protocol):
    name: str
    model: str

    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class MockLLM:
    """Deterministic extractive stub: echoes the top retrieved fragment.

    Keeps the whole pipeline (retrieval + prompt building + citations)
    verifiable in tests and CI without any API keys.
    """

    name = "mock"
    model = "mock-extractive"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        tail = user_prompt.split(CONTEXT_MARKER, 1)[-1].strip()
        first_block = tail.split("\n\n", 1)[0].strip()
        return f"[mock-ответ]\n{first_block[:600]}"


class OpenAICompatibleLLM:
    """Async client for OpenAI-compatible chat completion endpoints."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {data!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def get_llm(settings: Settings) -> LLM:
    """Build the LLM provider from settings."""
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.llm_provider in {"openai_compatible", "ollama"}:
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")
