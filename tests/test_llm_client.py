"""Tests for the OpenAI-compatible client: request building without network."""

from __future__ import annotations

from app.config import Settings
from app.llm import OpenAICompatibleLLM, _extract_delta, get_llm


def _client(**overrides) -> OpenAICompatibleLLM:
    values = {
        "base_url": "https://api.example.com",
        "api_key": "test-key",
        "model": "test-model",
        "temperature": 0.0,
        "timeout": 5.0,
    }
    values.update(overrides)
    return OpenAICompatibleLLM(**values)


def test_payload_contains_expected_fields() -> None:
    payload = _client().build_payload("system", "user", stream=False)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_provider_specific_switches_are_forwarded() -> None:
    # DeepSeek's thinking mode is off by default in the app: it is forwarded as
    # a provider-specific field rather than hardcoded in the client.
    client = _client(extra_body={"thinking": {"type": "disabled"}})
    payload = client.build_payload("s", "u", stream=True)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is True


def test_core_fields_win_over_extra_body() -> None:
    # A misconfigured extra_body must not be able to break streaming or messages.
    client = _client(extra_body={"stream": "wrong", "model": "hijacked", "messages": []})
    payload = client.build_payload("s", "u", stream=True)
    assert payload["stream"] is True
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["content"] == "s"


def test_extract_delta_ignores_non_content_chunks() -> None:
    assert _extract_delta('{"choices":[{"delta":{"content":"привет"}}]}') == "привет"
    assert _extract_delta('{"choices":[{"delta":{"reasoning_content":"думаю"}}]}') == ""
    assert _extract_delta("[DONE]") == ""
    assert _extract_delta('{"choices":[]}') == ""
    assert _extract_delta("not json") == ""


def test_extra_body_is_parsed_from_settings_json() -> None:
    settings = Settings(
        llm_provider="openai_compatible",
        llm_extra_body={"thinking": {"type": "disabled"}},
    )
    llm = get_llm(settings)
    assert llm.build_payload("s", "u", stream=False)["thinking"] == {"type": "disabled"}
