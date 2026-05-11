"""LLM factory.

OpenRouter exposes an OpenAI-compatible REST API, so we point
``langchain_openai.ChatOpenAI`` at it and tag every request with the optional
``HTTP-Referer`` / ``X-Title`` headers OpenRouter uses for traffic attribution.

``max_retries`` is wired through to the underlying ``openai`` client, which
already implements exponential backoff with jitter for 429s and 5xxs. We keep
returning a concrete ``ChatOpenAI`` (not a wrapped ``Runnable``) so callers
can still use ``with_structured_output`` and ``bind_tools``.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings


def build_llm(temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openrouter_model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else s.llm_max_tokens,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        max_retries=s.llm_max_retries,
        default_headers={
            "HTTP-Referer": s.openrouter_http_referer,
            "X-Title": s.openrouter_app_name,
        },
    )
