"""Unified LLM client + request helpers.

Single interface for chat completions, regardless of provider. Uses the
OpenAI SDK (v1.x) against any OpenAI-compatible endpoint (MiniMax / OpenAI /
DeepSeek / etc.) and optionally falls through to Anthropic SDK when backend
is 'hybrid' + model starts with 'claude-'.

Why not multi-SDK: keeps code simple. Anthropic via hybrid hits our local
claude -p proxy, which emits Anthropic-compatible JSON.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time
from typing import Any, Optional, Sequence
try:
    import zoneinfo  # py3.9+
except ImportError:  # pragma: no cover
    zoneinfo = None  # type: ignore

import openai

from .config import BackendConfig

log = logging.getLogger("hermes_sci.llm")

MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMP = 0.75
RETRY_SLEEP = (2, 4, 8, 16)  # exponential backoff


class LLMError(RuntimeError):
    pass


# ── Concurrency & peak-hour throttle ────────────────────────────────

# MiniMax token-plan peak hours (per platform FAQ): weekdays 15:00–17:30
# Asia/Shanghai (HQ). During peak, Starter/Plus get ~1 concurrent agent call;
# off-peak allows healthy parallelism.
_PEAK_TZ_NAME = "Asia/Shanghai"
_PEAK_START_H = 15.0
_PEAK_END_H = 17.5  # 17:30


def _peak_tz():
    if zoneinfo is None:
        return None
    try:
        return zoneinfo.ZoneInfo(_PEAK_TZ_NAME)
    except Exception:  # pragma: no cover
        return None


def is_minimax_peak(now: Optional[datetime.datetime] = None) -> bool:
    """Is it currently MiniMax's peak hour (weekday 15:00-17:30 Asia/Shanghai)?"""
    tz = _peak_tz()
    if now is None:
        now = datetime.datetime.now(tz=tz) if tz else datetime.datetime.now()
    elif tz:
        now = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    hhmm = now.hour + now.minute / 60.0
    return _PEAK_START_H <= hhmm < _PEAK_END_H


# Recommended concurrency per provider — conservative upper bound to avoid
# throttle. Tune with `--concurrency N` on the CLI to override.
PROVIDER_OFFPEAK_CONCURRENCY = {
    "minimax": 7,      # Starter/Plus off-peak handles 5-10 comfortably
    "openai": 16,      # generous for paid OpenAI
    "deepseek": 8,
    "moonshot": 4,
    "gemini": 8,
    "groq": 8,         # very high TPS but concurrent cap lower
    "anthropic": 8,
    "together": 8,
    "xai": 8,
    "zhipu": 4,
}


def recommended_concurrency(cfg: BackendConfig) -> int:
    """Suggested asyncio.gather concurrency limit for the current backend.

    Applies MiniMax peak-hour rules. Returns an int >= 1.
    """
    provider = cfg.provider.lower()
    base = PROVIDER_OFFPEAK_CONCURRENCY.get(provider, 4)
    if provider == "minimax" and is_minimax_peak():
        log.info("MiniMax peak hour detected — throttling concurrency to 1")
        return 1
    return max(1, base)


def make_openai_client(cfg: BackendConfig) -> openai.OpenAI:
    return openai.OpenAI(api_key=cfg.api_key, base_url=cfg.openai_base)


def make_openai_async_client(cfg: BackendConfig) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.openai_base)


def make_anthropic_client(cfg: BackendConfig):
    """Only used under hybrid backend (endpoint → claude_proxy)."""
    import anthropic  # deferred so minimax-only installs need not install anthropic
    return anthropic.Anthropic(
        api_key="hermes-claude-proxy-dummy",
        base_url=cfg.claude_proxy_url,
    )


def _is_claude_model(model: str) -> bool:
    return model.lower().startswith("claude-")


def complete(
    cfg: BackendConfig,
    *,
    system: str,
    user: str,
    history: Optional[Sequence[dict]] = None,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    n: int = 1,
) -> tuple[str, list[dict]]:
    """Single-shot chat completion. Returns (text, updated_history).

    For n>1, returns the first choice (use `complete_batch` for ensembles).
    """
    model = model or cfg.model
    history = list(history or [])
    new_msgs = history + [{"role": "user", "content": user}]

    if cfg.backend == "hybrid" and _is_claude_model(model):
        return _anthropic_complete(cfg, system, new_msgs, model, temperature, max_tokens)

    client = make_openai_client(cfg)
    last_err: Optional[Exception] = None
    for attempt in range(len(RETRY_SLEEP) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, *new_msgs],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            text = resp.choices[0].message.content or ""
            new_msgs.append({"role": "assistant", "content": text})
            return text, new_msgs
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.APITimeoutError, openai.InternalServerError) as e:
            last_err = e
            if attempt < len(RETRY_SLEEP):
                log.warning("LLM retry %d after %s", attempt + 1, e.__class__.__name__)
                time.sleep(RETRY_SLEEP[attempt])
            else:
                raise LLMError(f"LLM call failed after retries: {e}") from e
    raise LLMError(f"Unreachable: {last_err}")  # pragma: no cover


async def acomplete(
    cfg: BackendConfig,
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    """Async single-shot completion. Used for parallel section generation.

    Uses `async with` so the AsyncOpenAI client's underlying httpx pool is
    closed inside the caller's event loop (prevents "Event loop is closed"
    errors at GC time when callers do multiple asyncio.run() rounds).

    Does not support hybrid+claude (no async anthropic shim yet) — falls back
    to a thread-pool sync call so callers can still gather across sections.
    """
    model = model or cfg.model
    if cfg.backend == "hybrid" and _is_claude_model(model):
        return await asyncio.to_thread(
            lambda: complete(cfg, system=system, user=user, model=model,
                             temperature=temperature, max_tokens=max_tokens)[0]
        )
    last_err: Optional[Exception] = None
    for attempt in range(len(RETRY_SLEEP) + 1):
        try:
            async with make_openai_async_client(cfg) as client:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens, n=1,
                )
                return resp.choices[0].message.content or ""
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.APITimeoutError, openai.InternalServerError) as e:
            last_err = e
            if attempt < len(RETRY_SLEEP):
                log.warning("async LLM retry %d after %s", attempt + 1, e.__class__.__name__)
                await asyncio.sleep(RETRY_SLEEP[attempt])
            else:
                raise LLMError(f"async LLM call failed after retries: {e}") from e
    raise LLMError(f"Unreachable: {last_err}")


def _anthropic_complete(cfg, system, messages, model, temperature, max_tokens):
    client = make_anthropic_client(cfg)
    # Anthropic SDK wants content list-of-blocks
    anth_msgs = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in messages
    ]
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=anth_msgs,
    )
    text = resp.content[0].text if resp.content else ""
    return text, messages + [{"role": "assistant", "content": text}]


def complete_batch(
    cfg: BackendConfig,
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    n: int = 3,
) -> list[str]:
    """Return n sampled completions for ensemble voting / review."""
    model = model or cfg.model
    if cfg.backend == "hybrid" and _is_claude_model(model):
        # Anthropic proxy: loop n times
        return [_anthropic_complete(cfg, system,
                                    [{"role": "user", "content": user}],
                                    model, temperature, max_tokens)[0]
                for _ in range(n)]

    client = make_openai_client(cfg)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens, n=n,
    )
    return [c.message.content or "" for c in resp.choices]


# ── JSON extraction ───────────────────────────────────────────────────

_JSON_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_BRACES = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any:
    """Try to parse the first JSON object/array in `text`. Returns None on miss."""
    for pattern in (_JSON_FENCED, _JSON_BRACES):
        for match in pattern.finditer(text):
            chunk = match.group(1).strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                # strip control chars + retry
                cleaned = re.sub(r"[\x00-\x1F\x7F]", "", chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
    return None
