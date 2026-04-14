"""Backend resolution — reads ~/.hermes/config.yaml for the user's current
model/provider choice, maps to OpenAI-compatible endpoint + API key.

Single source of truth for:
  - `minimax` backend: all LLM calls → Hermes-selected provider
  - `hybrid` backend: text calls → Hermes provider; Anthropic-SDK calls → local
    `claude -p` proxy (started separately via scripts/claude_proxy_ctl.sh)
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Literal, Optional

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

Backend = Literal["minimax", "hybrid"]

# provider → (OpenAI-compat base URL, env var holding API key)
PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "minimax":   ("https://api.minimax.io/v1",           "MINIMAX_API_KEY"),
    "openai":    ("https://api.openai.com/v1",           "OPENAI_API_KEY"),
    "deepseek":  ("https://api.deepseek.com/v1",         "DEEPSEEK_API_KEY"),
    "moonshot":  ("https://api.moonshot.cn/v1",          "MOONSHOT_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1",        "ANTHROPIC_API_KEY"),
    "gemini":    ("https://generativelanguage.googleapis.com/v1beta/openai",
                  "GEMINI_API_KEY"),
    "groq":      ("https://api.groq.com/openai/v1",      "GROQ_API_KEY"),
    "together":  ("https://api.together.xyz/v1",         "TOGETHER_API_KEY"),
    "xai":       ("https://api.x.ai/v1",                 "XAI_API_KEY"),
    "zhipu":     ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}

FALLBACK_MODEL = "MiniMax-M2.7"
FALLBACK_PROVIDER = "minimax"


@dataclasses.dataclass
class BackendConfig:
    backend: Backend
    model: str                 # resolved model (user override or Hermes default)
    provider: str              # Hermes-selected provider
    openai_base: str           # OpenAI-compat endpoint
    api_key: str               # resolved from env
    claude_proxy_url: Optional[str] = None  # set when backend == "hybrid"


def _hermes_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HERMES_HOME", str(pathlib.Path.home() / ".hermes")))


def read_hermes_defaults() -> tuple[str, str]:
    """Return (model, provider) from ~/.hermes/config.yaml."""
    path = _hermes_home() / "config.yaml"
    if not path.exists():
        return FALLBACK_MODEL, FALLBACK_PROVIDER

    if yaml is not None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return FALLBACK_MODEL, FALLBACK_PROVIDER
        block = (data.get("model") or {})
        return (str(block.get("default") or FALLBACK_MODEL),
                str(block.get("provider") or FALLBACK_PROVIDER))

    # Manual parse — look for a model: block with default/provider keys.
    model = provider = None
    in_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.rstrip() == "model:":
            in_block = True
            continue
        if in_block:
            if not line.startswith(" "):
                break
            s = line.strip()
            if s.startswith("default:"):
                model = s.split(":", 1)[1].strip().strip("'\"")
            elif s.startswith("provider:"):
                provider = s.split(":", 1)[1].strip().strip("'\"")
    return (model or FALLBACK_MODEL, provider or FALLBACK_PROVIDER)


def _read_dotenv_key(name: str) -> str:
    """Read a key from ~/.hermes/.env directly.

    Needed when `hermes chat` sandboxes subprocess env (redacting real keys to
    `***` placeholders). The .env file on disk still holds real values.
    Supports plain `KEY=value`, `export KEY=value`, single/double quoted.
    """
    env_path = _hermes_home() / ".env"
    if not env_path.exists():
        return ""
    prefixes = (f"{name}=", f"export {name}=")
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            for pre in prefixes:
                if s.startswith(pre):
                    val = s[len(pre):].strip().strip("'\"")
                    if val and val != "***":
                        return val
    except OSError:
        pass
    return ""


def _resolve_api_key(key_var: str) -> str:
    """Env var first, then ~/.hermes/.env fallback (bypasses hermes chat sandbox)."""
    for src in (os.environ.get(key_var),
                _read_dotenv_key(key_var),
                os.environ.get("MINIMAX_API_KEY"),
                _read_dotenv_key("MINIMAX_API_KEY")):
        if src and src != "***":
            return src
    return ""


def resolve_backend(
    backend: Backend = "minimax",
    model_override: Optional[str] = None,
    claude_proxy_url: str = "http://127.0.0.1:9099",
) -> BackendConfig:
    """Resolve Hermes-selected provider + endpoint into a concrete BackendConfig."""
    model, provider = read_hermes_defaults()
    endpoint, key_var = PROVIDER_MAP.get(provider.lower(), PROVIDER_MAP["minimax"])

    api_key = _resolve_api_key(key_var)
    if not api_key:
        raise RuntimeError(
            f"API key not found (tried ${key_var}, $MINIMAX_API_KEY, "
            f"~/.hermes/.env). Set MINIMAX_API_KEY in ~/.hermes/.env."
        )
    # If we fell back to MiniMax key because provider-specific was missing,
    # flip endpoint to MiniMax so the credentials match.
    provider_key_present = bool(os.environ.get(key_var) or _read_dotenv_key(key_var))
    if not provider_key_present:
        endpoint = PROVIDER_MAP["minimax"][0]
        provider = "minimax"

    resolved_model = model_override or model

    if backend == "minimax":
        return BackendConfig(
            backend=backend, model=resolved_model, provider=provider,
            openai_base=endpoint, api_key=api_key, claude_proxy_url=None,
        )
    if backend == "hybrid":
        return BackendConfig(
            backend=backend, model=resolved_model, provider=provider,
            openai_base=endpoint, api_key=api_key,
            claude_proxy_url=claude_proxy_url,
        )
    raise ValueError(f"Unknown backend: {backend}")


def probe_claude_proxy(url: str, timeout_s: float = 1.5) -> bool:
    """Return True iff GET {url}/health responds {"ok": true} fast enough.

    Intended for CLI `--retry-backend hybrid` where we want to silently
    fall back to the main backend if the claude proxy isn't running —
    not every user has delegation wired up, and failing the whole run
    because of an optional-upgrade path would be hostile.
    """
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health",
                                     timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
            return bool(data.get("ok"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


def apply_env(cfg: BackendConfig) -> None:
    """Export env so 3rd-party SDKs (openai, anthropic) pick up the config."""
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    os.environ["OPENAI_API_BASE"] = cfg.openai_base
    os.environ["OPENAI_BASE_URL"] = cfg.openai_base
    if cfg.backend == "hybrid":
        os.environ["ANTHROPIC_API_KEY"] = "hermes-claude-proxy-dummy"
        os.environ["ANTHROPIC_BASE_URL"] = cfg.claude_proxy_url or ""
    else:
        os.environ["ANTHROPIC_API_KEY"] = cfg.api_key
        os.environ["ANTHROPIC_BASE_URL"] = cfg.openai_base.removesuffix("/v1")
