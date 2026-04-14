"""Progress reporting for long-running stages.

A `ProgressCallback` is threaded through the pipeline so CLI callers, or
agent integrations (see #9555 / #9556 for the upstream Hermes delegation
equivalents), can observe ideation → writeup → verify → compile → review
as they run, not after. The default callback is a no-op so library users
who don't care pay nothing.

The CLI wires human-readable (default) or JSONL output via
``--progress {human,jsonl,off}``.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from typing import Any, Callable, Literal, Optional, TextIO

ProgressKind = Literal["stage_start", "stage_end", "item", "retry", "warning"]

# Stages an orchestration function may emit against.
Stage = Literal[
    "ideate", "writeup", "section", "critique", "coherence",
    "verify", "compile", "review",
]


@dataclasses.dataclass
class Progress:
    """One progress event.

    kind : lifecycle phase of the event
    stage: named stage the event belongs to
    message: short human-readable detail (stage-specific)
    current / total: for per-item progress (e.g. section 3/7)
    meta : free-form extras (duration_s, model, retries, pdf_path, …)
    ts   : event wall-clock timestamp
    """
    kind: ProgressKind
    stage: Stage
    message: str = ""
    current: int = 0
    total: int = 0
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    ts: float = dataclasses.field(default_factory=time.time)


#: Type of a progress callback. Should never raise — exceptions are swallowed
#: by :func:`emit` so that a broken sink cannot kill the pipeline.
ProgressCallback = Callable[[Progress], None]


def noop(_: Progress) -> None:
    """Default sink that discards every event."""


def human(p: Progress, fd: Optional[TextIO] = None) -> None:
    """Human-readable rendering of a single event to stderr.

    Uses short markers instead of spinners so it stays readable when the
    agent captures stderr line-by-line and when two concurrent runs
    interleave their output.
    """
    fd = fd if fd is not None else sys.stderr
    if p.kind == "stage_start":
        line = f"[hermes-sci] → {p.stage}: {p.message}".rstrip()
    elif p.kind == "stage_end":
        dur = p.meta.get("duration_s")
        dur_str = f" ({dur:.1f}s)" if isinstance(dur, (int, float)) else ""
        line = f"[hermes-sci] ✓ {p.stage}{dur_str}  {p.message}".rstrip()
    elif p.kind == "item":
        bar = f" [{p.current}/{p.total}]" if p.total else ""
        line = f"[hermes-sci]     • {p.stage}{bar} {p.message}".rstrip()
    elif p.kind == "retry":
        line = f"[hermes-sci]     ↻ {p.stage}: {p.message}".rstrip()
    elif p.kind == "warning":
        line = f"[hermes-sci]     ⚠ {p.stage}: {p.message}".rstrip()
    else:  # defensive — should not happen given the Literal type
        line = f"[hermes-sci] ? {p.kind} {p.stage} {p.message}"
    print(line, file=fd, flush=True)


def jsonl(p: Progress, fd: Optional[TextIO] = None) -> None:
    """One-JSON-object-per-line rendering to stderr.

    Agent callers can parse this without regex and drive progress UI off
    the structured events. Keep keys stable — downstream integrations
    will pin to them.
    """
    fd = fd if fd is not None else sys.stderr
    fd.write(json.dumps({
        "kind": p.kind,
        "stage": p.stage,
        "message": p.message,
        "current": p.current,
        "total": p.total,
        "meta": p.meta,
        "ts": round(p.ts, 3),
    }, ensure_ascii=False) + "\n")
    fd.flush()


def emit(cb: ProgressCallback, p: Progress) -> None:
    """Safely dispatch an event. Silently swallow callback errors."""
    try:
        cb(p)
    except Exception:  # noqa: BLE001 — progress must never crash the pipeline
        pass


def _resolve_builtin(name: str) -> ProgressCallback:
    """Map a ``--progress`` CLI flag value to a callback."""
    if name == "off" or name == "none":
        return noop
    if name == "jsonl":
        return jsonl
    # "human" and anything else fall through to the friendly default
    return human
