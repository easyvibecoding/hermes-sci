"""Progress callback: sinks, safe dispatch, CLI flag resolution."""
from __future__ import annotations

import io
import json

import pytest

from hermes_sci.progress import (
    Progress,
    _resolve_builtin,
    emit,
    human,
    jsonl,
    noop,
)


def test_progress_defaults():
    p = Progress(kind="stage_start", stage="ideate")
    assert p.message == ""
    assert p.current == 0 and p.total == 0
    assert p.meta == {}
    assert p.ts > 0


def test_noop_sink_is_silent(capsys):
    noop(Progress(kind="stage_start", stage="ideate"))
    assert capsys.readouterr().err == ""


def test_human_sink_writes_to_provided_fd():
    buf = io.StringIO()
    human(Progress(kind="stage_start", stage="ideate", message="topic"), fd=buf)
    human(Progress(kind="item", stage="section", current=2, total=5,
                   message="method"), fd=buf)
    human(Progress(kind="stage_end", stage="verify", message="6/8",
                   meta={"duration_s": 12.4}), fd=buf)
    out = buf.getvalue()
    assert "→ ideate: topic" in out
    assert "[2/5]" in out and "method" in out
    assert "✓ verify" in out and "(12.4s)" in out


def test_human_sink_handles_unknown_kind():
    buf = io.StringIO()
    # The type says Literal, but runtime unknowns shouldn't crash.
    human(Progress(kind="wat", stage="ideate", message="x"), fd=buf)  # type: ignore[arg-type]
    assert "wat" in buf.getvalue()


def test_jsonl_is_parseable():
    buf = io.StringIO()
    jsonl(Progress(kind="item", stage="section", current=1, total=3,
                   message="intro", meta={"model": "m1"}), fd=buf)
    line = buf.getvalue().strip()
    obj = json.loads(line)
    assert obj["kind"] == "item"
    assert obj["stage"] == "section"
    assert obj["current"] == 1 and obj["total"] == 3
    assert obj["meta"] == {"model": "m1"}


def test_emit_swallows_callback_errors():
    def bad(p):
        raise RuntimeError("sink broke")
    # Must not raise — a broken sink cannot crash the pipeline.
    emit(bad, Progress(kind="stage_start", stage="ideate"))


@pytest.mark.parametrize("name,expected", [
    ("human", human),
    ("jsonl", jsonl),
    ("off", noop),
    ("none", noop),
    ("garbage", human),   # unknowns fall through to human
])
def test_resolve_builtin(name, expected):
    assert _resolve_builtin(name) is expected
