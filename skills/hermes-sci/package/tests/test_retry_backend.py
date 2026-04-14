"""`--retry-backend hybrid` flag: opt-in, silent fallback when proxy down."""
from __future__ import annotations

from unittest import mock

from hermes_sci.cli import _build_retry_cfg, build_parser
from hermes_sci.progress import Progress, noop


def _parse_writeup(*extra):
    p = build_parser()
    return p.parse_args([
        "writeup", "--ideas-json", "i.json", "-o", "out", *extra,
    ])


def test_flag_default_is_same():
    ns = _parse_writeup()
    assert ns.retry_backend == "same"
    assert ns.retry_model is None


def test_flag_accepts_hybrid():
    ns = _parse_writeup("--retry-backend", "hybrid")
    assert ns.retry_backend == "hybrid"


def test_flag_rejects_unknown_value():
    p = build_parser()
    import pytest
    with pytest.raises(SystemExit):
        p.parse_args(["writeup", "--ideas-json", "i.json", "-o", "out",
                      "--retry-backend", "anthropic"])


def test_retry_model_accepted():
    ns = _parse_writeup("--retry-backend", "hybrid",
                        "--retry-model", "claude-sonnet-4-5")
    assert ns.retry_model == "claude-sonnet-4-5"


def test_build_retry_cfg_default_returns_none():
    """retry-backend=same → no retry cfg built, retry keeps using main cfg."""
    ns = _parse_writeup()
    rc, rm = _build_retry_cfg(ns, primary_cfg=object(), progress_cb=noop)
    assert rc is None and rm is None


def test_build_retry_cfg_hybrid_probe_fail_falls_back_silently():
    """The whole point: users without delegation keep working.

    Proxy probe fails → _build_retry_cfg returns (None, None) + emits a
    `warning` progress event. No exception, no abort."""
    ns = _parse_writeup("--retry-backend", "hybrid")
    events: list[Progress] = []

    def capture(p: Progress) -> None:
        events.append(p)

    with mock.patch("hermes_sci.cli.probe_claude_proxy", return_value=False):
        rc, rm = _build_retry_cfg(ns, primary_cfg=object(), progress_cb=capture)

    assert rc is None and rm is None
    assert any(e.kind == "warning" and "unreachable" in e.message for e in events)


def test_build_retry_cfg_hybrid_probe_ok_builds_cfg():
    ns = _parse_writeup("--retry-backend", "hybrid")
    fake_cfg = object()

    with mock.patch("hermes_sci.cli.probe_claude_proxy", return_value=True), \
         mock.patch("hermes_sci.cli.resolve_backend", return_value=fake_cfg) \
             as rb:
        rc, rm = _build_retry_cfg(ns, primary_cfg=object(), progress_cb=noop)

    assert rc is fake_cfg
    assert rm == "claude-opus-4-5"  # default when --retry-model absent
    # Called with hybrid backend.
    assert rb.call_args.kwargs["backend"] == "hybrid"


def test_build_retry_cfg_hybrid_respects_explicit_retry_model():
    ns = _parse_writeup("--retry-backend", "hybrid",
                        "--retry-model", "claude-haiku-4-5")
    with mock.patch("hermes_sci.cli.probe_claude_proxy", return_value=True), \
         mock.patch("hermes_sci.cli.resolve_backend", return_value=object()):
        _, rm = _build_retry_cfg(ns, primary_cfg=object(), progress_cb=noop)
    assert rm == "claude-haiku-4-5"


def test_probe_claude_proxy_false_on_unreachable():
    """Real function — no server on this port, must return False fast."""
    from hermes_sci.config import probe_claude_proxy
    assert probe_claude_proxy("http://127.0.0.1:1", timeout_s=0.5) is False
