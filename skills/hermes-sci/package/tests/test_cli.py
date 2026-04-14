"""CLI argparse + validate-results end-to-end (no network)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from hermes_sci.cli import build_parser

PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_parser_lists_all_subcommands():
    p = build_parser()
    ns = p.parse_args(["ideate", "--topic", "x", "-o", "i.json"])
    assert ns.cmd == "ideate"
    ns = p.parse_args(["writeup", "--ideas-json", "i.json", "-o", "out"])
    assert ns.cmd == "writeup"
    ns = p.parse_args(["review", "--paper", "p.pdf"])
    assert ns.cmd == "review"
    ns = p.parse_args(["pipeline", "--topic", "x", "-o", "out"])
    assert ns.cmd == "pipeline"
    ns = p.parse_args(["validate-results", "r.json"])
    assert ns.cmd == "validate-results"


@pytest.mark.parametrize("sink", ["human", "jsonl", "off"])
def test_progress_flag_accepted_everywhere(sink):
    p = build_parser()
    ns = p.parse_args(["ideate", "--topic", "x", "-o", "i.json",
                       "--progress", sink])
    assert ns.progress == sink


def test_progress_rejects_unknown_value():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["ideate", "--topic", "x", "-o", "i.json",
                      "--progress", "spinner"])


def test_coherence_default_false():
    """Coherence is opt-in on every subcommand; pipeline previously had a
    bug where it used args.no_coherence (never defined)."""
    p = build_parser()
    ns = p.parse_args(["pipeline", "--topic", "x", "-o", "out"])
    assert ns.coherence is False
    ns = p.parse_args(["pipeline", "--topic", "x", "-o", "out", "--coherence"])
    assert ns.coherence is True


def test_validate_results_good_exits_0(tmp_path):
    doc = {"metrics": [{"name": "BLEU", "value": 28.3}]}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(doc))
    r = subprocess.run(
        [sys.executable, "-m", "hermes_sci.cli", "validate-results", str(p)],
        capture_output=True, text=True, cwd=str(PKG_ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert "matches results.json schema" in r.stdout


def test_validate_results_bad_exits_1(tmp_path):
    doc = {"metrics": [], "tables": [
        {"id": "bad id", "headers": ["x"], "rows": []}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    r = subprocess.run(
        [sys.executable, "-m", "hermes_sci.cli", "validate-results", str(p)],
        capture_output=True, text=True, cwd=str(PKG_ROOT),
    )
    assert r.returncode == 1
    assert "schema violation" in r.stderr


def test_validate_results_missing_file_exits_2(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "hermes_sci.cli",
         "validate-results", str(tmp_path / "nope.json")],
        capture_output=True, text=True, cwd=str(PKG_ROOT),
    )
    assert r.returncode == 2
