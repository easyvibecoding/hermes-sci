"""YAML rule loader for the sanitize pipeline.

Rules live in `hermes_sci/data/*.yaml`. Each data-driven sanitize pass loads
its section at import time and compiles the regex rules into (pattern, repl)
tuples ready for re.sub().

Community contribution surface: edit the YAML, run `pip install -e .`,
done. No Python changes needed for new regex-style rules.
"""
from __future__ import annotations

import functools
import pathlib
import re
from typing import Any, Iterable

import yaml

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

_FLAG_MAP = {
    "DOTALL": re.DOTALL,
    "MULTILINE": re.MULTILINE,
    "IGNORECASE": re.IGNORECASE,
    "VERBOSE": re.VERBOSE,
    "UNICODE": re.UNICODE,
}


@functools.lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    path = _DATA_DIR / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compile_flags(flag_names: Iterable[str] | None) -> int:
    out = 0
    for fname in flag_names or ():
        if fname not in _FLAG_MAP:
            raise ValueError(f"unknown regex flag: {fname}")
        out |= _FLAG_MAP[fname]
    return out


def compiled_rules(yaml_file: str, section: str) -> list[tuple[re.Pattern, str, str]]:
    """Compile one section of a rules YAML file.

    Returns a list of (compiled_pattern, replacement, name) tuples, in file
    order. Each tuple is ready for `pattern.sub(replacement, text)`.
    """
    data = _load_yaml(yaml_file)
    raw = data.get(section, [])
    out: list[tuple[re.Pattern, str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        pat = entry.get("pattern")
        repl = entry.get("replacement", "")
        name = entry.get("name") or pat[:32]
        if pat is None:
            continue
        flags = _compile_flags(entry.get("flags"))
        out.append((re.compile(pat, flags), repl, name))
    return out


def raw_list(yaml_file: str, section: str) -> list[Any]:
    """Return a section that's just a plain list (e.g. cjk_ranges)."""
    data = _load_yaml(yaml_file)
    return list(data.get(section, []))


def apply_rules(text: str, rules: list[tuple[re.Pattern, str, str]]) -> str:
    for pat, repl, _name in rules:
        text = pat.sub(repl, text)
    return text
