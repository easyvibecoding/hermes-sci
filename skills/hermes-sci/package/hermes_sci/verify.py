"""VerifiedRegistry — anti-hallucination audit of paper numerical claims.

After write_paper() produces section LaTeX, we:
  1. Extract every numeric token in the body (integers, decimals, percentages).
  2. Match each against Results.all_numeric_values() (with a tolerance).
  3. Emit a verification report: verified / likely-unverified / definitely-fabricated.

Design decisions:
  - Tolerance 0.5% relative OR 0.1 absolute (whichever is larger). A paper
    that rounds 28.31 to 28.3 should still pass.
  - Whitelist: integers 0-10 are assumed "structural" (list counts, epoch
    indices, etc.) and not audited — too many false positives.
  - Page / equation references (e.g. "Eq. 1", "Sec. 2") are ignored.
  - Hardware spec numbers (16 GB, 8 cores) are whitelisted if they match
    the HardwareProfile.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from typing import Optional

from .hardware import HardwareProfile, detect as detect_hardware
from .results import Results

log = logging.getLogger("hermes_sci.verify")

# Any decimal or percentage in prose. Excludes plain integers ≤ 10.
# Supported thousands separators: "21,346" / "21\,346" / "21{,}346"  (LaTeX idioms)
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.\\])"
    r"(\d{1,3}(?:(?:\\,|\{,\}|,)\s*\d{3})+(?:\.\d+)?"
    r"|\d+\.\d+"
    r"|\d{2,}"
    r")"
    r"\s*(%|\\%)?",
)

# Contexts to skip — equation / section / reference / table cells handled separately
_SKIP_CONTEXTS = re.compile(
    r"\\(?:ref|eqref|cite[tp]?|label|url|pageref)\*?\{[^}]*\}"
    r"|\\begin\{equation\*?\}.*?\\end\{equation\*?\}"
    r"|\\begin\{align\*?\}.*?\\end\{align\*?\}"
    r"|\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]+\$"
    r"|Eq\.?\s*\d+|Section\s*\d+|Sec\.?\s*\d+|Figure\s*\d+|Fig\.?\s*\d+|Table\s*\d+|Tab\.?\s*\d+",
    re.DOTALL | re.IGNORECASE,
)


@dataclasses.dataclass
class Claim:
    raw: str            # "40%", "28.3", "1,024"
    value: float        # normalised to float
    is_percentage: bool
    section: str        # which section it appeared in
    snippet: str        # 40-char context excerpt


@dataclasses.dataclass
class VerificationReport:
    verified: list[Claim]
    unverified: list[Claim]
    total_claims: int

    @property
    def verification_rate(self) -> float:
        return len(self.verified) / max(1, self.total_claims)

    def to_dict(self) -> dict:
        def c2d(c: Claim) -> dict:
            return {"raw": c.raw, "value": c.value,
                    "percent": c.is_percentage, "section": c.section,
                    "snippet": c.snippet}
        return {
            "total_claims": self.total_claims,
            "verified_count": len(self.verified),
            "unverified_count": len(self.unverified),
            "verification_rate": round(self.verification_rate, 3),
            "verified": [c2d(c) for c in self.verified],
            "unverified": [c2d(c) for c in self.unverified],
        }


def _parse_number(raw: str, pct: bool) -> Optional[float]:
    try:
        cleaned = (
            raw.replace("\\,", "")     # LaTeX thin space
               .replace("{,}", "")     # LaTeX grouped comma: 21{,}346
               .replace(",", "")       # plain comma
               .replace(" ", "")
        )
        val = float(cleaned)
    except ValueError:
        return None
    return val


def _mask_skip_contexts(text: str) -> str:
    """Replace each skip-context with spaces so offsets are preserved."""
    def pad(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    return _SKIP_CONTEXTS.sub(pad, text)


def extract_claims(section_name: str, body: str) -> list[Claim]:
    masked = _mask_skip_contexts(body)
    claims: list[Claim] = []
    for m in _NUMBER_TOKEN.finditer(masked):
        raw = m.group(1)
        pct = bool(m.group(2))
        val = _parse_number(raw, pct)
        if val is None:
            continue
        # Structural integers ≤ 10 (list counts) are not audited
        if not pct and "." not in raw and "," not in raw and val <= 10:
            continue
        start = max(0, m.start() - 20)
        end = min(len(body), m.end() + 20)
        claims.append(Claim(
            raw=raw + ("%" if pct else ""),
            value=val, is_percentage=pct,
            section=section_name,
            snippet=body[start:end].replace("\n", " ").strip(),
        ))
    return claims


def _build_registry(results: Optional[Results],
                    hw: Optional[HardwareProfile]) -> set[float]:
    reg: set[float] = set()
    if results:
        for v in results.all_numeric_values():
            reg.add(v)
    if hw:
        if hw.unified_ram_gb:
            reg.add(float(hw.unified_ram_gb))
        if hw.cpu_cores:
            reg.add(float(hw.cpu_cores))
    # Conservative whitelist: only items that are structural (dimensions,
    # years, common ML epoch counts). 100 / 1000 deliberately EXCLUDED so a
    # fabricated "99.99%" cannot hide behind them.
    reg.update({2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0, 2026.0,
                2027.0, 2028.0,                       # year
                16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                                                     # power-of-2 dims
                20.0, 30.0, 50.0})                    # common epoch counts
    return reg


def _match_tolerance(claim: Claim, registry: set[float]) -> bool:
    """Accept a claim if it matches any registry value within 0.5% rel or 0.1 abs."""
    for v in registry:
        if v == 0:
            if abs(claim.value) < 0.1:
                return True
            continue
        rel = abs(claim.value - v) / abs(v)
        if rel <= 0.005 or abs(claim.value - v) <= 0.1:
            return True
    return False


def audit(
    sections: dict[str, str],
    results: Optional[Results] = None,
    hw: Optional[HardwareProfile] = None,
    sections_to_audit: tuple[str, ...] = ("results", "experiments", "discussion",
                                          "introduction"),
) -> VerificationReport:
    """Audit numerical claims across given sections against results registry."""
    hw = hw or detect_hardware()
    registry = _build_registry(results, hw)
    verified: list[Claim] = []
    unverified: list[Claim] = []

    for sec_name in sections_to_audit:
        body = sections.get(sec_name)
        if not body:
            continue
        for claim in extract_claims(sec_name, body):
            if _match_tolerance(claim, registry):
                verified.append(claim)
            else:
                unverified.append(claim)
    total = len(verified) + len(unverified)
    log.info("verified %d/%d claims across %s", len(verified), total,
             sections_to_audit)
    return VerificationReport(verified=verified, unverified=unverified,
                              total_claims=total)


def annotate_unverified(body: str, claims: list[Claim]) -> str:
    """Wrap each unverified claim in \\textcolor{red}{...} + [unverified] tag.

    Applied conservatively — only when the exact raw token appears in the
    body. Longest claims first to avoid substring collision.
    """
    out = body
    for c in sorted(set(cl.raw for cl in claims), key=len, reverse=True):
        # Escape for regex; match as whole token
        pat = re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)")
        out = pat.sub(r"\\textcolor{red}{" + c + r"}\\textsuperscript{[?]}", out, count=1)
    return out
