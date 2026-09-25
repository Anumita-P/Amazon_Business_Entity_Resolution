"""Conservative baseline normalization for Stage 1.

Design rules (from the EDA plan):
  * Raw values are NEVER overwritten — normalization creates NEW columns.
  * The conservative form is the primary lexical representation.
  * The aggressive form exists ONLY for collision/recall analysis, never as a
    direct match rule.
  * We do NOT blindly remove house numbers, postal codes, business-type words
    or legal suffixes. Their behaviour must remain analyzable.

Unicode handling is deliberately conservative: we keep unicode letters
(e.g. French accents) instead of stripping everything to ASCII, because
France is unseen in training and must still be matchable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

# ---------------------------------------------------------------------------
# Analysis-only reference lists.
# These are used to *measure* suffix behaviour in EDA, NOT to strip fields in
# the conservative pipeline. The aggressive form uses them only so we can
# quantify over-normalization danger.
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_TOKENS = frozenset(
    {
        "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
        "co", "company", "corp.", "inc.", "ltd.", "llp", "pllc",
        "pvt", "private", "pte",
        "gmbh", "sarl", "sas", "sa", "sci", "eurl",  # seen in FR-style names
    }
)

BUSINESS_TYPE_TOKENS = frozenset(
    {
        "enterprises", "enterprise", "traders", "trading", "services",
        "solutions", "industries", "industry", "associates", "agency",
        "hotel", "hotels", "restaurant", "restaurants", "cafe", "bakery",
        "pharmacy", "hospital", "clinic", "store", "stores", "mart",
        "motors", "auto", "textiles", "jewellers", "jewelers",
    }
)

_WS_RE = re.compile(r"\s+")
# Numeric / alphanumeric tokens such as 12, 12A, B-17, 2/3, 4th, 560001.
_NUMERIC_TOKEN_RE = re.compile(r"[\w]*\d[\w]*(?:[/\-][\w]+)*", re.UNICODE)


def _collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_basic(value: object) -> str:
    """Conservative normalization.

    NFKC -> lowercase -> '&' to 'and' -> non-alphanumeric (unicode-aware) to
    space -> collapse whitespace. Digits, unicode letters, suffixes and word
    order are all preserved.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("&", " and ")
    # Keep unicode alphanumerics + whitespace; everything else becomes a space.
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return _collapse_ws(text)


def normalize_aggressive(value: object) -> str:
    """Aggressive normalization for COLLISION ANALYSIS ONLY.

    conservative form -> drop legal-suffix + business-type tokens -> sort the
    remaining tokens -> dedupe. This intentionally destroys order and suffix
    evidence so EDA can measure how dangerous it would be as a match rule.
    """
    base = normalize_basic(value)
    if not base:
        return ""
    kept = [
        tok
        for tok in base.split(" ")
        if tok not in LEGAL_SUFFIX_TOKENS and tok not in BUSINESS_TYPE_TOKENS
    ]
    if not kept:
        return ""
    return " ".join(sorted(set(kept)))


def tokenize(text: str) -> List[str]:
    """Whitespace tokenize an (already normalized) string."""
    if not text:
        return []
    return [t for t in text.split(" ") if t]


def extract_numeric_tokens(address: object) -> List[str]:
    """Extract numeric/alphanumeric address tokens, preserving values.

    Keeps 12, 12A, 560001, 4th, 2/3, B-17, ... Comparison-friendly: lowercase.
    """
    if address is None:
        return []
    text = unicodedata.normalize("NFKC", str(address))
    toks = _NUMERIC_TOKEN_RE.findall(text)
    return [t.lower() for t in toks if t and t.strip("/-")]


def extract_postcode_like_tokens(address: object) -> List[str]:
    """Heuristic postcode/PIN-like tokens: all-digit tokens of length 5-6.

    This is a *country-agnostic heuristic* used for diagnostics, not a
    country-specific parser (France must work without special-casing).
    """
    return [t for t in extract_numeric_tokens(address) if t.isdigit() and len(t) in (5, 6)]


def string_stats(value: object) -> Dict[str, float]:
    """Character-level diagnostics for one raw string (country-shift EDA)."""
    text = "" if value is None else str(value)
    n = len(text)
    n_ascii = sum(1 for ch in text if ord(ch) < 128)
    digit_count = sum(1 for ch in text if ch.isdigit())
    punct_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    toks = text.split()
    return {
        "char_len": float(n),
        "digit_count": float(digit_count),
        "punct_count": float(punct_count),
        "non_ascii_ratio": float((n - n_ascii) / n) if n else 0.0,
        "token_count": float(len(toks)),
        "numeric_token_count": float(len(extract_numeric_tokens(text))),
    }
