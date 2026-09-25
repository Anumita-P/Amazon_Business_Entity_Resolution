"""Pair-level similarity features for positive / hard-negative EDA.

Everything here is computed from the two records' own fields only — no
external data, no geocoding, no internet. Country is treated as an open-set
string (compared for equality, never one-hotted against a fixed list).

RapidFuzz is preferred for string scores; if it is unavailable a difflib-based
fallback keeps the pipeline runnable (with a warning).
"""

from __future__ import annotations

import difflib
import logging
import warnings
from typing import Dict, List, Tuple

import pandas as pd

from .normalization import (
    extract_numeric_tokens,
    extract_postcode_like_tokens,
    tokenize,
)

logger = logging.getLogger("ber")

try:
    from rapidfuzz.fuzz import WRatio, ratio, token_set_ratio
    from rapidfuzz.distance import JaroWinkler

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    _HAS_RAPIDFUZZ = False
    warnings.warn(
        "rapidfuzz not available — using difflib fallback (slower, scores differ). "
        "Install requirements.txt for canonical numbers."
    )


def _ratio_fallback(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzz_ratio(a: str, b: str) -> float:
    """0..1 character similarity (RapidFuzz ratio or difflib fallback)."""
    a, b = a or "", b or ""
    if _HAS_RAPIDFUZZ:
        return float(ratio(a, b) / 100.0)
    return float(_ratio_fallback(a, b))


def fuzz_token_set(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if _HAS_RAPIDFUZZ:
        return float(token_set_ratio(a, b) / 100.0)
    # Fallback: Jaccard over tokens blended with sequence ratio.
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    jac = len(sa & sb) / len(sa | sb)
    return float(0.5 * jac + 0.5 * _ratio_fallback(a, b))


def fuzz_wratio(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if _HAS_RAPIDFUZZ:
        return float(WRatio(a, b) / 100.0)
    return float(_ratio_fallback(a, b))


def fuzz_jaro_winkler(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if _HAS_RAPIDFUZZ:
        try:
            return float(JaroWinkler.normalized_similarity(a, b))
        except Exception:
            return float(_ratio_fallback(a, b))
    return float(_ratio_fallback(a, b))


# ---------------------------------------------------------------------------
# token helpers
# ---------------------------------------------------------------------------

def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a or "")), set(tokenize(b or ""))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_containment(a: str, b: str) -> Tuple[float, float]:
    """(containment of a in b, containment of b in a) over token sets."""
    sa, sb = set(tokenize(a or "")), set(tokenize(b or ""))
    if not sa and not sb:
        return 1.0, 1.0
    inter = len(sa & sb)
    return (
        (inter / len(sa)) if sa else 0.0,
        (inter / len(sb)) if sb else 0.0,
    )


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a or ""), len(b or "")
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


# ---------------------------------------------------------------------------
# numeric-address agreement
# ---------------------------------------------------------------------------

def numeric_agreement(a_addr: str, b_addr: str) -> Dict[str, float]:
    """Compare numeric token sets. Missing-number vs conflicting-number split.

    conflict = both sides HAVE numeric tokens but share NONE.
    one_side_missing = exactly one side has numeric tokens.
    """
    sa = set(extract_numeric_tokens(a_addr))
    sb = set(extract_numeric_tokens(b_addr))
    both_empty = (not sa) and (not sb)
    out: Dict[str, float] = {
        "numeric_exact_set": float(sa == sb and not both_empty),
        "numeric_any_overlap": float(bool(sa & sb)),
        "numeric_conflict": float(bool(sa) and bool(sb) and not (sa & sb)),
        "numeric_one_side_missing": float(bool(sa) != bool(sb)),
        "numeric_both_missing": float(both_empty),
        "numeric_jaccard": float(len(sa & sb) / len(sa | sb)) if (sa | sb) else 1.0,
    }
    pa, pb = set(extract_postcode_like_tokens(a_addr)), set(extract_postcode_like_tokens(b_addr))
    out["postcode_any_overlap"] = float(bool(pa & pb))
    out["postcode_both_present"] = float(bool(pa) and bool(pb))
    # "House number" heuristic: first numeric token of each address agrees.
    la, lb = sorted(sa), sorted(sb)
    out["house_number_agree"] = float(bool(sa) and bool(sb) and (la[0] == lb[0]))
    return out


# ---------------------------------------------------------------------------
# full pair feature vector
# ---------------------------------------------------------------------------

def country_equal(a: object, b: object) -> float:
    """Open-set country equality: stripped, case-insensitive string compare."""
    sa = ("" if a is None else str(a)).strip().lower()
    sb = ("" if b is None else str(b)).strip().lower()
    if not sa or not sb:
        return 0.0
    return float(sa == sb)


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_country: str,
    cand_name: str, cand_addr: str, cand_country: str,
    s1_name_norm: str, s1_addr_norm: str,
    cand_name_norm: str, cand_addr_norm: str,
) -> Dict[str, float]:
    """Feature dict for one (S1, candidate) pair."""
    s1_name_norm = s1_name_norm or ""
    cand_name_norm = cand_name_norm or ""
    s1_addr_norm = s1_addr_norm or ""
    cand_addr_norm = cand_addr_norm or ""

    name_exact = float(s1_name_norm == cand_name_norm and s1_name_norm != "")
    addr_exact = float(s1_addr_norm == cand_addr_norm and s1_addr_norm != "")
    cont_ab, cont_ba = token_containment(s1_addr_norm, cand_addr_norm)

    feats: Dict[str, float] = {
        # name
        "name_exact": name_exact,
        "name_ratio": fuzz_ratio(s1_name_norm, cand_name_norm),
        "name_token_set": fuzz_token_set(s1_name_norm, cand_name_norm),
        "name_wratio": fuzz_wratio(s1_name_norm, cand_name_norm),
        "name_jaro_winkler": fuzz_jaro_winkler(s1_name_norm, cand_name_norm),
        "name_len_s1": float(len(s1_name_norm)),
        "name_len_cand": float(len(cand_name_norm)),
        "name_len_ratio": length_ratio(s1_name_norm, cand_name_norm),
        "name_sorted_token_equal": float(
            sorted(tokenize(s1_name_norm)) == sorted(tokenize(cand_name_norm))
            and bool(tokenize(s1_name_norm))
        ),
        # address
        "addr_exact": addr_exact,
        "addr_ratio": fuzz_ratio(s1_addr_norm, cand_addr_norm),
        "addr_token_jaccard": token_jaccard(s1_addr_norm, cand_addr_norm),
        "addr_containment_s1_in_cand": float(cont_ab),
        "addr_containment_cand_in_s1": float(cont_ba),
        "addr_len_ratio": length_ratio(s1_addr_norm, cand_addr_norm),
        # country (open set)
        "country_equal": country_equal(s1_country, cand_country),
        # missingness
        "s1_name_missing": float(not (s1_name or "").strip()),
        "cand_name_missing": float(not (cand_name or "").strip()),
        "s1_addr_missing": float(not (s1_addr or "").strip()),
        "cand_addr_missing": float(not (cand_addr or "").strip()),
        "either_addr_missing": float(not (s1_addr or "").strip() or not (cand_addr or "").strip()),
    }
    feats.update(numeric_agreement(s1_addr or "", cand_addr or ""))
    return feats


# Columns produced by compute_pair_features (fixed order for summaries).
FEATURE_COLUMNS: List[str] = [
    "name_exact", "name_ratio", "name_token_set", "name_wratio",
    "name_jaro_winkler", "name_len_s1", "name_len_cand", "name_len_ratio",
    "name_sorted_token_equal", "addr_exact", "addr_ratio", "addr_token_jaccard",
    "addr_containment_s1_in_cand", "addr_containment_cand_in_s1",
    "addr_len_ratio", "country_equal", "s1_name_missing", "cand_name_missing",
    "s1_addr_missing", "cand_addr_missing", "either_addr_missing",
    "numeric_exact_set", "numeric_any_overlap", "numeric_conflict",
    "numeric_one_side_missing", "numeric_both_missing", "numeric_jaccard",
    "postcode_any_overlap", "postcode_both_present", "house_number_agree",
]


def add_pair_features(
    pairs: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    cand_lookup: pd.DataFrame,
    cols: Dict[str, str],
    label: int,
    neg_type: str = "",
) -> pd.DataFrame:
    """Attach raw/normalized fields + similarity features to pair rows.

    pairs must have [source1_entity_id, candidate_entity_id, source_pair].
    s1_lookup / cand_lookup must have entity_id, raw fields and the
    ``name_norm`` / ``address_norm`` columns (see eda.ensure_normalized_columns).
    """
    c_id, c_name, c_addr, c_cty = (
        cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]
    )
    s1 = s1_lookup.set_index(c_id)
    cand = cand_lookup.set_index(c_id)
    rows: List[Dict[str, object]] = []
    for r in pairs.itertuples(index=False):
        s1_id = str(r.source1_entity_id)
        cand_id = str(r.candidate_entity_id)
        try:
            a = s1.loc[s1_id]
            b = cand.loc[cand_id]
        except KeyError:
            continue  # dangling reference — counted in validation, skipped here
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        feats = compute_pair_features(
            str(a[c_name]), str(a[c_addr]), str(a[c_cty]),
            str(b[c_name]), str(b[c_addr]), str(b[c_cty]),
            str(a["name_norm"]), str(a["address_norm"]),
            str(b["name_norm"]), str(b["address_norm"]),
        )
        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "source_pair": str(r.source_pair),
                "s1_country": str(a[c_cty]),
                "cand_country": str(b[c_cty]),
                "s1_name_raw": str(a[c_name]),
                "cand_name_raw": str(b[c_name]),
                "s1_addr_raw": str(a[c_addr]),
                "cand_addr_raw": str(b[c_addr]),
                "s1_name_norm": str(a["name_norm"]),
                "cand_name_norm": str(b["name_norm"]),
                "s1_addr_norm": str(a["address_norm"]),
                "cand_addr_norm": str(b["address_norm"]),
                "label": label,
                "neg_type": neg_type,
                **feats,
            }
        )
    out = pd.DataFrame(rows)
    return out


def summarize_feature_frame(
    df: pd.DataFrame, group_cols: List[str], feature_cols: List[str]
) -> pd.DataFrame:
    """Long-format summary: per group x feature -> n, mean, p50, p10, p90, exact-rate."""
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, "feature", "n", "mean", "p10", "p50", "p90", "rate_eq_1"]
        )
    records: List[Dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [([], df)]
    for keys, grp in grouped:
        if group_cols:
            keys = list(keys) if isinstance(keys, tuple) else [keys]
        else:
            keys = []
        for feat in feature_cols:
            if feat not in grp.columns:
                continue
            vals = pd.to_numeric(grp[feat], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            records.append(
                {
                    **{g: k for g, k in zip(group_cols, keys)},
                    "feature": feat,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "p10": float(vals.quantile(0.10)),
                    "p50": float(vals.quantile(0.50)),
                    "p90": float(vals.quantile(0.90)),
                    "rate_eq_1": float((vals == 1.0).mean()),
                }
            )
    return pd.DataFrame(
        records,
        columns=[*group_cols, "feature", "n", "mean", "p10", "p50", "p90", "rate_eq_1"],
    )
