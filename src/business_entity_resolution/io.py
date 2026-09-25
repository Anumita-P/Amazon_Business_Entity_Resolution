"""Dataset I/O helpers.

All challenge files are TAB-separated. Every read uses ``sep="\\t"`` and
``dtype=str`` so IDs / PIN codes are never coerced to numbers and parsing is
identical on every machine.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import AppConfig

TSV_SEP = "\t"


# --------------------------------------------------------------------------
# low-level reading
# --------------------------------------------------------------------------

def read_tsv(
    path: str | Path,
    expected_columns: Optional[List[str]] = None,
    strict_columns: bool = False,
) -> pd.DataFrame:
    """Read a challenge TSV with ``sep="\\t"`` and ``dtype=str``.

    Empty cells become "" (never NaN) so downstream string code is total.
    Raises FileNotFoundError if missing, ValueError on column mismatch when
    strict_columns=True.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    df = pd.read_csv(
        path,
        sep=TSV_SEP,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python",
    )
    # Defensive: strip a UTF-8 BOM from the first column name if present.
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    # Ensure every value is a str (paranoia for mixed-type edge cases).
    for col in df.columns:
        df[col] = df[col].astype(str)
    if expected_columns is not None and strict_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def try_read_tsv(
    path: str | Path,
    expected_columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """read_tsv that returns None (instead of raising) when the file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    return read_tsv(path, expected_columns=expected_columns, strict_columns=False)


# --------------------------------------------------------------------------
# challenge layout
# --------------------------------------------------------------------------

def load_training_tables(cfg: AppConfig) -> Dict[str, Optional[pd.DataFrame]]:
    """Load train_source1/2/3 + ground truth (None for any missing file)."""
    cols = cfg.columns
    src_cols = [cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]]
    gt_cols = [cols["gt_source1"], cols["gt_matched"]]
    paths = cfg.train_paths()
    return {
        "train_s1": try_read_tsv(paths["train_source1"], src_cols),
        "train_s2": try_read_tsv(paths["train_source2"], src_cols),
        "train_s3": try_read_tsv(paths["train_source3"], src_cols),
        "train_gt": try_read_tsv(paths["train_ground_truth"], gt_cols),
    }


def load_test_tables(cfg: AppConfig) -> Dict[str, Optional[pd.DataFrame]]:
    """Load test_source1/2/3 (None for any missing file)."""
    cols = cfg.columns
    src_cols = [cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]]
    paths = cfg.test_paths()
    return {
        "test_s1": try_read_tsv(paths["test_source1"], src_cols),
        "test_s2": try_read_tsv(paths["test_source2"], src_cols),
        "test_s3": try_read_tsv(paths["test_source3"], src_cols),
    }


# --------------------------------------------------------------------------
# ground truth helpers
# --------------------------------------------------------------------------

def parse_matched_list(cell: str) -> List[str]:
    """Parse a `matched_entity_ids` cell into a clean ID list.

    Empty/blank cells -> []. Whitespace is stripped; empty fragments dropped.
    """
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [frag.strip() for frag in text.split(",") if frag.strip()]


def expand_ground_truth(
    gt_df: pd.DataFrame, cols: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_s1_stats, positive_pairs).

    per_s1_stats: one row per Source-1 entity with n_matches, n_s2, n_s3,
        singleton / s2_only / s3_only / mixed flags.
    positive_pairs: one row per (source1_entity_id, candidate_entity_id) with
        source_pair in {"S1_S2", "S1_S3"}.
    """
    c_s1, c_match = cols["gt_source1"], cols["gt_matched"]
    gt = gt_df.copy()
    gt["_matched_list"] = gt[c_s1].astype(str)  # placeholder replaced below
    gt["_matched_list"] = gt[c_match].map(parse_matched_list)
    gt["n_matches"] = gt["_matched_list"].map(len)
    gt["n_s2_matches"] = gt["_matched_list"].map(
        lambda ids: sum(1 for x in ids if x.startswith("S2-"))
    )
    gt["n_s3_matches"] = gt["_matched_list"].map(
        lambda ids: sum(1 for x in ids if x.startswith("S3-"))
    )
    gt["is_singleton"] = (gt["n_matches"] == 0).astype(int)
    gt["is_s2_only"] = ((gt["n_s2_matches"] > 0) & (gt["n_s3_matches"] == 0)).astype(int)
    gt["is_s3_only"] = ((gt["n_s3_matches"] > 0) & (gt["n_s2_matches"] == 0)).astype(int)
    gt["is_mixed"] = ((gt["n_s2_matches"] > 0) & (gt["n_s3_matches"] > 0)).astype(int)

    rows: List[Dict[str, str]] = []
    for s1_id, matched in zip(gt[c_s1].astype(str), gt["_matched_list"]):
        for mid in matched:
            if mid.startswith("S2-"):
                pair = "S1_S2"
            elif mid.startswith("S3-"):
                pair = "S1_S3"
            else:
                pair = "S1_OTHER"  # unexpected prefix — flagged by validation
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": mid,
                    "source_pair": pair,
                }
            )
    positives = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "source_pair"])
    return gt, positives


# --------------------------------------------------------------------------
# file metadata (for run logging)
# --------------------------------------------------------------------------

def file_mtime_iso(path: str | Path) -> Optional[str]:
    try:
        ts = os.path.getmtime(path)
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def dataset_file_inventory(cfg: AppConfig) -> Dict[str, Dict[str, Optional[str]]]:
    """Map every expected dataset file -> {exists, mtime_utc, size_bytes}."""
    inv: Dict[str, Dict[str, Optional[str]]] = {}
    all_paths = {**cfg.train_paths(), **cfg.test_paths()}
    for key, p in all_paths.items():
        exists = p.exists()
        size: Optional[str] = None
        if exists:
            try:
                size = str(p.stat().st_size)
            except OSError:
                size = None
        inv[key] = {
            "path": str(p),
            "exists": "yes" if exists else "no",
            "mtime_utc": file_mtime_iso(p),
            "size_bytes": size,
        }
    return inv
