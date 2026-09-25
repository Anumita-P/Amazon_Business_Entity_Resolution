"""Dataset integrity validation (Stage 1).

Checks TSV parsing, expected columns, ID prefixes, duplicate IDs, unexpected
source prefixes and ground-truth referential integrity. Never touches the
network; never mutates data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .config import AppConfig
from .io import parse_matched_list


@dataclass
class TableValidation:
    table: str
    path: str
    exists: bool
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    tables: List[TableValidation] = field(default_factory=list)
    ground_truth_errors: List[str] = field(default_factory=list)
    ground_truth_warnings: List[str] = field(default_factory=list)
    ground_truth_stats: Dict[str, object] = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(len(t.errors) for t in self.tables) + len(self.ground_truth_errors)

    @property
    def n_warnings(self) -> int:
        return sum(len(t.warnings) for t in self.tables) + len(self.ground_truth_warnings)

    def to_console(self) -> str:
        lines = ["", "==== DATA VALIDATION REPORT ===="]
        for t in self.tables:
            status = "OK " if t.exists and not t.errors else ("MISS" if not t.exists else "FAIL")
            lines.append(f"[{status}] {t.table}: {t.path} (rows={t.rows})")
            for e in t.errors:
                lines.append(f"       ERROR: {e}")
            for w in t.warnings:
                lines.append(f"       warn : {w}")
        if self.ground_truth_errors or self.ground_truth_warnings or self.ground_truth_stats:
            lines.append("[ GT ] ground-truth referential integrity")
            for e in self.ground_truth_errors:
                lines.append(f"       ERROR: {e}")
            for w in self.ground_truth_warnings:
                lines.append(f"       warn : {w}")
            for k, v in self.ground_truth_stats.items():
                lines.append(f"       stat : {k} = {v}")
        lines.append(f"TOTAL: {self.n_errors} error(s), {self.n_warnings} warning(s)")
        lines.append("================================")
        return "\n".join(lines)


def validate_source_table(
    df: Optional[pd.DataFrame],
    table_key: str,
    path: str | Path,
    expected_columns: List[str],
    expected_prefix: Optional[str],
    id_column: str,
) -> TableValidation:
    rep = TableValidation(table=table_key, path=str(path), exists=df is not None)
    if df is None:
        rep.warnings.append("file not found — downstream steps using it will be skipped")
        return rep
    rep.rows = len(df)
    rep.columns = list(df.columns)

    # 1. single-column smell => TSV was (or would be) mis-parsed
    if df.shape[1] == 1:
        rep.errors.append(
            f"only 1 column parsed ({list(df.columns)}). "
            'File must be TAB-separated; read with sep="\\t".'
        )
    # 2. expected columns
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        rep.errors.append(f"missing expected columns: {missing}")
        return rep
    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        rep.warnings.append(f"unexpected extra columns (ignored): {extra}")

    ids = df[id_column].astype(str)
    # 3. empty IDs
    n_empty = int((ids.str.strip() == "").sum())
    if n_empty:
        rep.errors.append(f"{n_empty} empty entity_id values")
    # 4. duplicate IDs
    dup_mask = ids.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_ids = int(ids[dup_mask].nunique()) if n_dup_rows else 0
    rep.stats["duplicate_id_rows"] = n_dup_rows
    rep.stats["duplicate_ids"] = n_dup_ids
    if n_dup_rows:
        rep.errors.append(f"{n_dup_rows} rows share {n_dup_ids} duplicate entity_id values")
    # 5. prefix checks
    if expected_prefix:
        bad = ids[~ids.str.startswith(expected_prefix)]
        if len(bad):
            sample = sorted(set(bad.str.slice(0, 4)))[:8]
            rep.errors.append(
                f"{len(bad)} IDs do not start with expected prefix '{expected_prefix}' "
                f"(observed prefixes e.g. {sample})"
            )
    # unexpected S1/S2/S3 prefixes anywhere
    for pref in ("S1-", "S2-", "S3-"):
        if pref != expected_prefix:
            n = int(ids.str.startswith(pref).sum())
            if n:
                rep.warnings.append(f"{n} IDs carry unexpected prefix {pref} in {table_key}")
    # 6. embedded tabs/newlines inside fields (malformed rows)
    for col in expected_columns:
        if col == id_column:
            continue
        series = df[col].astype(str)
        n_tab = int(series.str.contains("\t").sum())
        n_nl = int(series.str.contains("\n").sum())
        if n_tab:
            rep.warnings.append(f"column '{col}': {n_tab} values contain embedded TABs")
        if n_nl:
            rep.warnings.append(f"column '{col}': {n_nl} values contain embedded newlines")
    return rep


def validate_ground_truth(
    gt: Optional[pd.DataFrame],
    s1_ids: Set[str],
    s2_ids: Set[str],
    s3_ids: Set[str],
    cols: Dict[str, str],
) -> tuple[List[str], List[str], Dict[str, object]]:
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, object] = {}
    if gt is None:
        warnings.append("train_ground_truth.tsv not found — supervised EDA sections will be skipped")
        return errors, warnings, stats
    c_s1, c_match = cols["gt_source1"], cols["gt_matched"]
    for c in (c_s1, c_match):
        if c not in gt.columns:
            errors.append(f"ground truth missing column '{c}'")
            return errors, warnings, stats

    gt_ids = gt[c_s1].astype(str)
    n_dup = int(gt_ids.duplicated().sum())
    stats["gt_rows"] = int(len(gt))
    stats["gt_unique_s1"] = int(gt_ids.nunique())
    if n_dup:
        errors.append(f"{n_dup} duplicate source1_entity_id rows in ground truth")

    unknown_s1 = sorted(set(gt_ids) - s1_ids) if s1_ids else []
    if unknown_s1:
        errors.append(
            f"{len(unknown_s1)} ground-truth S1 ids not found in train_source1 "
            f"(e.g. {unknown_s1[:5]})"
        )
    missing_gt = sorted(s1_ids - set(gt_ids)) if s1_ids else []
    if missing_gt:
        warnings.append(
            f"{len(missing_gt)} train S1 ids have no ground-truth row (e.g. {missing_gt[:5]})"
        )

    pool = s2_ids | s3_ids
    n_pos = n_s2 = n_s3 = 0
    n_unknown = n_self = n_dup_in_list = n_bad_prefix = 0
    for cell in gt[c_match].astype(str):
        ids = parse_matched_list(cell)
        if len(ids) != len(set(ids)):
            n_dup_in_list += 1
        for mid in ids:
            n_pos += 1
            if mid.startswith("S2-"):
                n_s2 += 1
            elif mid.startswith("S3-"):
                n_s3 += 1
            elif mid.startswith("S1-"):
                n_self += 1
            else:
                n_bad_prefix += 1
            if mid not in pool:
                n_unknown += 1
    stats.update(
        {
            "positive_pairs": n_pos,
            "positive_s1_s2": n_s2,
            "positive_s1_s3": n_s3,
            "unknown_candidate_ids": n_unknown,
            "self_s1_references": n_self,
            "rows_with_dup_ids_in_list": n_dup_in_list,
            "bad_prefix_ids": n_bad_prefix,
        }
    )
    if n_unknown:
        errors.append(f"{n_unknown} matched ids do not exist in train S2/S3")
    if n_self:
        errors.append(f"{n_self} matched ids are S1 self-references (must be S2/S3 only)")
    if n_dup_in_list:
        errors.append(f"{n_dup_in_list} ground-truth rows contain duplicate ids in the list")
    if n_bad_prefix:
        errors.append(f"{n_bad_prefix} matched ids have unexpected prefixes (not S2-/S3-)")
    if n_pos == 0:
        warnings.append("ground truth contains zero positive pairs")
    return errors, warnings, stats


def run_all_validations(
    cfg: AppConfig,
    train: Dict[str, Optional[pd.DataFrame]],
    test: Dict[str, Optional[pd.DataFrame]],
) -> ValidationReport:
    cols = cfg.columns
    src_cols = [cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]]
    prefixes = cfg.id_prefixes
    train_paths, test_paths = cfg.train_paths(), cfg.test_paths()

    report = ValidationReport()
    mapping = [
        ("train_source1", train.get("train_s1"), train_paths["train_source1"], prefixes.get("train_source1", "S1-")),
        ("train_source2", train.get("train_s2"), train_paths["train_source2"], prefixes.get("train_source2", "S2-")),
        ("train_source3", train.get("train_s3"), train_paths["train_source3"], prefixes.get("train_source3", "S3-")),
        ("test_source1", test.get("test_s1"), test_paths["test_source1"], prefixes.get("test_source1", "S1-")),
        ("test_source2", test.get("test_s2"), test_paths["test_source2"], prefixes.get("test_source2", "S2-")),
        ("test_source3", test.get("test_s3"), test_paths["test_source3"], prefixes.get("test_source3", "S3-")),
    ]
    for key, df, path, pref in mapping:
        report.tables.append(
            validate_source_table(df, key, path, src_cols, pref, cols["entity_id"])
        )

    gt = train.get("train_gt")
    gt_rep = TableValidation(
        table="train_ground_truth",
        path=str(train_paths["train_ground_truth"]),
        exists=gt is not None,
        rows=0 if gt is None else len(gt),
        columns=[] if gt is None else list(gt.columns),
    )
    if gt is None:
        gt_rep.warnings.append("file not found — supervised EDA sections will be skipped")
    elif gt.shape[1] == 1:
        gt_rep.errors.append('only 1 column parsed; file must be TAB-separated (sep="\\t")')
    report.tables.append(gt_rep)

    def ids_of(df: Optional[pd.DataFrame]) -> Set[str]:
        if df is None or cols["entity_id"] not in df.columns:
            return set()
        return set(df[cols["entity_id"]].astype(str))

    e, w, s = validate_ground_truth(
        gt, ids_of(train.get("train_s1")), ids_of(train.get("train_s2")), ids_of(train.get("train_s3")), cols
    )
    report.ground_truth_errors, report.ground_truth_warnings, report.ground_truth_stats = e, w, s
    return report
