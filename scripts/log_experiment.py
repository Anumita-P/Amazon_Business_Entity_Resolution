#!/usr/bin/env python3
"""Human-facing experiment logger -> appends one row to logs/experiment_log.csv.

Interactive mode (asks questions):
    python scripts/log_experiment.py

Non-interactive mode:
    python scripts/log_experiment.py --team-member Asha --experiment-type new ^
        --description "TF-IDF name top-k 20->50" --hypothesis "..." --day 1

Quick rerun record (same code/config/data, run again):
    python scripts/log_experiment.py --rerun EXP-003 --team-member Asha

Show recent rows:
    python scripts/log_experiment.py --list 10

Experiment types: new | rerun | submission | analysis | fix
  - new        : something substantive changed (see logs/README.md)
  - rerun      : identical code/config/data re-executed (set --parent)
  - submission : leaderboard upload (set --submission-number + score later)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import find_project_root, load_config
from business_entity_resolution.utils import get_git_commit

COLUMNS = [
    "experiment_id", "timestamp", "day", "team_member", "experiment_type",
    "parent_experiment_id", "description", "hypothesis", "code_version",
    "data_version", "normalization_version", "blocking_version",
    "feature_version", "model_version", "hyperparameters_changed",
    "hyperparameters", "validation_split", "local_f05", "local_precision",
    "local_recall", "candidate_recall", "avg_candidates_per_s1",
    "max_candidates_per_s1", "public_submission", "submission_number",
    "public_leaderboard_score", "status", "notes",
]

EXPERIMENT_TYPES = ["new", "rerun", "submission", "analysis", "fix"]


def log_path_for(project_root: Path) -> Path:
    return project_root / "logs" / "experiment_log.csv"


def ensure_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(COLUMNS)


def read_rows(path: Path) -> list[dict]:
    ensure_log(path)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def next_experiment_id(path: Path) -> str:
    rows = read_rows(path)
    best = 0
    for r in rows:
        eid = (r.get("experiment_id") or "").strip()
        if eid.upper().startswith("EXP-"):
            try:
                best = max(best, int(eid.split("-", 1)[1]))
            except ValueError:
                pass
    return f"EXP-{best + 1:03d}"


def append_row(path: Path, row: dict) -> None:
    ensure_log(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    if header != COLUMNS:
        raise SystemExit(
            f"Refusing to append: {path} header does not match expected columns.\n"
            f"Expected: {COLUMNS}\nFound: {header}"
        )
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=COLUMNS).writerow({c: row.get(c, "") for c in COLUMNS})


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        val = ""
    return val or default


def interactive(project_root: Path) -> dict:
    print("\n==== Log a team experiment (Stage 1 EDA ok — model fields may be blank) ====")
    print(f"Existing log: {log_path_for(project_root)}")
    exp_type = ask(f"experiment_type {EXPERIMENT_TYPES}", "new")
    if exp_type not in EXPERIMENT_TYPES:
        print(f"Unknown type '{exp_type}' — using 'new'.")
        exp_type = "new"
    row = {
        "experiment_id": next_experiment_id(log_path_for(project_root)),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "day": ask("day (1/2/3)", "1"),
        "team_member": ask("team member (your name)"),
        "experiment_type": exp_type,
        "parent_experiment_id": ask("parent_experiment_id (required for rerun; blank otherwise)"),
        "description": ask("description (what changed / what this run is)"),
        "hypothesis": ask("hypothesis (what you expect to learn)"),
        "code_version": ask("code_version (git hash or NA)", get_git_commit(project_root)),
        "data_version": ask("data_version", "train-v1"),
        "normalization_version": ask("normalization_version", ""),
        "blocking_version": ask("blocking_version", ""),
        "feature_version": ask("feature_version", ""),
        "model_version": ask("model_version (NA during EDA)", "NA-eda"),
        "hyperparameters_changed": ask("hyperparameters_changed (yes/no)", "no" if exp_type == "rerun" else ""),
        "hyperparameters": ask("hyperparameters (key=value; ...)", ""),
        "validation_split": ask("validation_split", ""),
        "local_f05": ask("local_f05", ""),
        "local_precision": ask("local_precision", ""),
        "local_recall": ask("local_recall", ""),
        "candidate_recall": ask("candidate_recall", ""),
        "avg_candidates_per_s1": ask("avg_candidates_per_s1", ""),
        "max_candidates_per_s1": ask("max_candidates_per_s1", ""),
        "public_submission": ask("public_submission (yes/no)", "yes" if exp_type == "submission" else "no"),
        "submission_number": ask("submission_number (1-5/day, blank if none)", ""),
        "public_leaderboard_score": ask("public_leaderboard_score (fill after scoring)", ""),
        "status": ask("status (done/running/failed/...)", "done"),
        "notes": ask("notes", ""),
    }
    if exp_type == "rerun" and not row["parent_experiment_id"]:
        print("WARNING: rerun without parent_experiment_id — lineage will be unclear.")
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append an experiment to logs/experiment_log.csv")
    ap.add_argument("--list", type=int, default=None, metavar="N",
                    help="Show the last N experiments and exit.")
    ap.add_argument("--rerun", default=None, metavar="EXP-ID",
                    help="Record a rerun of the given parent experiment id.")
    for col in COLUMNS:
        if col in ("experiment_id", "timestamp"):
            continue
        ap.add_argument(f"--{col.replace('_', '-')}", default=None, help=f"Set {col}.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    path = log_path_for(cfg.project_root)

    if args.list is not None:
        rows = read_rows(path)
        tail = rows[-args.list:] if args.list > 0 else rows
        print(f"\n{path} — showing {len(tail)} of {len(rows)} row(s):")
        for r in tail:
            print(f"  {r.get('experiment_id')} | {r.get('timestamp')} | {r.get('team_member')} | "
                  f"{r.get('experiment_type')} | parent={r.get('parent_experiment_id') or '-'} | "
                  f"{(r.get('description') or '')[:90]}")
        return 0

    if args.rerun:
        row = {c: "" for c in COLUMNS}
        row["experiment_id"] = next_experiment_id(path)
        row["timestamp"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        row["experiment_type"] = "rerun"
        row["parent_experiment_id"] = args.rerun
        row["hyperparameters_changed"] = "no"
        row["public_submission"] = "no"
        row["code_version"] = get_git_commit(cfg.project_root)
        for col in COLUMNS:
            flag = col.replace("_", "-")
            val = getattr(args, flag.replace("-", "_"), None)
            if val is not None and col not in ("experiment_id", "timestamp"):
                row[col] = val
        if not row.get("description"):
            row["description"] = f"RERUN of {args.rerun} (identical code/config/data)"
    elif any(getattr(args, c.replace("_", "-").replace("-", "_"), None) is not None for c in COLUMNS):
        row = {c: "" for c in COLUMNS}
        row["experiment_id"] = next_experiment_id(path)
        row["timestamp"] = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        for col in COLUMNS:
            attr = col.replace("-", "_")
            val = getattr(args, attr, None)
            if val is not None and col not in ("experiment_id", "timestamp"):
                row[col] = val
        if not row.get("experiment_type"):
            row["experiment_type"] = "new"
        if not row.get("code_version"):
            row["code_version"] = get_git_commit(cfg.project_root)
        if not row.get("public_submission"):
            row["public_submission"] = "no"
    else:
        row = interactive(cfg.project_root)

    print("\n---- Row to append ----")
    for c in COLUMNS:
        print(f"  {c}: {row.get(c, '')}")
    if not args.yes:
        try:
            ok = input("\nAppend this row? [y/N]: ").strip().lower()
        except EOFError:
            ok = ""
        if ok not in ("y", "yes"):
            print("Aborted — nothing written.")
            return 1
    append_row(path, row)
    print(f"\nRecorded {row['experiment_id']} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
