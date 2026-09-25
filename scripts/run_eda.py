#!/usr/bin/env python3
"""Run the full Stage-1 pipeline: load -> validate -> EDA -> artifacts -> summary.

Usage (Windows / VS Code):
    python scripts/run_eda.py --data-root "C:\\path\\to\\dataset"
    python scripts/run_eda.py --data-root dataset          # dataset/ inside repo root
    set BER_DATA_ROOT=C:\\path\\to\\dataset && python scripts/run_eda.py

Every execution appends a record to logs/run_history.jsonl. No internet access
is required or used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow `python scripts/run_eda.py` from the repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.eda import run_full_eda
from business_entity_resolution.io import dataset_file_inventory, load_test_tables, load_training_tables
from business_entity_resolution.utils import (
    command_argv,
    get_git_commit,
    get_key_package_versions,
    get_python_version,
    now_iso,
    record_run,
    seed_everything,
    setup_logging,
)
from business_entity_resolution.validation import run_all_validations


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 1: dataset audit + entity-resolution EDA")
    ap.add_argument("--data-root", default=None,
                    help="Path to challenge dataset/ dir (train/*.tsv, test/*.tsv). "
                         "Overrides BER_DATA_ROOT env var and config.yaml.")
    ap.add_argument("--config", default=None, help="Path to config YAML (default: config/config.yaml).")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed.")
    ap.add_argument("--no-run-log", action="store_true",
                    help="Skip appending to logs/run_history.jsonl.")
    return ap.parse_args()


def main() -> int:
    logger = setup_logging()
    args = parse_args()
    started = time.time()

    cfg = load_config(config_path=args.config, data_root_override=args.data_root)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.raw["seed"] = args.seed
        cfg.raw.setdefault("eda", {})["random_seed"] = args.seed
    seed_everything(int(cfg.eda.get("random_seed", cfg.seed)))

    logger.info("project_root : %s", cfg.project_root)
    logger.info("data_root    : %s", cfg.data_root)
    logger.info("config       : %s (hash %s)", cfg.config_path, cfg.config_hash())

    # 1-2. load + validate ---------------------------------------------------
    train = load_training_tables(cfg)
    test = load_test_tables(cfg)
    report = run_all_validations(cfg, train, test)
    print(report.to_console())
    if train.get("train_s1") is None and train.get("train_s2") is None and train.get("train_s3") is None:
        logger.error("No training source files found under %s — nothing to do.", cfg.data_root / "train")
        return 2

    # 3-7. EDA ----------------------------------------------------------------
    run_meta = {
        "timestamp_utc": now_iso(),
        "command": " ".join(command_argv()),
        "script": "scripts/run_eda.py",
        "git_commit": get_git_commit(cfg.project_root),
        "python_version": get_python_version(),
        "package_versions": get_key_package_versions(),
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "config_hash": cfg.config_hash(),
        "data_root": str(cfg.data_root),
        "dataset_files": dataset_file_inventory(cfg),
        "seed": int(cfg.eda.get("random_seed", cfg.seed)),
    }
    outcome = run_full_eda(cfg, train, test, run_meta)

    # 8. run log ---------------------------------------------------------------
    elapsed = time.time() - started
    logger.info("EDA complete in %.1fs — %d artifact(s):", elapsed, len(outcome["artifacts"]))
    for a in outcome["artifacts"]:
        logger.info("  - %s", a)

    if not args.no_run_log:
        record = dict(run_meta)
        record["runtime_seconds"] = round(elapsed, 1)
        record["status"] = "success"
        record["n_errors_validation"] = report.n_errors
        record["n_warnings_validation"] = report.n_warnings
        record["generated_artifacts"] = outcome["artifacts"]
        saved = record_run(cfg.logs_dir, record)
        if saved.get("identical_to_previous_run"):
            logger.info("NOTE: this run looks IDENTICAL to the previous run "
                        "(same command/config/code/dataset mtimes) — do not log it as a new experiment.")
        else:
            logger.info("Run recorded in %s", cfg.logs_dir / "run_history.jsonl")
    if report.n_errors:
        logger.warning("Validation reported %d error(s) — see report above.", report.n_errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
