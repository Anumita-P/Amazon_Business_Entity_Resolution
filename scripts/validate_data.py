#!/usr/bin/env python3
"""Validate the challenge dataset without running EDA.

Usage:
    python scripts/validate_data.py --data-root "C:\\path\\to\\dataset"
    python scripts/validate_data.py --data-root dataset --report validation_report.txt

Exit code 0 = no errors (warnings allowed), 1 = validation errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_test_tables, load_training_tables
from business_entity_resolution.utils import setup_logging
from business_entity_resolution.validation import run_all_validations


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate challenge TSVs (offline).")
    ap.add_argument("--data-root", default=None, help="Path to challenge dataset/ dir.")
    ap.add_argument("--config", default=None, help="Path to config YAML.")
    ap.add_argument("--report", default=None, help="Optional path to also write the report.")
    return ap.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_config(config_path=args.config, data_root_override=args.data_root)
    train = load_training_tables(cfg)
    test = load_test_tables(cfg)
    report = run_all_validations(cfg, train, test)
    text = report.to_console()
    print(text)
    if args.report:
        Path(args.report).write_text(
            f"data_root: {cfg.data_root}\nconfig: {cfg.config_path}\n{text}\n", encoding="utf-8"
        )
        print(f"Report written to {args.report}")
    return 1 if report.n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
