"""Shared utilities: seeding, logging, run-history recording, timing."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# seeding / logging
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ber")


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    start = time.time()
    yield
    elapsed = time.time() - start
    (logger or logging.getLogger("ber")).info("%s in %.1fs", name, elapsed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# environment introspection (offline-safe)
# ---------------------------------------------------------------------------

def get_git_commit(project_root: Optional[str | Path] = None) -> str:
    """Short git HEAD hash, or "not_available" (never raises, never uses net)."""
    try:
        cwd = str(project_root) if project_root else None
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "not_available"


def get_python_version() -> str:
    return sys.version.replace("\n", " ")


KEY_PACKAGES = [
    "pandas", "numpy", "scipy", "sklearn", "rapidfuzz",
    "matplotlib", "seaborn", "yaml",
]


def get_key_package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for mod in KEY_PACKAGES:
        try:
            m = importlib.import_module(mod)
            versions[mod] = str(getattr(m, "__version__", "installed"))
        except Exception:
            versions[mod] = "missing"
    return versions


def sha256_short(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# run history (machine-readable log of every pipeline execution)
# ---------------------------------------------------------------------------

RUN_HISTORY_FILENAME = "run_history.jsonl"


def run_history_path(logs_dir: str | Path) -> Path:
    return Path(logs_dir) / RUN_HISTORY_FILENAME


def read_last_run(logs_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = run_history_path(logs_dir)
    if not path.exists():
        return None
    last: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
    except OSError:
        return None
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def runs_look_identical(prev: Optional[Dict[str, Any]], curr: Dict[str, Any]) -> bool:
    """Heuristic: same command + config hash + code version + dataset mtimes."""
    if not prev:
        return False
    keys = ["command", "config_hash", "git_commit"]
    if any(prev.get(k) != curr.get(k) for k in keys):
        return False
    prev_files = (prev.get("dataset_files") or {})
    curr_files = (curr.get("dataset_files") or {})
    if set(prev_files) != set(curr_files):
        return False
    for k in curr_files:
        if (prev_files[k] or {}).get("mtime_utc") != (curr_files[k] or {}).get("mtime_utc"):
            return False
        if (prev_files[k] or {}).get("size_bytes") != (curr_files[k] or {}).get("size_bytes"):
            return False
    return True


def record_run(logs_dir: str | Path, record: Dict[str, Any]) -> Dict[str, Any]:
    """Append one JSON line to logs/run_history.jsonl (creates dirs as needed)."""
    ensure_dir(logs_dir)
    prev = read_last_run(logs_dir)
    record = dict(record)
    record.setdefault("timestamp_utc", now_iso())
    record["identical_to_previous_run"] = runs_look_identical(prev, record)
    path = run_history_path(logs_dir)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def command_argv() -> List[str]:
    return list(sys.argv)
