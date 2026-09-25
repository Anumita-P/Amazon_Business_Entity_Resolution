"""Configuration loading and path resolution.

Precedence for the dataset root:
    1. explicit CLI argument (--data-root)
    2. environment variable BER_DATA_ROOT (or BER_CONFIG_DIR-independent)
    3. config/config.yaml `data_root` value (default: "dataset")

All relative paths are resolved against the repository root so the project
works identically on Windows / macOS / Linux without editing source code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


ENV_DATA_ROOT = "BER_DATA_ROOT"
ENV_CONFIG = "BER_CONFIG"


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the repo root (folder containing config/config.yaml).

    Walks up from `start` (default: this file's location). Falls back to the
    current working directory if nothing is found.
    """
    here = Path(start) if start else Path(__file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "config.yaml").exists():
            return candidate
    return Path.cwd()


@dataclass
class AppConfig:
    raw: Dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=Path.cwd)
    config_path: Optional[Path] = None
    data_root: Path = field(default_factory=lambda: Path("dataset"))

    # resolved output dirs
    eda_dir: Path = field(default_factory=lambda: Path("eda"))
    figures_dir: Path = field(default_factory=lambda: Path("eda/figures"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    seed: int = 42

    @property
    def eda(self) -> Dict[str, Any]:
        return dict(self.raw.get("eda", {}))

    @property
    def columns(self) -> Dict[str, str]:
        return dict(
            self.raw.get(
                "columns",
                {
                    "entity_id": "entity_id",
                    "business_name": "business_name",
                    "business_address": "business_address",
                    "country": "country",
                    "gt_source1": "source1_entity_id",
                    "gt_matched": "matched_entity_ids",
                },
            )
        )

    @property
    def id_prefixes(self) -> Dict[str, str]:
        return dict(self.raw.get("id_prefixes", {}))

    def config_hash(self) -> str:
        """Short stable hash of the effective configuration (for run logging)."""
        canonical = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    # -- dataset file layout -------------------------------------------------
    def train_paths(self) -> Dict[str, Path]:
        d = self.data_root / "train"
        return {
            "train_source1": d / "train_source1.tsv",
            "train_source2": d / "train_source2.tsv",
            "train_source3": d / "train_source3.tsv",
            "train_ground_truth": d / "train_ground_truth.tsv",
        }

    def test_paths(self) -> Dict[str, Path]:
        d = self.data_root / "test"
        return {
            "test_source1": d / "test_source1.tsv",
            "test_source2": d / "test_source2.tsv",
            "test_source3": d / "test_source3.tsv",
        }


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def load_config(
    config_path: Optional[str | Path] = None,
    data_root_override: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
) -> AppConfig:
    """Load YAML config and resolve all paths against the project root."""
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required: pip install -r requirements.txt")

    root = Path(project_root) if project_root else find_project_root()

    cfg_path: Optional[Path] = None
    if config_path:
        cfg_path = _resolve(Path(config_path), root)
    elif os.environ.get(ENV_CONFIG):
        cfg_path = _resolve(Path(os.environ[ENV_CONFIG]), root)
    else:
        default = root / "config" / "config.yaml"
        cfg_path = default if default.exists() else None

    raw: Dict[str, Any] = {}
    if cfg_path and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # data_root precedence: CLI > env > yaml > default
    data_root_raw: Any = data_root_override or os.environ.get(ENV_DATA_ROOT) or raw.get(
        "data_root", "dataset"
    )
    data_root = _resolve(Path(str(data_root_raw)), root)

    outputs = raw.get("outputs", {}) or {}

    def out(name: str, default: str) -> Path:
        return _resolve(Path(str(outputs.get(name, default))), root)

    cfg = AppConfig(
        raw=raw,
        project_root=root,
        config_path=cfg_path,
        data_root=data_root,
        eda_dir=out("eda_dir", "eda"),
        figures_dir=out("figures_dir", "eda/figures"),
        reports_dir=out("reports_dir", "reports"),
        logs_dir=out("logs_dir", "logs"),
        output_dir=out("output_dir", "output"),
        seed=int(raw.get("seed", 42)),
    )
    for d in (cfg.eda_dir, cfg.figures_dir, cfg.reports_dir, cfg.logs_dir, cfg.output_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
