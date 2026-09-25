"""Business Entity Resolution — Amazon ML Challenge 2026.

Stage 1 implements dataset audit + entity-resolution EDA only.
Later stages (blocking pipeline, pair classifier, thresholding, inference)
will extend this package without restructuring it.

Submodules:
    config         — YAML config loading, path resolution, config hashing
    io             — TSV loading, ground-truth parsing, file helpers
    normalization  — conservative/aggressive string normalization (raw untouched)
    features       — pair-level name/address/country similarity features
    validation     — dataset integrity checks
    eda            — Stage 1 analyses producing eda/*.csv, figures, casebook, summary
    utils          — seeding, logging, run-history recording
"""

__version__ = "0.1.0-stage1"

__all__ = [
    "config",
    "io",
    "normalization",
    "features",
    "validation",
    "eda",
    "utils",
]
