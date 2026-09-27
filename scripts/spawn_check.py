#!/usr/bin/env python3
"""30-second Windows spawn-safety check for the MVP process featurizer.

Full infer uses ProcessPoolExecutor (spawn on Windows): each child must
import business_entity_resolution.mvp, unpickle string-tuple tasks, run
_featurize_tuple_block, and return float32 rows. This script exercises that
exact path on fake tuples. Exit 0 + "SPAWN-OK" means the full run is safe
to launch with infer.processes > 0.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.mvp import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    _featurize_tuple_block,
)


def main() -> int:
    row = ("acme traders pvt ltd", "12 mg road bengaluru", "IN",
           "acme traders private limited", "12 m.g. road bangalore", "IN",
           "acme traders pvt ltd", "12 mg road bengaluru",
           "acme traders private limited", "12 m.g. road bangalore",
           "S2-000001")
    tasks = [[row] * 500 for _ in range(16)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(_featurize_tuple_block, tasks))
    n = sum(len(o) for o in outs)
    assert n == 8000, n
    assert all(o.shape[1] == len(ALL_FEATURE_COLUMNS) for o in outs)
    assert all(np.isfinite(np.asarray(o, dtype=np.float64)).all()
               for o in outs)
    print(f"SPAWN-OK: 16 tasks x 500 rows = {n} featurized "
          f"({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
