"""Stage 2 blocking benchmark harness — candidate RETRIEVAL only.

Scope (per STAGE1_FREEZE.md §7): reproduce the frozen Stage-1 baseline, then
benchmark scalable candidate generators independently and in combination on the
LARGE haystack. This module must never:

- modify normalization v1 (imports ``normalize_basic``/``tokenize`` read-only),
- redefine frozen blockers (imports the frozen implementations),
- build pair features, train a matcher, or emit entity-level decisions,
- turn retrieval/ANN similarity scores into match or ranking scores
  (artifacts store PAIRS only — no scores, no ranks).

All sampling is deterministic (``numpy.RandomState`` with recorded seeds).
"""
from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig
from .io import anchor_truth_to_basic, parse_anchor_truth
from .normalization import normalize_basic, tokenize

logger = logging.getLogger(__name__)

# Frozen implementations — imported, never redefined. The private-helper import
# is deliberate: exact reproduction beats interface purity for a frozen baseline.
from .eda import (  # noqa: E402  (documented intentional import)
    _exact_join_blocker,
    _inverted_index_blocker,
    build_tfidf_index,
    run_blocking_eda,
)

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as _sk_normalize

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    from scipy import sparse as _sp

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# anchors + normalization
# ---------------------------------------------------------------------------

def select_anchors(
    s1_df: pd.DataFrame,
    gt_df: Optional[pd.DataFrame],
    cols: Dict[str, str],
    n_anchors: int,
    seed: int,
    matched_only: bool = True,
    stratify_country: bool = True,
    min_per_country: int = 300,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Deterministically sample anchor S1 ids + parse their GT truth.

    When ``stratify_country`` is set, anchors are drawn proportionally to the
    eligible-anchor country mix (with ``min_per_country`` floor per country).
    ``matched_only`` restricts to anchors with >=1 GT match (recall needs
    positives); every anchor still counts toward per-S1 burden.
    """
    c_id, c_country = cols["entity_id"], cols["country"]
    rng = np.random.RandomState(seed)
    ids = s1_df[c_id].astype(str).tolist()
    countries = (
        s1_df[c_country].astype(str).tolist() if c_country in s1_df.columns
        else ["UNKNOWN"] * len(ids)
    )
    truth = parse_anchor_truth(gt_df, ids, cols)
    eligible = [i for i in ids if (truth.get(i) or (not matched_only))]
    if not eligible:
        return [], {}
    eligible_set = set(eligible)
    by_country: Dict[str, List[str]] = {}
    for i, c in zip(ids, countries):
        if i in eligible_set:
            by_country.setdefault(str(c), []).append(i)
    for c in by_country:
        by_country[c] = sorted(by_country[c])

    picks: List[str] = []
    if stratify_country and len(by_country) > 1:
        total = sum(len(v) for v in by_country.values())
        quotas = {
            c: max(min_per_country, int(round(n_anchors * len(v) / total)))
            for c, v in by_country.items()
        }
        # scale quotas down proportionally if they overshoot
        over = sum(quotas.values()) - n_anchors
        if over > 0:
            for c in sorted(quotas, key=lambda k: -quotas[k]):
                cut = min(over, quotas[c] - min_per_country)
                quotas[c] -= cut
                over -= cut
                if over <= 0:
                    break
        for c in sorted(by_country):
            pool = by_country[c]
            k = min(quotas.get(c, 0), len(pool))
            picks.extend(rng.choice(sorted(pool), size=k, replace=False).tolist())
        # top up / trim to exactly n_anchors
        if len(picks) < n_anchors:
            picked = set(picks)
            rest = [i for i in eligible if i not in picked]
            k = min(n_anchors - len(picks), len(rest))
            picks.extend(rng.choice(sorted(rest), size=k, replace=False).tolist())
        picks = picks[:n_anchors]
    else:
        k = min(n_anchors, len(eligible))
        picks = rng.choice(sorted(eligible), size=k, replace=False).tolist()
    anchor_truth = {a: truth.get(a, []) for a in picks}
    return picks, anchor_truth


def normalize_frame(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    """Add frozen ``name_norm``/``address_norm`` columns (normalization v1)."""
    out = df.copy()
    out["name_norm"] = out[cols["business_name"]].map(normalize_basic)
    out["address_norm"] = out[cols["business_address"]].map(normalize_basic)
    return out


def build_pool(
    s2_df: Optional[pd.DataFrame],
    s3_df: Optional[pd.DataFrame],
    cols: Dict[str, str],
    sample_n: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Concatenate S2+S3 into one haystack pool (deterministic id order).

    ``sample_n`` takes a deterministic uniform subsample WITHOUT forcing truth
    (unlike Stage 1 EDA pools); callers must report the pool-hit-rate ceiling.
    """
    c_id = cols["entity_id"]
    frames = [d for d in (s2_df, s3_df) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.sort_values(c_id).reset_index(drop=True)
    if sample_n is not None and len(pool) > sample_n:
        rng = np.random.RandomState(seed)
        keep = np.sort(
            rng.choice(np.arange(len(pool)), size=sample_n, replace=False)
        )
        pool = pool.iloc[keep].reset_index(drop=True)
    return pool


# ---------------------------------------------------------------------------
# exact sparse top-k at scale (frozen TF-IDF vectorizer + chunked query)
# ---------------------------------------------------------------------------

@dataclass
class SparseTopK:
    """Exact cosine top-k over a CSR pool in bounded-memory chunks."""

    vectorizer: Any = None
    pool_matrix: Any = None  # CSR float32, L2-normalized rows
    pool_ids: List[str] = field(default_factory=list)
    fit_seconds: float = 0.0

    def fit(
        self, pool_texts: List[str], pool_ids: List[str], cfg: AppConfig
    ) -> "SparseTopK":
        t0 = time.perf_counter()
        vec, mat = build_tfidf_index(pool_texts, cfg)  # frozen vectorizer
        if vec is None or mat is None:
            raise RuntimeError("TF-IDF fit failed (empty vocabulary?).")
        self.vectorizer = vec
        self.pool_matrix = mat.astype(np.float32).tocsr()
        self.pool_matrix.sort_indices()
        self.pool_ids = [str(i) for i in pool_ids]
        self.fit_seconds = time.perf_counter() - t0
        logger.info(
            "SparseTopK fit: pool=%d vocab=%d nnz=%d (%.1fs)",
            len(self.pool_ids), len(vec.vocabulary_),
            self.pool_matrix.nnz, self.fit_seconds,
        )
        return self

    def query(
        self,
        query_texts: List[str],
        query_ids: List[str],
        k: int,
        query_chunk: int = 256,
        doc_chunk: int = 250000,
    ) -> pd.DataFrame:
        """Exact top-k pairs; artifact stores PAIRS ONLY (no scores)."""
        if self.vectorizer is None or self.pool_matrix is None:
            raise RuntimeError("SparseTopK.query before fit.")
        t0 = time.perf_counter()
        qmat = self.vectorizer.transform(query_texts).astype(np.float32).tocsr()
        qmat.sort_indices()
        nq, nd = qmat.shape[0], self.pool_matrix.shape[0]
        k = min(k, nd)
        best_idx = np.full((nq, k), -1, dtype=np.int64)
        best_val = np.full((nq, k), -np.inf, dtype=np.float32)
        p_csr = self.pool_matrix
        for qs in range(0, nq, query_chunk):
            qe = min(nq, qs + query_chunk)
            qb = qmat[qs:qe]
            for ds in range(0, nd, doc_chunk):
                de = min(nd, ds + doc_chunk)
                block = (qb @ p_csr[ds:de].T).toarray()  # (qc x dc) float32
                if ds == 0:
                    take = min(k, block.shape[1])
                    part = np.argpartition(-block, take - 1, axis=1)[:, :take]
                    rows = np.arange(qe - qs)[:, None]
                    best_idx[qs:qe, :take] = part + ds
                    best_val[qs:qe, :take] = block[rows, part]
                else:
                    rows = np.arange(qe - qs)[:, None]
                    cand_idx = np.concatenate(
                        [best_idx[qs:qe], np.tile(np.arange(ds, de, dtype=np.int64), (qe - qs, 1))],
                        axis=1,
                    )
                    cand_val = np.concatenate([best_val[qs:qe], block], axis=1)
                    take = min(k, cand_val.shape[1])
                    part = np.argpartition(-cand_val, take - 1, axis=1)[:, :take]
                    best_idx[qs:qe, :] = cand_idx[rows, part]
                    best_val[qs:qe, :] = cand_val[rows, part]
                del block
            gc.collect()
        rows: List[Tuple[str, str]] = []
        for qi, qid in enumerate(query_ids):
            for j in best_idx[qi]:
                if int(j) >= 0:
                    rows.append((str(qid), self.pool_ids[int(j)]))
        logger.info(
            "SparseTopK query: nq=%d k=%d pairs=%d (%.1fs)",
            nq, k, len(rows), time.perf_counter() - t0,
        )
        self.last_query_seconds = time.perf_counter() - t0
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------------------------
# IVF ANN (SVD-dense + MiniBatchKMeans + exact in-cluster rescan)
# ---------------------------------------------------------------------------

@dataclass
class IVFIndex:
    """Approximate top-k: coarse quantizer routes, exact cosine decides.

    Retrieval mechanism only — similarities never leave this class as scores.
    """

    svd: Any = None
    kmeans: Any = None
    vectorizer: Any = None
    pool_ids: List[str] = field(default_factory=list)
    dense: Any = None  # memmap (n_docs x dim) float32, L2-normalized
    assign: Any = None  # memmap (n_docs,) int32 cluster ids
    members: Any = None  # npz: offsets + sorted row indices per cluster
    work_dir: Optional[Path] = None
    fit_seconds: float = 0.0
    last_query_seconds: float = 0.0
    last_scan_p50: float = 0.0

    def fit(
        self,
        pool_texts: List[str],
        pool_ids: List[str],
        cfg: AppConfig,
        n_components: int = 128,
        sample_n: int = 200000,
        n_clusters: int = 1024,
        minibatch_size: int = 10000,
        seed: int = 42,
        work_dir: Optional[Path] = None,
        dense_chunk: int = 50000,
    ) -> "IVFIndex":
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for the IVF index.")
        t0 = time.perf_counter()
        self.work_dir = Path(work_dir) if work_dir else Path("output/stage2/cache")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        vec, _ = build_tfidf_index(pool_texts[: max(1000, min(len(pool_texts), 5000))], cfg)
        # NOTE: vectorizer above is only a probe that vocab exists; the real
        # transform below reuses the SAME frozen params via a fresh fit on the
        # SVD sample for memory-bounded training. Vocabulary is recorded.
        rng = np.random.RandomState(seed)
        ns = min(sample_n, len(pool_texts))
        sidx = np.sort(rng.choice(np.arange(len(pool_texts)), size=ns, replace=False))
        sample_texts = [pool_texts[i] for i in sidx]
        vec, smat = build_tfidf_index(sample_texts, cfg)  # frozen params
        if vec is None or smat is None:
            raise RuntimeError("TF-IDF fit failed on the IVF training sample.")
        self.vectorizer = vec
        n_components = int(max(1, min(n_components, min(smat.shape) - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        dense_sample = svd.fit_transform(smat)
        self.svd = svd
        km = MiniBatchKMeans(
            n_clusters=min(n_clusters, len(dense_sample)),
            random_state=seed,
            batch_size=min(minibatch_size, len(dense_sample)),
            n_init=3,
        )
        km.fit(_sk_normalize(dense_sample.astype(np.float32)))
        self.kmeans = km
        del smat, dense_sample
        gc.collect()

        # transform the full pool in chunks -> memmap
        n = len(pool_texts)
        dim = int(n_components)
        dense_path = self.work_dir / "ivf_dense.dat"
        assign_path = self.work_dir / "ivf_assign.dat"
        for p in (dense_path, assign_path):
            if p.exists():
                p.unlink()
        self.dense = np.memmap(
            str(dense_path), dtype=np.float32, mode="w+", shape=(n, dim)
        )
        self.assign = np.memmap(
            str(assign_path), dtype=np.int32, mode="w+", shape=(n,)
        )
        centroids = _sk_normalize(km.cluster_centers_.astype(np.float32))
        for s in range(0, n, dense_chunk):
            e = min(n, s + dense_chunk)
            chunk = vec.transform(pool_texts[s:e])
            dd = _sk_normalize(svd.transform(chunk).astype(np.float32))
            self.dense[s:e] = dd
            self.assign[s:e] = np.argmax(dd @ centroids.T, axis=1).astype(np.int32)
            del chunk, dd
        self.dense.flush()
        self.assign.flush()
        del centroids
        gc.collect()
        # cluster -> members via argsort
        order = np.argsort(np.asarray(self.assign), kind="stable")
        counts = np.bincount(
            np.asarray(self.assign), minlength=int(km.n_clusters)
        ).astype(np.int64)
        offsets = np.zeros(int(km.n_clusters) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        np.savez_compressed(
            str(self.work_dir / "ivf_members.npz"), offsets=offsets, order=order
        )
        self.members = {"offsets": offsets, "order": order}
        self.pool_ids = [str(i) for i in pool_ids]
        self.fit_seconds = time.perf_counter() - t0
        logger.info(
            "IVF fit: pool=%d dim=%d clusters=%d (%.1fs)",
            n, dim, int(km.n_clusters), self.fit_seconds,
        )
        return self

    def query(
        self,
        query_texts: List[str],
        query_ids: List[str],
        k: int,
        nprobe: int = 8,
        max_scan_per_query: int = 100000,
    ) -> pd.DataFrame:
        if self.svd is None or self.kmeans is None or self.vectorizer is None:
            raise RuntimeError("IVFIndex.query before fit.")
        t0 = time.perf_counter()
        qmat = self.vectorizer.transform(query_texts)
        qd = _sk_normalize(self.svd.transform(qmat).astype(np.float32))
        centroids = _sk_normalize(self.kmeans.cluster_centers_.astype(np.float32))
        csc = qd @ centroids.T
        nprobe = max(1, min(int(nprobe), csc.shape[1]))
        topc = np.argpartition(-csc, nprobe - 1, axis=1)[:, :nprobe]
        offsets = self.members["offsets"]
        order = self.members["order"]
        dense = np.asarray(self.dense)
        rows: List[Tuple[str, str]] = []
        scans: List[int] = []
        for qi, qid in enumerate(query_ids):
            cand_rows: List[np.ndarray] = []
            for c in topc[qi]:
                s, e = int(offsets[int(c)]), int(offsets[int(c) + 1])
                if e > s:
                    cand_rows.append(order[s:e])
            if not cand_rows:
                scans.append(0)
                continue
            members = np.concatenate(cand_rows)
            if len(members) > max_scan_per_query:
                members = members[:max_scan_per_query]
            scans.append(len(members))
            sims = dense[members] @ qd[qi]
            take = min(k, len(members))
            part = np.argpartition(-sims, take - 1)[:take]
            for j in members[part]:
                rows.append((str(qid), self.pool_ids[int(j)]))
        self.last_query_seconds = time.perf_counter() - t0
        self.last_scan_p50 = float(np.median(scans)) if scans else 0.0
        logger.info(
            "IVF query: nq=%d k=%d nprobe=%d pairs=%d scan_p50=%.0f (%.1fs)",
            len(query_ids), k, nprobe, len(rows),
            self.last_scan_p50, self.last_query_seconds,
        )
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------------------------
# frozen exact blockers at full-pool scale (anchor-restricted, exact)
# ---------------------------------------------------------------------------

def exact_name_pairs(
    anchors_df: pd.DataFrame, pool_df: pd.DataFrame, cols: Dict[str, str]
) -> pd.DataFrame:
    """Frozen ``exact_norm_name`` evaluated for anchors against the full pool."""
    c_id = cols["entity_id"]
    need = {
        t
        for t in anchors_df["name_norm"].astype(str).tolist()
        if t and t != ""
    }
    if not need:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    sub = pool_df.loc[pool_df["name_norm"].isin(need), [c_id, "name_norm"]]
    if sub.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return _exact_join_blocker(anchors_df, sub, "name_norm", c_id)


def exact_rare_token_pairs(
    anchors_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cols: Dict[str, str],
    rare_max_df: int = 25,
    rare_min_len: int = 4,
    chunk: int = 500000,
) -> pd.DataFrame:
    """Frozen ``exact_rare_name_token`` (pool DF<=25, len>=4) at full scale.

    DF is counted over the FULL pool but only for tokens in the anchor
    universe, so memory stays bounded while the definition stays exact.
    """
    c_id = cols["entity_id"]
    anchor_toks = [
        {t for t in tokenize(t) if len(t) >= rare_min_len}
        for t in anchors_df["name_norm"].astype(str).tolist()
    ]
    universe = set().union(*anchor_toks) if anchor_toks else set()
    if not universe:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    pool_names = pool_df["name_norm"].astype(str).tolist()
    pool_ids = pool_df[c_id].astype(str).tolist()
    df_counter: Counter = Counter()
    for s in range(0, len(pool_names), chunk):
        for text in pool_names[s: s + chunk]:
            for t in set(tokenize(text)):
                if t in universe:
                    df_counter[t] += 1
    rare_vocab = {t for t, c in df_counter.items() if c <= rare_max_df}
    if not rare_vocab:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    s1_toks = [toks & rare_vocab for toks in anchor_toks]
    # pool postings restricted to anchor-relevant rare tokens, built in chunks
    postings: Dict[str, List[int]] = {}
    for s in range(0, len(pool_names), chunk):
        for off, text in enumerate(pool_names[s: s + chunk]):
            for t in set(tokenize(text)) & rare_vocab:
                postings.setdefault(t, []).append(s + off)
    s1_ids = anchors_df[c_id].astype(str).tolist()
    rows: List[Tuple[str, str]] = []
    for a, toks in zip(s1_ids, s1_toks):
        cands: set = set()
        for t in toks:
            for j in postings.get(t, ()):
                cands.add(pool_ids[j])
        for b in cands:
            rows.append((a, b))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------------------------
# evaluation (recall / burden / rescue identity / overlap)
# ---------------------------------------------------------------------------

def _cap_per_s1(
    pairs: pd.DataFrame, cap: int
) -> Tuple[pd.DataFrame, int]:
    if pairs.empty or cap <= 0:
        return pairs, 0
    counts = pairs.groupby("source1_entity_id").cumcount()
    keep = counts < cap
    return (
        pairs.loc[keep].reset_index(drop=True),
        int((~keep).sum()),
    )


def _source_pair_of(mid: str) -> str:
    m = str(mid)
    if m.startswith("S2-"):
        return "S1_S2"
    if m.startswith("S3-"):
        return "S1_S3"
    return "S1_OTHER"


def evaluate_blockers(
    pairs_by_blocker: Dict[str, pd.DataFrame],
    positives_df: pd.DataFrame,
    anchor_ids: List[str],
    anchor_country: Dict[str, str],
    pool_size: int,
    pool_hit_ceiling: Optional[float] = None,
    cap_per_s1: int = 5000,
    runtimes: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Score every blocker + build the per-positive rescue matrix.

    Returns (metrics_df, per_positive_df, overlap).
    ``per_positive_df`` carries one ``hit_<blocker>`` column per blocker plus
    ``n_hits``/``rescued_only_by`` so every positive is attributable.
    """
    runtimes = runtimes or {}
    anchor_ids = [str(a) for a in anchor_ids]
    anchor_set = set(anchor_ids)
    pos = positives_df.copy()
    pos["source1_entity_id"] = pos["source1_entity_id"].astype(str)
    pos["candidate_entity_id"] = pos["candidate_entity_id"].astype(str)
    if "source_pair" not in pos.columns:
        pos["source_pair"] = pos["candidate_entity_id"].map(_source_pair_of)
    pos["s1_country"] = pos["source1_entity_id"].map(
        lambda a: anchor_country.get(str(a), "UNKNOWN")
    )
    pos = pos.sort_values(
        ["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    per_pos = pos[
        ["source1_entity_id", "candidate_entity_id", "source_pair", "s1_country"]
    ].copy()
    metrics_rows: List[Dict[str, Any]] = []
    names = list(pairs_by_blocker.keys())
    for name in names:
        pairs, _ = _cap_per_s1(
            pairs_by_blocker[name] if pairs_by_blocker[name] is not None
            else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]),
            cap_per_s1,
        )
        if pairs.empty:
            hit = np.zeros(len(per_pos), dtype=bool)
            per_s1 = pd.Series(0, index=anchor_ids)
            n_pairs = 0
        else:
            pairs = pairs.copy()
            pairs["source1_entity_id"] = pairs["source1_entity_id"].astype(str)
            pairs["candidate_entity_id"] = pairs["candidate_entity_id"].astype(str)
            key_b = set(
                zip(
                    pairs["source1_entity_id"].tolist(),
                    pairs["candidate_entity_id"].tolist(),
                )
            )
            hit = np.array(
                [
                    (a, b) in key_b
                    for a, b in zip(
                        per_pos["source1_entity_id"].tolist(),
                        per_pos["candidate_entity_id"].tolist(),
                    )
                ],
                dtype=bool,
            )
            cnt = pairs.groupby("source1_entity_id").size()
            per_s1 = pd.Series(0, index=anchor_ids)
            per_s1.update(cnt)
            n_pairs = len(pairs)
        per_pos[f"hit_{name}"] = hit.astype(np.int64)
        vals = per_s1.to_numpy(dtype=np.int64)
        sub = per_pos[["source_pair", "s1_country"]].copy()
        sub["hit"] = hit
        row: Dict[str, Any] = {
            "blocker": name,
            "n_candidate_pairs": int(n_pairs),
            "n_anchors": len(anchor_ids),
            "n_positives": len(per_pos),
            "s1_s2_recall": float(sub.loc[sub.source_pair == "S1_S2", "hit"].mean())
            if (sub.source_pair == "S1_S2").any() else float("nan"),
            "s1_s3_recall": float(sub.loc[sub.source_pair == "S1_S3", "hit"].mean())
            if (sub.source_pair == "S1_S3").any() else float("nan"),
            "overall_recall": float(sub["hit"].mean()) if len(sub) else float("nan"),
            "avg_candidates_per_s1": float(vals.mean()) if len(vals) else 0.0,
            "p50_candidates_per_s1": float(np.median(vals)) if len(vals) else 0.0,
            "p95_candidates_per_s1": float(np.percentile(vals, 95)) if len(vals) else 0.0,
            "max_candidates_per_s1": int(vals.max()) if len(vals) else 0,
            "candidate_burden": float(n_pairs / (len(anchor_ids) * pool_size))
            if anchor_ids and pool_size else 0.0,
            "s1_with_zero_candidates": int((vals == 0).sum()),
        }
        for country in sorted(sub["s1_country"].unique().tolist()):
            m = sub["s1_country"] == country
            row[f"recall_country::{country}"] = float(sub.loc[m, "hit"].mean())
            row[f"n_pos_country::{country}"] = int(m.sum())
        if pool_hit_ceiling is not None:
            row["pool_hit_ceiling"] = float(pool_hit_ceiling)
        rt = runtimes.get(name, {})
        row["fit_seconds"] = float(rt.get("fit", float("nan")))
        row["query_seconds"] = float(rt.get("query", float("nan")))
        metrics_rows.append(row)

    hit_cols = [f"hit_{n}" for n in names]
    per_pos["n_hits"] = per_pos[hit_cols].sum(axis=1).astype(int) if hit_cols else 0
    single = per_pos[hit_cols].idxmax(axis=1).str.replace("^hit_", "", regex=True)
    per_pos["rescued_only_by"] = np.where(
        per_pos["n_hits"] == 0, "none",
        np.where(per_pos["n_hits"] == 1, single, "multiple"),
    )
    overlap = _overlap_stats(per_pos, names)
    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, per_pos, overlap


def _overlap_stats(per_pos: pd.DataFrame, names: List[str]) -> Dict[str, Any]:
    n = len(per_pos)
    hit = {
        nm: set(per_pos.loc[per_pos[f"hit_{nm}"] == 1].index.tolist())
        for nm in names
    }
    union: set = set().union(*hit.values()) if hit else set()
    inter: set = set.intersection(*hit.values()) if hit else set()
    unique_rescue = {}
    for nm in names:
        others: set = set().union(*(v for k, v in hit.items() if k != nm)) if len(hit) > 1 else set()
        unique_rescue[nm] = sorted(hit[nm] - others)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a}||{b}"] = {
                "n_both": len(hit[a] & hit[b]),
                "n_a_only": len(hit[a] - hit[b]),
                "n_b_only": len(hit[b] - hit[a]),
                "jaccard": (len(hit[a] & hit[b]) / len(hit[a] | hit[b]))
                if (hit[a] | hit[b]) else 0.0,
            }
    return {
        "n_positives": n,
        "n_union_covered": len(union),
        "union_coverage": (len(union) / n) if n else 0.0,
        "n_missed_by_all": n - len(union),
        "n_hit_by_all": len(inter),
        "n_hit_by_exactly_one": int((per_pos["n_hits"] == 1).sum()) if n else 0,
        "unique_rescue_counts": {k: len(v) for k, v in unique_rescue.items()},
        "unique_rescue_row_idx": unique_rescue,
        "pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# artifacts / history / experiment log
# ---------------------------------------------------------------------------

def write_experiment_artifacts(
    exp_dir: Path,
    metrics_df: pd.DataFrame,
    per_positive_df: pd.DataFrame,
    overlap: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    p_metrics = exp_dir / "metrics.csv"
    metrics_df.to_csv(p_metrics, index=False)
    paths.append(str(p_metrics))
    p_per = exp_dir / "per_positive.csv"
    per_positive_df.to_csv(p_per, index=False)
    paths.append(str(p_per))
    p_missed = exp_dir / "missed_positives.csv"
    missed = per_positive_df.loc[per_positive_df["n_hits"] == 0].drop(
        columns=[c for c in per_positive_df.columns if c.startswith("hit_")],
        errors="ignore",
    )
    missed.to_csv(p_missed, index=False)
    paths.append(str(p_missed))
    p_overlap = exp_dir / "overlap.json"
    with open(p_overlap, "w", encoding="utf-8") as fh:
        json.dump(overlap, fh, indent=2, default=str)
    paths.append(str(p_overlap))
    pw_rows = [
        {"pair": k, **v} for k, v in overlap.get("pairwise", {}).items()
    ]
    p_pw = exp_dir / "pairwise_overlap.csv"
    pd.DataFrame(pw_rows).to_csv(p_pw, index=False)
    paths.append(str(p_pw))
    if extra:
        p_extra = exp_dir / "runtime.json"
        with open(p_extra, "w", encoding="utf-8") as fh:
            json.dump(extra, fh, indent=2, default=str)
        paths.append(str(p_extra))
    return paths


def append_stage2_history(logs_dir: Path, record: Dict[str, Any]) -> str:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "stage2_history.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return str(path)


def log_experiment_row(
    project_root: Path,
    team_member: str,
    description: str,
    hypothesis: str,
    blocking_version: str,
    hyperparameters: str,
    candidate_recall: Optional[float],
    avg_candidates: Optional[float],
    max_candidates: Optional[int],
    notes: str,
    code_version: str,
    day: str = "2",
) -> str:
    """Append one row via the repo's logger (schema owned by log_experiment.py)."""
    script = Path(project_root) / "scripts" / "log_experiment.py"
    cmd = [
        sys.executable, str(script), "--yes",
        "--team-member", team_member,
        "--experiment-type", "new",
        "--description", description,
        "--hypothesis", hypothesis,
        "--day", str(day),
        "--code-version", code_version,
        "--data-version", "train-v1",
        "--normalization-version", "conservative-v1",
        "--blocking-version", blocking_version,
        "--feature-version", "NA-stage2",
        "--model-version", "NA-stage2",
        "--hyperparameters", hyperparameters,
        "--validation-split", "stage2-anchors",
        "--status", "done",
        "--notes", notes,
        "--public-submission", "no",
    ]
    if candidate_recall is not None:
        cmd += ["--candidate-recall", f"{candidate_recall:.6f}"]
    if avg_candidates is not None:
        cmd += ["--avg-candidates-per-s1", f"{avg_candidates:.3f}"]
    if max_candidates is not None:
        cmd += ["--max-candidates-per-s1", str(int(max_candidates))]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    if out.returncode != 0:
        raise RuntimeError(f"log_experiment.py failed: {out.stderr[-2000:]}")
    return (out.stdout or "").strip()[-500:]
