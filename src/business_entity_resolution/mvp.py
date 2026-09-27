"""MVP end-to-end pipeline: retrieval -> features -> matcher -> submission.

Speed-first, reuse-first implementation for the solo 10-hour sprint:

- retrieval: word-prefilter (rarest-first, budgeted) + exact frozen char-TF-IDF
  rescore. Approximates the benchmarked exact top-k at ~100x the speed.
  Validated against the same anchors/metrics as Stage 2 (expect ~94% union).
- features: frozen ``compute_pair_features`` (30) + 5 small extras, computed
  over merge-joined frames with a threaded row loop (RapidFuzz releases GIL).
- matcher: HistGradientBoosting default (always installed); LightGBM/CatBoost
  opt-in via --model when importable. No install fights.
- decisions: per-entity threshold -> ranked list or abstain; macro-F0.5 sweep.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig
from .features import FEATURE_COLUMNS, compute_pair_features, token_jaccard
from .normalization import tokenize

logger = logging.getLogger("mvp")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    from rapidfuzz.fuzz import WRatio as _rf_wr, partial_ratio as _rf_pr

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

EXTRA_COLUMNS = [
    "name_partial_ratio", "name_token_jaccard", "addr_partial_ratio",
    "addr_wratio", "is_s3",
]
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + EXTRA_COLUMNS


def _partial(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(_rf_pr(a, b)) / 100.0
    return 0.0


def _wr(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(_rf_wr(a, b)) / 100.0
    return 0.0


# ---------------------------------------------------------------------------
# two-stage retrieval: word prefilter + exact char-TF-IDF rescore
# ---------------------------------------------------------------------------

class TwoStageRetrieval:
    """Fast approximate top-k: rarest-first word prefilter, exact rescore.

    Per query: take query word tokens rarest-first, union their postings until
    ``prefilter_min_union`` docs or ``prefilter_max_postings`` postings scanned,
    then rank that subset by exact frozen char-TF-IDF cosine and keep top-k.
    """

    def __init__(self) -> None:
        self.word_vec: Any = None
        self.word_csc: Any = None  # V x N (CSC for fast column/posting slices)
        self.char_vec: Any = None
        self.char_csr: Any = None  # N x V float32
        self.pool_ids: List[str] = []
        self.token_df: Optional[np.ndarray] = None

    def fit(
        self, pool_texts: List[str], pool_ids: List[str], cfg: AppConfig,
        max_word_features: int = 2000000,
    ) -> "TwoStageRetrieval":
        t0 = time.perf_counter()
        self.pool_ids = [str(i) for i in pool_ids]
        self.word_vec = CountVectorizer(
            analyzer="word", token_pattern=r"(?u)\S+", lowercase=False,
            max_features=max_word_features,
        )
        w = self.word_vec.fit_transform(pool_texts)
        self.word_csc = w.tocsc()
        self.word_csc.sort_indices()
        indptr = self.word_csc.indptr
        self.token_df = np.diff(indptr).astype(np.int64)
        # frozen char TF-IDF params (same as Stage 1/2 blockers)
        from .eda import build_tfidf_index

        vec, mat = build_tfidf_index(pool_texts, cfg)
        if vec is None or mat is None:
            raise RuntimeError("char TF-IDF fit failed.")
        self.char_vec = vec
        self.char_csr = mat.astype(np.float32).tocsr()
        self.char_csr.sort_indices()
        logger.info("TwoStage fit: pool=%d word_vocab=%d char_nnz=%d (%.0fs)",
                    len(self.pool_ids), len(self.word_vec.vocabulary_),
                    self.char_csr.nnz, time.perf_counter() - t0)
        return self

    def _prefilter_rows(self, query_text: str, min_union: int,
                        max_postings: int) -> np.ndarray:
        toks = [t for t in tokenize(query_text or "")]
        if not toks:
            return np.zeros(0, dtype=np.int64)
        vocab = self.word_vec.vocabulary_
        cols = sorted({vocab[t] for t in toks if t in vocab},
                      key=lambda c: self.token_df[c])
        if not cols:
            return np.zeros(0, dtype=np.int64)
        indptr = self.word_csc.indptr
        indices = self.word_csc.indices
        parts: List[np.ndarray] = []
        scanned = 0
        union_est = 0
        for c in cols:
            s, e = int(indptr[c]), int(indptr[c + 1])
            parts.append(indices[s:e])
            scanned += e - s
            union_est += e - s
            if union_est >= min_union or scanned >= max_postings:
                break
        if not parts:
            return np.zeros(0, dtype=np.int64)
        concat = np.concatenate(parts)
        if len(concat) == 0:
            return concat
        # top docs by token-hit count
        uniq, counts = np.unique(concat, return_counts=True)
        if len(uniq) > min_union:
            top = np.argpartition(-counts, min_union - 1)[:min_union]
            return uniq[top]
        return uniq

    def query(
        self, query_texts: List[str], query_ids: List[str], k: int,
        prefilter_min_union: int = 3000, prefilter_max_postings: int = 300000,
        chunk: int = 2000, n_threads: int = 1,
    ) -> pd.DataFrame:
        """Top-k pairs (PAIRS ONLY). Threaded over query spans; order-stable.

        The char n-gram query matrix is transformed PER SPAN inside workers
        (never the full query set upfront), so peak RAM stays flat at any
        test scale; values are identical either way.
        """
        n = len(query_texts)
        t0 = time.perf_counter()
        spans = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]

        def _work(se: Tuple[int, int]) -> List[Tuple[str, str]]:
            s, e = se
            qmat = self.char_vec.transform(
                query_texts[s:e]).astype(np.float32).tocsr()
            out: List[Tuple[str, str]] = []
            for li, qi in enumerate(range(s, e)):
                cand = self._prefilter_rows(query_texts[qi], prefilter_min_union,
                                            prefilter_max_postings)
                if len(cand) == 0:
                    continue
                sims = (qmat[li] @ self.char_csr[cand].T).toarray().ravel()
                take = min(k, len(cand))
                part = np.argpartition(-sims, take - 1)[:take]
                for j in cand[part]:
                    out.append((str(query_ids[qi]), self.pool_ids[int(j)]))
            return out

        if n_threads and n_threads > 1 and len(spans) > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                parts = list(ex.map(_work, spans))
            rows = [r for part in parts for r in part]
        else:
            rows = []
            for se in spans:
                rows.extend(_work(se))
                logger.info("  retrieval %d/%d queries (%.0fs)", se[1], n,
                            time.perf_counter() - t0)
        logger.info("TwoStage query: nq=%d k=%d pairs=%d (%d threads, %.0fs)",
                    n, k, len(rows), n_threads, time.perf_counter() - t0)
        self.last_query_seconds = time.perf_counter() - t0
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------------------------
# fast featurizer (frozen per-pair math, merge join, threaded rows)
# ---------------------------------------------------------------------------

def _featurize_tuple_block(
    block: List[Tuple[str, ...]],
) -> np.ndarray:
    """Top-level worker (spawn-picklable): tuples -> float32 feature matrix."""
    out = np.zeros((len(block), len(ALL_FEATURE_COLUMNS)), dtype=np.float32)
    for i, t in enumerate(block):
        (a_n, a_a, a_c, b_n, b_a, b_c,
         a_nn, a_an, b_nn, b_an, cid) = t
        f = compute_pair_features(a_n, a_a, a_c, b_n, b_a, b_c,
                                  a_nn, a_an, b_nn, b_an)
        row = [float(f[c]) for c in FEATURE_COLUMNS]
        row += [_partial(a_nn, b_nn),
                float(token_jaccard(a_nn, b_nn)),
                _partial(a_an, b_an), _wr(a_an, b_an),
                float(str(cid).startswith("S3-"))]
        out[i] = row
    return out


def featurize_pairs(
    pairs: pd.DataFrame, s1_df: pd.DataFrame, pool_df: pd.DataFrame,
    cols: Dict[str, str], n_threads: int = 8, chunk_rows: int = 50000,
    executor: Any = None, proc_task_rows: int = 25000,
) -> pd.DataFrame:
    """Attach ALL_FEATURE_COLUMNS to pairs. Returns pairs + features (+ids).

    ``executor`` (a process pool) switches the row loop to multiprocess —
    same numbers, no GIL. Tasks are plain string tuples (small pickles);
    workers never see the pool frames.
    """
    c_id, c_name, c_addr, c_cty = (cols["entity_id"], cols["business_name"],
                                  cols["business_address"], cols["country"])
    keep = [c_id, c_name, c_addr, c_cty, "name_norm", "address_norm"]
    left = pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    left["source1_entity_id"] = left["source1_entity_id"].astype(str)
    left["candidate_entity_id"] = left["candidate_entity_id"].astype(str)
    m = left.merge(s1_df[keep].rename(
        columns={c_id: "source1_entity_id", c_name: "a_name", c_addr: "a_addr",
                 c_cty: "a_cty", "name_norm": "a_nn", "address_norm": "a_an"}),
        on="source1_entity_id", how="left")
    m = m.merge(pool_df[keep].rename(
        columns={c_id: "candidate_entity_id", c_name: "b_name", c_addr: "b_addr",
                 c_cty: "b_cty", "name_norm": "b_nn", "address_norm": "b_an"}),
        on="candidate_entity_id", how="left")
    m = m.reset_index(drop=True)
    n = len(m)
    cols_out = ALL_FEATURE_COLUMNS
    X = np.zeros((n, len(cols_out)), dtype=np.float32)
    a_n = m["a_name"].astype(str).tolist()
    a_a = m["a_addr"].astype(str).tolist()
    a_c = m["a_cty"].astype(str).tolist()
    b_n = m["b_name"].astype(str).tolist()
    b_a = m["b_addr"].astype(str).tolist()
    b_c = m["b_cty"].astype(str).tolist()
    a_nn = m["a_nn"].astype(str).tolist()
    a_an = m["a_an"].astype(str).tolist()
    b_nn = m["b_nn"].astype(str).tolist()
    b_an = m["b_an"].astype(str).tolist()
    cand_ids = m["candidate_entity_id"].tolist()

    def _work(s: int, e: int) -> None:
        for i in range(s, e):
            f = compute_pair_features(a_n[i], a_a[i], a_c[i], b_n[i], b_a[i],
                                      b_c[i], a_nn[i], a_an[i], b_nn[i], b_an[i])
            row = [float(f[c]) for c in FEATURE_COLUMNS]
            row += [_partial(a_nn[i], b_nn[i]),
                    float(token_jaccard(a_nn[i], b_nn[i])),
                    _partial(a_an[i], b_an[i]), _wr(a_an[i], b_an[i]),
                    float(str(cand_ids[i]).startswith("S3-"))]
            X[i] = row

    spans = [(s, min(n, s + chunk_rows)) for s in range(0, n, chunk_rows)]
    t0 = time.perf_counter()
    if executor is not None and n >= proc_task_rows:
        tuples = list(zip(a_n, a_a, a_c, b_n, b_a, b_c, a_nn, a_an, b_nn,
                          b_an, cand_ids))
        tasks = [tuples[i:i + proc_task_rows]
                 for i in range(0, n, proc_task_rows)]
        X = np.vstack(list(executor.map(_featurize_tuple_block, tasks)))
        logger.info("featurized %d pairs (%d procs, %.0fs)", n,
                    getattr(executor, "_max_workers", "?"),
                    time.perf_counter() - t0)
    elif n_threads and n_threads > 1 and len(spans) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda se: _work(*se), spans))
        logger.info("featurized %d pairs (%d threads, %.0fs)", n, n_threads,
                    time.perf_counter() - t0)
    else:
        for s, e in spans:
            _work(s, e)
        logger.info("featurized %d pairs (serial, %.0fs)", n,
                    time.perf_counter() - t0)
    feat = pd.DataFrame(X, columns=cols_out)
    out = pd.concat([m[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True),
                     feat], axis=1)
    return out


# ---------------------------------------------------------------------------
# matcher
# ---------------------------------------------------------------------------

def build_model(name: str = "hgb", seed: int = 42) -> Any:
    name = (name or "hgb").lower()
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=50,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=600, learning_rate=0.06, depth=6,
                                  random_seed=seed, verbose=False,
                                  allow_writing_files=False)
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             tree_method="hist", eval_metric="logloss")
    from sklearn.ensemble import HistGradientBoostingClassifier

    try:
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=seed)
    except TypeError:  # older sklearn without class_weight
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=seed)


# ---------------------------------------------------------------------------
# entity-level F0.5 + thresholding + submission
# ---------------------------------------------------------------------------

def f05_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    if prec == 0 and rec == 0:
        return 0.0, prec, rec
    beta2 = 0.25
    f = (1 + beta2) * prec * rec / (beta2 * prec + rec)
    return f, prec, rec


def entity_scores(
    scored: pd.DataFrame, truth: Dict[str, List[str]], threshold: float,
) -> pd.DataFrame:
    """Per-entity P/R/F0.5 at a threshold. scored: [s1, cand, score]."""
    s1c, cc = scored.columns[0], scored.columns[1]
    pred = scored.loc[scored["score"] >= threshold].copy()
    pred_lists: Dict[str, List[str]] = {}
    if not pred.empty:
        for s1, g in pred.groupby(s1c):
            seen, ordered = set(), []
            for c in g.sort_values("score", ascending=False)[cc].astype(str):
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            pred_lists[str(s1)] = ordered
    rows = []
    for s1 in sorted(set(list(truth.keys())) | set(pred_lists.keys())):
        tset = set(truth.get(s1, []))
        pset = set(pred_lists.get(s1, []))
        tp = len(tset & pset)
        fp = len(pset - tset)
        fn = len(tset - pset)
        f, p, r = f05_from_counts(tp, fp, fn)
        rows.append({"source1_entity_id": s1, "n_truth": len(tset),
                     "n_pred": len(pset), "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f05": f,
                     "is_singleton_truth": int(len(tset) == 0)})
    return pd.DataFrame(rows)


def summarize_entities(ent: pd.DataFrame) -> Dict[str, float]:
    return {
        "macro_f05": float(ent["f05"].mean()) if len(ent) else 0.0,
        "macro_precision": float(ent["precision"].mean()) if len(ent) else 0.0,
        "macro_recall": float(ent["recall"].mean()) if len(ent) else 0.0,
        "mean_pred_per_s1": float(ent["n_pred"].mean()) if len(ent) else 0.0,
        "frac_pred_empty": float((ent["n_pred"] == 0).mean()) if len(ent) else 0.0,
        "n_entities": int(len(ent)),
    }


def threshold_sweep(
    scored: pd.DataFrame, truth: Dict[str, List[str]],
    coarse: Optional[List[float]] = None,
) -> pd.DataFrame:
    coarse = coarse or [round(x, 2) for x in
                        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]
    rows = []
    for t in coarse:
        ent = entity_scores(scored, truth, t)
        s = summarize_entities(ent)
        rows.append({"threshold": t, **s})
    df = pd.DataFrame(rows)
    best = df.loc[df["macro_f05"].idxmax(), "threshold"]
    fine = sorted({round(float(best) + d, 2)
                   for d in (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04)
                   if 0.01 < float(best) + d < 0.99})
    for t in fine:
        ent = entity_scores(scored, truth, t)
        s = summarize_entities(ent)
        rows.append({"threshold": t, **s})
    out = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return out


def predictions_to_lists(
    scored: pd.DataFrame, all_s1: List[str], threshold: float,
) -> pd.DataFrame:
    """Every S1 -> exactly one row: (source1_entity_id, matched_entity_ids).

    matched_entity_ids is comma-separated (GT format) or empty for abstain.
    """
    s1c, cc = scored.columns[0], scored.columns[1]
    pred = scored.loc[scored["score"] >= threshold]
    lists: Dict[str, str] = {}
    if not pred.empty:
        for s1, g in pred.groupby(s1c):
            seen, ordered = set(), []
            for c in g.sort_values("score", ascending=False)[cc].astype(str):
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            lists[str(s1)] = ",".join(ordered)
    rows = [{"source1_entity_id": str(a), "matched_entity_ids": lists.get(str(a), "")}
            for a in all_s1]
    return pd.DataFrame(rows)


def validate_submission(
    matching: pd.DataFrame, candidates: pd.DataFrame, test_s1_ids: List[str],
    pool_ids: set, cand_path=None, cand_chunksize: int = 2000000,
) -> List[str]:
    """Return a list of problems (empty = valid).

    ``cand_path`` streams the candidate TSV in chunks instead of holding it
    in RAM (same pool-membership check); ``candidates`` may then be None.
    """
    problems: List[str] = []
    want = [str(a) for a in test_s1_ids]
    if matching["source1_entity_id"].astype(str).tolist() != want:
        problems.append("matching_results rows != test S1 list (order/coverage).")
    if matching["source1_entity_id"].duplicated().any():
        problems.append("duplicate S1 rows in matching_results.")
    bad = 0
    for cell in matching["matched_entity_ids"].astype(str).tolist():
        for frag in [f.strip() for f in cell.split(",") if f.strip()]:
            if frag not in pool_ids:
                bad += 1
    if bad:
        problems.append(f"{bad} matched ids not in the test pool.")
    if cand_path is not None:
        cands_bad = 0
        for ch in pd.read_csv(cand_path, sep="\t", dtype=str,
                              chunksize=cand_chunksize,
                              usecols=["candidate_entity_id"]):
            cands_bad += int((~ch["candidate_entity_id"].astype(str).isin(
                pool_ids)).sum())
    else:
        if candidates["source1_entity_id"].duplicated().any():
            pass  # candidates are long-format; dup S1 expected
        cands_bad = (~candidates["candidate_entity_id"].astype(str).isin(
            pool_ids)).sum()
    if cands_bad:
        problems.append(f"{cands_bad} candidate ids not in the test pool.")
    return problems
