#!/usr/bin/env python3
"""MVP end-to-end pipeline: retrieval -> features -> matcher -> submission.

Modes:
  train : train anchors -> candidates -> features -> model -> valid F0.5 sweep
  infer : test S1 -> candidates -> features -> scores -> matching_results.tsv

Examples:
  python scripts/run_mvp.py --mode train --data-root <dataset>
  python scripts/run_mvp.py --mode infer --data-root <dataset> --threshold 0.42
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.blocking import (  # noqa: E402
    append_stage2_history,
    build_pool,
    evaluate_blockers,
    exact_name_pairs,
    exact_rare_token_pairs,
    log_experiment_row,
    normalize_frame,
    select_anchors,
)
from business_entity_resolution.config import find_project_root, load_config  # noqa: E402
from business_entity_resolution.io import (  # noqa: E402
    anchor_truth_to_basic,
    load_test_tables,
    load_training_tables,
    parse_anchor_truth,
)
from business_entity_resolution.mvp import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    TwoStageRetrieval,
    build_model,
    entity_scores,
    featurize_pairs,
    predictions_to_lists,
    summarize_entities,
    threshold_sweep,
    validate_submission,
)
from business_entity_resolution.utils import get_git_commit  # noqa: E402

logger = logging.getLogger("mvp-run")

FROZEN_CONFIG_HASH = "90d125f76e75"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MVP end-to-end pipeline.")
    ap.add_argument("--mode", required=True, choices=["train", "infer"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--mvp-config", default="config/mvp.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--n-train-anchors", type=int, default=None)
    ap.add_argument("--n-valid-anchors", type=int, default=None)
    ap.add_argument("--max-test-s1", type=int, default=None,
                    help="smoke-test: only first N test S1")
    ap.add_argument("--team-member", default="team")
    ap.add_argument("--day", default="2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args(argv)


def _retrieve_all(s1_df, pool_df, cols, cfg, rcfg, tag, topk=None,
                  n_threads=1):
    """Two-stage name+addr top-k plus frozen exact nets. Returns (pairs, info)."""
    topk = int(topk or rcfg.get("topk", 50))
    c_id = cols["entity_id"]
    qids = s1_df[c_id].astype(str).tolist()
    parts, info = {}, {}
    for field in ("name", "address"):
        idx = TwoStageRetrieval().fit(
            pool_df[f"{field}_norm"].astype(str).tolist(),
            pool_df[c_id].astype(str).tolist(), cfg,
            max_word_features=int(rcfg.get("max_word_features", 2000000)))
        q = idx.query(s1_df[f"{field}_norm"].astype(str).tolist(), qids, topk,
                      int(rcfg.get("prefilter_min_union", 3000)),
                      int(rcfg.get("prefilter_max_postings", 300000)),
                      n_threads=n_threads)
        parts[field] = q
        info[f"{field}_query_s"] = idx.last_query_seconds
        del idx
        gc.collect()
    if rcfg.get("include_exact_name", True):
        parts["exact_name"] = exact_name_pairs(s1_df, pool_df, cols)
    if rcfg.get("include_exact_rare", True):
        ret = cfg.eda.get("retrieval", {})
        parts["exact_rare"] = exact_rare_token_pairs(
            s1_df, pool_df, cols,
            rare_max_df=int(ret.get("rare_token_max_df", 25)),
            rare_min_len=int(ret.get("rare_token_min_len", 4)))
    pairs = pd.concat(list(parts.values()),
                      ignore_index=True).drop_duplicates().reset_index(drop=True)
    logger.info("%s: union=%d pairs over %d S1 (%.1f/S1)", tag, len(pairs),
                len(qids), len(pairs) / max(1, len(qids)))
    return pairs, info


def _history(logs_dir, record):
    p = Path(logs_dir) / "mvp_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    root = find_project_root(Path(__file__).resolve())
    cfg = load_config(data_root_override=args.data_root, project_root=root)
    if cfg.config_hash() != FROZEN_CONFIG_HASH:
        logger.error("config hash %s != frozen %s — refusing.",
                     cfg.config_hash(), FROZEN_CONFIG_HASH)
        return 2
    with open(root / args.mvp_config, "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh) or {}
    seed = int(args.seed) if args.seed is not None else int(mc.get("seed", 42))
    out_dir = Path(args.output_dir) if args.output_dir else root / str(
        mc.get("output_dir", "output/mvp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = cfg.columns
    c_id = cols["entity_id"]
    git_commit = get_git_commit(root)
    tcfg, rcfg, icfg = mc.get("train", {}), mc.get("retrieval", {}), mc.get("infer", {})

    def _log(desc, hyp, bver, fver, mver, hyper, thr, f05, prec, rec, notes):
        if args.no_log:
            return
        log_experiment_row(root, args.team_member, desc, hyp, bver, hyper,
                           None, None, None, notes, git_commit, day=args.day)
        # NOTE: local metrics ride in notes (repo log has no f05 column by default).

    if args.mode == "train":
        train = load_training_tables(cfg)
        s1_raw, s2_raw, s3_raw, gt_df = (train["train_s1"], train["train_s2"],
                                         train["train_s3"], train["train_gt"])
        n_tr = int(args.n_train_anchors or tcfg.get("n_train_anchors", 15000))
        n_va = int(args.n_valid_anchors or tcfg.get("n_valid_anchors", 2000))
        tr_ids, tr_truth = select_anchors(
            s1_raw, gt_df, cols, n_tr, seed,
            matched_only=bool(tcfg.get("matched_only_train", True)))
        va_ids, va_truth_all = select_anchors(
            s1_raw, gt_df, cols, n_va + len(tr_ids), seed + 999,
            matched_only=False)
        va_ids = [a for a in va_ids if a not in set(tr_ids)][:n_va]
        va_truth = {a: va_truth_all[a] for a in va_ids}
        if not tcfg.get("include_singletons_valid", True):
            keep = [a for a in va_ids if va_truth[a]]
            va_truth = {a: va_truth[a] for a in keep}
            va_ids = keep
        logger.info("train anchors=%d valid anchors=%d (%d singleton valid)",
                    len(tr_ids), len(va_ids),
                    sum(1 for a in va_ids if not va_truth[a]))
        pool = normalize_frame(build_pool(s2_raw, s3_raw, cols), cols)
        tr_df = normalize_frame(
            s1_raw.loc[s1_raw[c_id].astype(str).isin(set(tr_ids))].copy(), cols)
        va_df = normalize_frame(
            s1_raw.loc[s1_raw[c_id].astype(str).isin(set(va_ids))].copy(), cols)

        tr_pairs, _ = _retrieve_all(tr_df, pool, cols, cfg, rcfg, "train")
        va_pairs, _ = _retrieve_all(va_df, pool, cols, cfg, rcfg, "valid")
        # retrieval recall check on valid anchors (same yardstick as Stage 2)
        va_pos = anchor_truth_to_basic({a: t for a, t in va_truth.items() if t})
        va_cty = dict(zip(s1_raw[c_id].astype(str),
                          s1_raw[cols["country"]].astype(str)))
        met, _, _ = evaluate_blockers(
            {"mvp_union": va_pairs}, va_pos,
            va_df[c_id].astype(str).tolist(),
            {a: va_cty.get(a, "?") for a in va_ids}, len(pool))
        rec = float(met.loc[0, "overall_recall"])
        bur = float(met.loc[0, "avg_candidates_per_s1"])
        logger.info("RETRIEVAL CHECK: valid-anchor union recall=%.4f @ %.1f/S1 "
                    "(S2=%.4f S3=%.4f)", rec, bur,
                    float(met.loc[0, "s1_s2_recall"]),
                    float(met.loc[0, "s1_s3_recall"]))
        if rec < float(rcfg.get("warn_recall_below", 0.90)):
            logger.warning("Union recall %.4f below %.2f — prefilter may be "
                           "lossy; consider raising prefilter budgets.", rec,
                           float(rcfg.get("warn_recall_below", 0.90)))

        # labels: positives = anchor GT; negatives = candidates minus truth
        prime = {a: set(t) for a, t in tr_truth.items()}
        is_pos = np.array([str(b) in prime.get(str(a), set()) for a, b in zip(
            tr_pairs["source1_entity_id"].astype(str),
            tr_pairs["candidate_entity_id"].astype(str))])
        pos_pairs = tr_pairs.loc[is_pos].reset_index(drop=True)
        neg_pool = tr_pairs.loc[~is_pos].reset_index(drop=True)
        ratio = float(tcfg.get("neg_ratio", 3))
        n_neg = min(len(neg_pool), int(len(pos_pairs) * ratio))
        rng = np.random.RandomState(seed + 7)
        neg_pairs = neg_pool.iloc[np.sort(rng.choice(
            np.arange(len(neg_pool)), size=n_neg, replace=False))].reset_index(drop=True)
        logger.info("labeled: pos=%d neg=%d (ratio 1:%.1f)", len(pos_pairs),
                    len(neg_pairs), len(neg_pairs) / max(1, len(pos_pairs)))
        threads = int(icfg.get("threads", 8))
        fpos = featurize_pairs(pos_pairs, tr_df, pool, cols, threads)
        fneg = featurize_pairs(neg_pairs, tr_df, pool, cols, threads)
        fpos["label"], fneg["label"] = 1, 0
        train_df = pd.concat([fpos, fneg], ignore_index=True)
        train_df.to_pickle(out_dir / "train_pairs_featurized.pkl")
        X, y = train_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32), \
            train_df["label"].to_numpy()
        model_name = (args.model or tcfg.get("model", "hgb")).lower()
        try:
            from business_entity_resolution.mvp import build_model as _bm
            model = _bm(model_name, seed)
        except ImportError:
            logger.warning("%s not installed — falling back to hgb.", model_name)
            from business_entity_resolution.mvp import build_model as _bm
            model = _bm("hgb", seed)
            model_name = "hgb"
        t0 = time.perf_counter()
        model.fit(X, y)
        logger.info("model=%s trained on %d pairs in %.0fs", model_name, len(y),
                    time.perf_counter() - t0)
        import joblib

        joblib.dump(model, out_dir / "model.joblib")

        # validate: valid candidates -> features -> scores -> threshold sweep
        fva = featurize_pairs(va_pairs, va_df, pool, cols, threads)
        scores = model.predict_proba(
            fva[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
        scored = pd.DataFrame({"source1_entity_id": fva["source1_entity_id"],
                               "candidate_entity_id": fva["candidate_entity_id"],
                               "score": scores})
        sweep = threshold_sweep(scored, {a: list(t) for a, t in va_truth.items()})
        sweep.to_csv(out_dir / "valid_threshold_sweep.csv", index=False)
        best = sweep.loc[sweep["macro_f05"].idxmax()]
        logger.info("VALID SWEEP (n=%d entities):\n%s",
                    len(va_ids), sweep.to_string(index=False))
        logger.info("BEST threshold=%.2f macroF05=%.4f P=%.4f R=%.4f "
                    "pred/S1=%.2f empty=%.3f", best["threshold"],
                    best["macro_f05"], best["macro_precision"],
                    best["macro_recall"], best["mean_pred_per_s1"],
                    best["frac_pred_empty"])
        with open(out_dir / "threshold.json", "w", encoding="utf-8") as fh:
            json.dump({"threshold": float(best["threshold"]),
                       "macro_f05": float(best["macro_f05"]),
                       "macro_precision": float(best["macro_precision"]),
                       "macro_recall": float(best["macro_recall"])}, fh, indent=2)
        _history(cfg.logs_dir, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "train", "git_commit": git_commit, "seed": seed,
            "retrieval_recall": rec, "retrieval_burden": bur,
            "n_train_pairs": len(train_df), "model": model_name,
            "best_threshold": float(best["threshold"]),
            "macro_f05": float(best["macro_f05"])})
        _log("MVP train: twostage retrieval + 35 feats + " + model_name,
             "MVP pipeline reaches competitive local macro-F0.5.",
             "mvp-twostage-v1", "mvp-feats-v1", model_name,
             f"neg_ratio={ratio}", float(best["threshold"]),
             float(best["macro_f05"]), float(best["macro_precision"]),
             float(best["macro_recall"]),
             f"retrieval={rec:.4f}@{bur:.1f}; F05={best['macro_f05']:.4f} "
             f"P={best['macro_precision']:.4f} R={best['macro_recall']:.4f}")
        return 0

    # ------------------------------- infer -------------------------------
    test = load_test_tables(cfg)
    s1_test, s2_test, s3_test = (test["test_s1"], test["test_s2"], test["test_s3"])
    if s1_test is None or s1_test.empty:
        logger.error("No test_source1 — nothing to do.")
        return 2
    if args.max_test_s1:
        s1_test = s1_test.head(int(args.max_test_s1))
    all_s1 = s1_test[c_id].astype(str).tolist()
    pool = normalize_frame(build_pool(s2_test, s3_test, cols), cols)
    pool_ids = pool[c_id].astype(str).tolist()
    pool_set = set(pool_ids)
    prow_of = {p: j for j, p in enumerate(pool_ids)}
    s1_test_n = normalize_frame(s1_test.copy(), cols)
    import joblib

    model = joblib.load(out_dir / "model.joblib")
    thr = float(args.threshold) if args.threshold is not None else float(
        json.load(open(out_dir / "threshold.json"))["threshold"])
    logger.info("infer: test_s1=%d pool=%d threshold=%.3f", len(all_s1),
                len(pool), thr)
    tchunk = int(icfg.get("test_chunk", 50000))
    threads = int(icfg.get("threads", 8))
    scored_dir = out_dir / "scored_chunks"
    scored_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidate_pairs.tsv"
    if cand_path.exists():
        cand_path.unlink()
    first = True
    for cs in range(0, len(all_s1), tchunk):
        ce = min(len(all_s1), cs + tchunk)
        chunk_df = s1_test_n.iloc[cs:ce].reset_index(drop=True)
        logger.info("infer chunk %d-%d/%d", cs, ce, len(all_s1))
        pairs, _ = _retrieve_all(chunk_df, pool, cols, cfg, rcfg,
                                 f"infer[{cs}:{ce}]")
        pairs[["source1_entity_id", "candidate_entity_id"]].to_csv(
            cand_path, sep="\t", index=False, header=first, mode="a")
        first = False
        # featurize+predict in row chunks
        fch = int(icfg.get("feature_chunk_rows", 200000))
        s_ids, c_rows, sc = [], [], []
        for rs in range(0, len(pairs), fch):
            sub = pairs.iloc[rs:rs + fch].reset_index(drop=True)
            fz = featurize_pairs(sub, chunk_df, pool, cols, threads)
            pr = model.predict_proba(
                fz[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
            s_ids.append(fz["source1_entity_id"].to_numpy())
            c_rows.append(np.array([prow_of[c] for c in
                                    fz["candidate_entity_id"].astype(str)],
                                   dtype=np.int32))
            sc.append(pr.astype(np.float32))
            del fz, pr
            gc.collect()
        if s_ids:
            np.savez_compressed(
                scored_dir / f"scored_{cs:07d}_{ce:07d}.npz",
                s1=np.concatenate(s_ids),
                cand_row=np.concatenate(c_rows),
                score=np.concatenate(sc))
        del pairs
        gc.collect()
    np.save(scored_dir / "pool_ids.npy", np.array(pool_ids))
    # threshold -> matching_results (streaming over scored chunks)
    lists: Dict[str, List[Tuple[float, str]]] = {a: [] for a in all_s1}
    for npz in sorted(scored_dir.glob("scored_*.npz")):
        z = np.load(npz, allow_pickle=True)
        s1a, cra, sca = z["s1"], z["cand_row"], z["score"]
        for a, r, s in zip(s1a.tolist(), cra.tolist(), sca.tolist()):
            if s >= thr:
                lists[str(a)].append((float(s), pool_ids[int(r)]))
    rows = []
    for a in all_s1:
        seen, ordered = set(), []
        for _, c in sorted(lists[a], reverse=True):
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        rows.append({"source1_entity_id": a,
                     "matched_entity_ids": ",".join(ordered)})
    matching = pd.DataFrame(rows)
    matching.to_csv(out_dir / "matching_results.tsv", sep="\t", index=False)
    cands = pd.read_csv(cand_path, sep="\t", dtype=str)
    problems = validate_submission(matching, cands, all_s1, pool_set)
    n_pred = int((matching["matched_entity_ids"].astype(str) != "").sum())
    logger.info("SUBMISSION: rows=%d predicted-nonempty=%d empty=%d", len(matching),
                n_pred, len(matching) - n_pred)
    if problems:
        for p in problems:
            logger.error("SUBMISSION PROBLEM: %s", p)
        return 3
    logger.info("submission VALID: %s + %s", out_dir / "matching_results.tsv",
                cand_path)
    _history(cfg.logs_dir, {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "infer", "git_commit": git_commit, "threshold": thr,
        "n_test_s1": len(all_s1), "n_candidate_pairs": len(cands),
        "n_predicted_nonempty": n_pred})
    _log("MVP infer: submission at threshold %.3f" % thr,
         "First leaderboard submission.", "mvp-twostage-v1", "mvp-feats-v1",
         "see-train-row", "threshold=%.3f" % thr, thr, None, None, None,
         f"s1={len(all_s1)} pairs={len(cands)} nonempty={n_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
