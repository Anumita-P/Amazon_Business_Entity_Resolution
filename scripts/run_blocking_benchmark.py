#!/usr/bin/env python3
"""Stage 2 blocking benchmark harness — retrieval-only, no matcher, no ranking.

Reproduces the frozen Stage-1 baseline (EXP-001) and benchmarks scalable
candidate generators on the LARGE haystack (EXP-002..006). Outputs live under
``output/stage2/`` — separate from the frozen Stage 1 artifacts. History goes
to ``logs/stage2_history.jsonl``; one row per experiment is appended to
``logs/experiment_log.csv`` (unless --no-log).

Retrieval/ANN similarities are NEVER persisted as scores and NEVER used for
ranking — artifacts store candidate PAIRS only.

Examples:
    python scripts/run_blocking_benchmark.py --data-root <dataset> --experiments all
    python scripts/run_blocking_benchmark.py --data-root <dataset> --experiments EXP-001
    python scripts/run_blocking_benchmark.py --data-root <dataset> --haystack sample:1000000
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
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
    IVFIndex,
    SparseTopK,
    append_stage2_history,
    build_pool,
    evaluate_blockers,
    exact_name_pairs,
    exact_rare_token_pairs,
    log_experiment_row,
    normalize_frame,
    run_blocking_eda,
    select_anchors,
    write_experiment_artifacts,
)
from business_entity_resolution.config import find_project_root, load_config  # noqa: E402
from business_entity_resolution.io import (  # noqa: E402
    anchor_truth_to_basic,
    load_training_tables,
)
from business_entity_resolution.utils import get_git_commit  # noqa: E402

logger = logging.getLogger("stage2")

FROZEN_CONFIG_HASH = "90d125f76e75"  # STAGE1_FREEZE.md §1

ALL_EXPERIMENTS = ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-005", "EXP-006"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2 blocking benchmark harness.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--stage1-config", default="config/config.yaml")
    ap.add_argument("--stage2-config", default="config/stage2.yaml")
    ap.add_argument("--experiments", default="all",
                    help="comma list like EXP-001,EXP-002 or 'all'")
    ap.add_argument("--haystack", default=None,
                    help="override: 'full' or 'sample:N'")
    ap.add_argument("--n-anchors", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--team-member", default="team")
    ap.add_argument("--day", default="2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-log", action="store_true",
                    help="skip logs/experiment_log.csv appends")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read/write the normalized-pool cache")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    root = find_project_root(Path(__file__).resolve())
    cfg = load_config(config_path=args.stage1_config,
                      data_root_override=args.data_root, project_root=root)
    if cfg.config_hash() != FROZEN_CONFIG_HASH:
        logger.error("config hash %s != frozen %s — refusing to run: %s",
                     cfg.config_hash(), FROZEN_CONFIG_HASH, cfg.config_path)
        return 2
    logger.info("Stage-1 config verified frozen (%s).", FROZEN_CONFIG_HASH)

    with open(root / args.stage2_config, "r", encoding="utf-8") as fh:
        s2 = yaml.safe_load(fh) or {}
    seed = int(args.seed) if args.seed is not None else int(s2.get("seed", 42))
    out_dir = Path(args.output_dir) if args.output_dir else root / str(s2.get("output_dir", "output/stage2"))
    cache_dir = root / str(s2.get("cache_dir", "output/stage2/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)

    hay = str(args.haystack) if args.haystack else str(s2.get("haystack", {}).get("mode", "full"))
    sample_n = None
    if hay.startswith("sample"):
        sample_n = int(hay.split(":", 1)[1]) if ":" in hay else int(
            s2.get("haystack", {}).get("sample_n", 1000000))
        hay = "sample"
    a_cfg = s2.get("anchors", {})
    n_anchors = int(args.n_anchors) if args.n_anchors else int(a_cfg.get("n_anchors", 2000))

    wanted = ALL_EXPERIMENTS if args.experiments.strip().lower() == "all" else [
        e.strip().upper().replace("EXP", "EXP-") if e.strip().upper().startswith("EXP") and "-" not in e else e.strip().upper()
        for e in args.experiments.split(",") if e.strip()
    ]
    wanted = [f"EXP-{e}" if e.isdigit() else e for e in wanted]
    unknown = [e for e in wanted if e not in ALL_EXPERIMENTS]
    if unknown:
        logger.error("Unknown experiments: %s (choose from %s)", unknown, ALL_EXPERIMENTS)
        return 2

    logger.info("project_root : %s", root)
    logger.info("data_root    : %s", cfg.data_root)
    logger.info("experiments  : %s", wanted)
    logger.info("haystack     : %s%s", hay, f" (n={sample_n})" if sample_n else " (FULL train S2+S3)")
    logger.info("anchors      : n=%d seed=%d", n_anchors, seed)

    train = load_training_tables(cfg)
    s1_raw, s2_raw, s3_raw, gt_df = (train.get("train_s1"), train.get("train_s2"),
                                     train.get("train_s3"), train.get("train_gt"))
    if s1_raw is None or s1_raw.empty:
        logger.error("No train_source1 under %s — nothing to do.", cfg.data_root / "train")
        return 2
    cols = cfg.columns
    git_commit = get_git_commit(root)

    # ---- shared anchors (one draw per run => comparable experiments) ----
    t0 = time.perf_counter()
    anchor_ids, anchor_truth = select_anchors(
        s1_raw, gt_df, cols, n_anchors, seed,
        matched_only=bool(a_cfg.get("matched_only", True)),
        stratify_country=bool(a_cfg.get("stratify_country", True)),
        min_per_country=int(a_cfg.get("min_per_country", 300)),
    )
    if not anchor_ids:
        logger.error("Anchor selection returned nothing — check GT coverage.")
        return 2
    positives = anchor_truth_to_basic(anchor_truth)
    c_id, c_country = cols["entity_id"], cols["country"]
    s1_country = dict(zip(s1_raw[c_id].astype(str), s1_raw[c_country].astype(str)))
    anchor_country = {a: s1_country.get(a, "UNKNOWN") for a in anchor_ids}
    anchors_df = normalize_frame(
        s1_raw.loc[s1_raw[c_id].astype(str).isin(set(anchor_ids))].copy(), cols)
    anchors_df.to_pickle(out_dir / "run_anchors.pkl")
    pd.DataFrame({"source1_entity_id": anchor_ids,
                  "country": [anchor_country[a] for a in anchor_ids],
                  "n_truth": [len(anchor_truth[a]) for a in anchor_ids]}
                 ).to_csv(out_dir / "run_anchors.csv", index=False)
    logger.info("anchors: %d (positives=%d) in %.1fs",
                len(anchor_ids), len(positives), time.perf_counter() - t0)

    history_common = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/run_blocking_benchmark.py",
        "git_commit": git_commit,
        "config_hash_stage1": cfg.config_hash(),
        "stage2_config": str(args.stage2_config),
        "data_root": str(cfg.data_root),
        "seed": seed,
        "haystack": hay,
        "haystack_sample_n": sample_n,
        "n_anchors": len(anchor_ids),
        "n_positives": len(positives),
    }
    logs_dir = cfg.logs_dir

    def _log_row(exp, desc, hyp, bver, hyper, recall, avg, mx, notes):
        if args.no_log:
            return "skipped(--no-log)"
        return log_experiment_row(root, args.team_member, desc, hyp, bver, hyper,
                                  recall, avg, mx, notes, git_commit, day=args.day)

    # ================= EXP-001: frozen baseline reproduction =================
    if "EXP-001" in wanted:
        exp_dir = out_dir / "exp001_frozen_baseline"
        exp_dir.mkdir(parents=True, exist_ok=True)
        cfg_e1 = dataclasses.replace(cfg, eda_dir=exp_dir, figures_dir=exp_dir / "figures")
        (exp_dir / "figures").mkdir(parents=True, exist_ok=True)
        logger.info("EXP-001: calling frozen run_blocking_eda -> %s", exp_dir)
        t0 = time.perf_counter()
        res = run_blocking_eda(s1_raw, s2_raw, s3_raw, gt_df, None, cfg_e1)
        runtime = time.perf_counter() - t0
        metrics = pd.read_csv(exp_dir / "06_blocking_metrics.csv")
        union_row = metrics.loc[metrics["blocker"] == "union_all"].iloc[0]
        # non-postcode union recall from the coverage matrix (freeze §4 baseline)
        cov = pd.read_csv(exp_dir / "07_blocking_positive_coverage.csv")
        hit_cols = [c for c in cov.columns if c.startswith("hit_") and c != "hit_union_all"]
        non_pc = [c for c in hit_cols if c != "hit_exact_postcode"]
        union_non_pc = float(cov[non_pc].sum(axis=1).gt(0).mean()) if len(cov) else float("nan")
        summary = {
            "frozen_union_recall": float(union_row["overall_recall"]),
            "frozen_union_burden": float(union_row["avg_candidates_per_s1"]),
            "non_postcode_union_recall": union_non_pc,
            "runtime_seconds": runtime,
            "artifacts": res.get("artifacts", []),
        }
        with open(exp_dir / "reproduction_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("EXP-001: union=%.4f @ %.1f/S1 (non-postcode=%.4f) in %.1fs",
                    summary["frozen_union_recall"], summary["frozen_union_burden"],
                    union_non_pc, runtime)
        append_stage2_history(logs_dir, {**history_common, "experiment": "EXP-001",
                                         "status": "done", **summary})
        _log_row("EXP-001", "Stage2 EXP-001: reproduce frozen Stage-1 blocking baseline",
                 "Frozen code+config reproduces 98.7% union recall @ ~99/S1.",
                 "diagnostic-v1 (frozen reproduction)", "frozen sampling",
                 float(union_row["overall_recall"]), float(union_row["avg_candidates_per_s1"]),
                 int(union_row["max_candidates_per_s1"]),
                 f"non-postcode union={union_non_pc:.4f}; runtime={runtime:.0f}s; "
                 f"artifacts=output/stage2/exp001_frozen_baseline/")

    needs_pool = [e for e in wanted if e != "EXP-001"]
    if not needs_pool:
        return 0

    # ---- large haystack pool (normalized once, cached for reruns) ----
    pool = build_pool(s2_raw, s3_raw, cols,
                      sample_n=sample_n if hay == "sample" else None,
                      seed=seed + 1)
    cache_path = cache_dir / f"pool_{hay}_{len(pool)}_seed{seed}.pkl"
    if not args.no_cache and cache_path.exists():
        logger.info("Loading cached normalized pool (%s)...", cache_path)
        pool = pd.read_pickle(cache_path)
    else:
        logger.info("Normalizing pool (%d rows)...", len(pool))
        t0 = time.perf_counter()
        pool = normalize_frame(pool, cols)
        logger.info("Pool normalized in %.1fs.", time.perf_counter() - t0)
        if not args.no_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            pool.to_pickle(cache_path)
            logger.info("Cached pool -> %s", cache_path)
    pool_ids = pool[c_id].astype(str).tolist()
    pool_id_set = set(pool_ids)
    pool_hit_ceiling = float(positives["candidate_entity_id"].astype(str).isin(pool_id_set).mean()) \
        if len(positives) else 1.0
    logger.info("pool=%d rows; pool-hit ceiling=%.4f", len(pool), pool_hit_ceiling)
    chunks = s2.get("chunks", {})
    q_chunk = int(chunks.get("query_chunk", 256))
    d_chunk = int(chunks.get("doc_chunk", 250000))
    cap = int(s2.get("candidate_cap_per_s1", 5000))

    anchor_name_texts = anchors_df["name_norm"].astype(str).tolist()
    anchor_addr_texts = anchors_df["address_norm"].astype(str).tolist()
    anchor_id_list = anchors_df[c_id].astype(str).tolist()

    sparse_cache: Dict[str, SparseTopK] = {}

    def _sparse(field: str) -> SparseTopK:
        if field not in sparse_cache:
            idx = SparseTopK().fit(pool[f"{field}_norm"].astype(str).tolist(),
                                   pool_ids, cfg)
            sparse_cache[field] = idx
        return sparse_cache[field]

    def _eval_and_write(exp_tag: str, pairs: Dict[str, pd.DataFrame],
                        runtimes: Dict[str, Dict[str, float]], extra: Dict,
                        desc: str, hyp: str, bver: str, headline: str):
        exp_dir = out_dir / exp_tag
        metrics, per_pos, overlap = evaluate_blockers(
            pairs, positives, anchor_id_list, anchor_country, len(pool),
            pool_hit_ceiling=pool_hit_ceiling if hay == "sample" else None,
            cap_per_s1=cap, runtimes=runtimes)
        paths = write_experiment_artifacts(exp_dir, metrics, per_pos, overlap, extra)
        logger.info("%s: wrote %d artifacts -> %s", exp_tag, len(paths), exp_dir)
        for _, r in metrics.iterrows():
            logger.info("  %-28s recall=%.4f burden=%6.1f p50=%4.0f p95=%4.0f max=%4d q=%.0fs",
                        r["blocker"], r["overall_recall"], r["avg_candidates_per_s1"],
                        r["p50_candidates_per_s1"], r["p95_candidates_per_s1"],
                        r["max_candidates_per_s1"],
                        r["query_seconds"] if pd.notna(r["query_seconds"]) else -1)
        append_stage2_history(logs_dir, {**history_common, "experiment": exp_tag,
                                         "status": "done", **extra})
        top = metrics.sort_values("overall_recall", ascending=False).iloc[0]
        _log_row(exp_tag, desc, hyp, bver, headline,
                 float(top["overall_recall"]), float(top["avg_candidates_per_s1"]),
                 int(top["max_candidates_per_s1"]),
                 f"grid={headline}; ceiling={pool_hit_ceiling:.4f}; artifacts=output/stage2/{exp_tag}/")
        return metrics, per_pos, overlap

    # ================= EXP-002/003: exact sparse top-k =================
    for exp, field in (("EXP-002", "name"), ("EXP-003", "address")):
        tag = f"exp{exp.split('-')[1]}_{'exact_name' if field == 'name' else 'exact_addr'}"
        if exp not in wanted:
            continue
        idx = _sparse(field)
        qtexts = anchor_name_texts if field == "name" else anchor_addr_texts
        pairs, runtimes = {}, {}
        for k in s2.get("exact_topk_grid", [20, 50, 100]):
            name = f"exact_{field}_top{k}"
            pairs[name] = idx.query(qtexts, anchor_id_list, int(k), q_chunk, d_chunk)
            runtimes[name] = {"fit": idx.fit_seconds, "query": idx.last_query_seconds}
        _eval_and_write(
            tag, pairs, runtimes,
            {"haystack_rows": len(pool), "fit_seconds_shared": idx.fit_seconds,
             "vectorizer": "frozen char_wb 3-5 maxfeat=30000 sublinear"},
            f"Stage2 {exp}: exact sparse {field} TF-IDF top-k on large haystack",
            f"Frozen {field} TF-IDF scales to the full haystack with modest recall loss vs 277K pool.",
            "stage2-exact-tfidf-v1",
            f"topk={list(s2.get('exact_topk_grid', [20, 50, 100]))}")

    # ================= EXP-004/005: IVF ANN =================
    ivf_cfg = s2.get("ivf", {})
    for exp, field in (("EXP-004", "name"), ("EXP-005", "address")):
        tag = f"exp{exp.split('-')[1]}_{'ivf_name' if field == 'name' else 'ivf_addr'}"
        if exp not in wanted:
            continue
        ivf = IVFIndex().fit(
            pool[f"{field}_norm"].astype(str).tolist(), pool_ids, cfg,
            n_components=int(ivf_cfg.get("svd_components", 128)),
            sample_n=int(ivf_cfg.get("svd_sample", 200000)),
            n_clusters=int(ivf_cfg.get("n_clusters", 1024)),
            minibatch_size=int(ivf_cfg.get("minibatch_size", 10000)),
            seed=seed, work_dir=cache_dir / tag,
            dense_chunk=int(chunks.get("dense_chunk", 50000)))
        qtexts = anchor_name_texts if field == "name" else anchor_addr_texts
        pairs, runtimes, scans = {}, {}, {}
        for nprobe in ivf_cfg.get("nprobe_grid", [1, 4, 8, 16]):
            for k in ivf_cfg.get("topk_grid", [20, 50, 100]):
                name = f"ivf_{field}_np{int(nprobe)}_top{int(k)}"
                pairs[name] = ivf.query(qtexts, anchor_id_list, int(k), int(nprobe),
                                        int(ivf_cfg.get("max_scan_per_query", 100000)))
                runtimes[name] = {"fit": ivf.fit_seconds, "query": ivf.last_query_seconds}
                scans[name] = ivf.last_scan_p50
        _eval_and_write(
            tag, pairs, runtimes,
            {"haystack_rows": len(pool), "fit_seconds_shared": ivf.fit_seconds,
             "svd_components": int(ivf_cfg.get("svd_components", 128)),
             "n_clusters": int(ivf_cfg.get("n_clusters", 1024)),
             "scan_p50_by_blocker": scans},
            f"Stage2 {exp}: IVF ANN {field} (nprobe x topk) on large haystack",
            f"IVF recall approaches exact top-k at a fraction of query cost for {field}.",
            "stage2-ivf-v1",
            f"nprobe={list(ivf_cfg.get('nprobe_grid', []))};topk={list(ivf_cfg.get('topk_grid', []))}")

    # ================= EXP-006: unions + overlap =================
    if "EXP-006" in wanted:
        tag = "exp006_unions"
        logger.info("EXP-006: frozen exact blockers at full-pool scale...")
        t0 = time.perf_counter()
        p_exact_name = exact_name_pairs(anchors_df, pool, cols)
        t_name = time.perf_counter() - t0
        t0 = time.perf_counter()
        ret_cfg = cfg.eda.get("retrieval", {})
        p_rare = exact_rare_token_pairs(
            anchors_df, pool, cols,
            rare_max_df=int(ret_cfg.get("rare_token_max_df", 25)),
            rare_min_len=int(ret_cfg.get("rare_token_min_len", 4)))
        t_rare = time.perf_counter() - t0
        logger.info("exact_name=%d exact_rare=%d pairs (%.0fs + %.0fs)",
                    len(p_exact_name), len(p_rare), t_name, t_rare)
        idx_n, idx_a = _sparse("name"), _sparse("address")
        p_n50 = idx_n.query(anchor_name_texts, anchor_id_list, 50, q_chunk, d_chunk)
        q_n50 = idx_n.last_query_seconds
        p_a50 = idx_a.query(anchor_addr_texts, anchor_id_list, 50, q_chunk, d_chunk)
        q_a50 = idx_a.last_query_seconds
        unions = {
            "frozen_exact_union": pd.concat([p_exact_name, p_rare]).drop_duplicates(),
            "exact_tfidf_union_top50": pd.concat([p_n50, p_a50]).drop_duplicates(),
            "full_union_top50": pd.concat(
                [p_exact_name, p_rare, p_n50, p_a50]).drop_duplicates(),
        }
        runtimes = {
            "frozen_exact_union": {"fit": 0.0, "query": t_name + t_rare},
            "exact_tfidf_union_top50": {"fit": idx_n.fit_seconds + idx_a.fit_seconds,
                                        "query": q_n50 + q_a50},
            "full_union_top50": {"fit": idx_n.fit_seconds + idx_a.fit_seconds,
                                 "query": t_name + t_rare + q_n50 + q_a50},
        }
        members = {"exact_norm_name": p_exact_name, "exact_rare_name_token": p_rare,
                   "name_tfidf_top50": p_n50, "address_tfidf_top50": p_a50,
                   **unions}
        runtimes.update({
            "exact_norm_name": {"fit": 0.0, "query": t_name},
            "exact_rare_name_token": {"fit": 0.0, "query": t_rare},
            "name_tfidf_top50": {"fit": idx_n.fit_seconds, "query": q_n50},
            "address_tfidf_top50": {"fit": idx_a.fit_seconds, "query": q_a50},
        })
        _eval_and_write(
            tag, members, runtimes, {"haystack_rows": len(pool)},
            "Stage2 EXP-006: unions + overlap/redundancy on large haystack",
            "Full union preserves ~frozen recall; overlap shows which blockers are redundant.",
            "stage2-union-v1", "unions of frozen-exact + tfidf-top50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
