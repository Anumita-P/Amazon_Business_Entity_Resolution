# STAGE 1 FREEZE — Business Entity Resolution

**Status: FROZEN.** This document is the stable reference point for all downstream stages.
Any change to a `DO NOT CHANGE` item requires a new dated amendment section at the bottom —
never a silent edit.

## 1. What was frozen (provenance)

| Item | Frozen value |
|---|---|
| Dataset | 6 TSVs (`sep="\t"`, `dtype=str`): train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316; GT 2,206,821 rows → 7,638,365 positive pairs (S1–S2 3,693,619 / S1–S3 3,944,746) |
| Code | `600edca` (`scripts/run_eda.py`, `src/business_entity_resolution/`) |
| Config | `config/config.yaml`, hash `90d125f76e75` |
| Full EDA run | `fast_mode=False`, 1985.1s, 2026-09-26T10:02 UTC, validation 0 errors / 0 warnings |
| Outputs | branch `outputs/real-data-full-run1` (`fa6164d`) — 16 CSVs + 20 figures + casebook + summary + run history |
| Fast EDA run | branch `outputs/real-data-fast-run1` — estimates verified against full (all pool-independent metrics ±0.7 pts) |
| Reports | `reports/eda_fast_mode_report.md`, `reports/eda_full_run_report.md` |

## 2. Frozen normalization (v1)

`normalize_basic` (`src/business_entity_resolution/normalization.py:63`) — the ONLY normalization
used for retrieval keys and (later) pair features:

> NFKC → lowercase → `&`→`and` → keep Unicode alphanumerics + whitespace, everything else to space → collapse whitespace.

Preserves: digits, Unicode letters (incl. Devanagari/Indic), legal suffixes, word order.
`normalize_aggressive` (suffix-strip + token-sort) is **collision-analysis-only** — it must never
become a retrieval key or match rule. Transliteration (`builtin` backend) is an **additional
feature only, never canonical**; raw strings are always preserved.

## 3. Frozen blocker definitions (v1)

All blockers operate on normalized text (`name_norm` / `address_norm`) except postcode (raw address):

| Blocker | Exact definition |
|---|---|
| `exact_norm_name` | Equi-join on `name_norm`; empty keys excluded both sides |
| `exact_rare_name_token` | Candidates sharing ≥1 rare name token; rare = norm-name token with length ≥ 4 and pool document-frequency ≤ 25 |
| `name_tfidf_top{20,50}` | `TfidfVectorizer(analyzer=char_wb, ngram_range=(3,5), max_features=30000, sublinear_tf=True, lowercase=False)` fit on pool `name_norm`; per-S1 top-k cosine (query chunk 256) |
| `address_tfidf_top{20,50}` | Same vectorizer fit on pool `address_norm`; per-S1 top-k cosine |
| `exact_postcode` | Candidates sharing ≥1 postcode-like token = all-digit token of length 5–6 from raw address — **EXCLUDED from production union (see §6)** |
| `union_all` | Dedup union of all enabled blockers; safety cap 5000/S1/blocker (diagnostics only; never hit, `truncated=0`) |

## 4. Frozen baseline metrics (full run)

> **Baseline blocking:** union of current non-postcode blockers = **98.7% positive recall at ~99 candidates/S1** (measured: 8000 S1 × 277K pool, 27643 positives, 352 rescued-by-none).

| Blocker | Recall | Burden/S1 |
|---|---|---|
| union_all | 98.7% | 98.9 |
| address_tfidf_top50 / top20 | 91.2% / 89.6% | 50 / 20 |
| name_tfidf_top50 / top20 | 83.9% / 79.9% | 50 / 20 |
| exact_rare_name_token | 34.7% | 3.8 |
| exact_norm_name | 21.6% | 1.0 |
| exact_postcode | 4.3% | 0.2 |

Key EDA facts Stage 2+ may rely on:

- GT: 89.0% multi-match (48.0% 4+), 80.5% mixed S2+S3, 5.6% singleton → top-1 invalid; final layer must abstain; macro-F0.5 needs precision-biased per-entity thresholds.
- Positives (n=30K): name_exact 21.9% (p50 ratio 0.875); addr_exact 8.2% (Jaccard p50 0.625); house-agree 66.5%; numeric-conflict 11.9%; country_equal 100%.
- Hard negatives: name_hard token_set mean **0.88 vs positives 0.86** → names alone cannot separate; numerics separate (conflict 79–87% neg vs 12% pos; house-agree ≤4% neg vs 66.5% pos).
- S2-vs-S3: names equal, addresses differ (addr_exact 12.5% vs 4.2%) → address-mismatch means something different per source; matcher must be source-conditioned.
- Cross-script positives 7.2%, all India–India S1-Latin × non-Latin; all current sims 0.0; Deva translit bridge p50 0.53, other scripts 0.09.
- France: 15% of test, 0% of train; token coverage 0.16–0.20, char-3gram 0.73–0.82.
- Data quality: no dup IDs; GT integrity clean; names never missing; addresses missing 3.3% train / 2.7% test (S2/S3 only, empty-string).

## 5. Validated / ruled out / uncertain

**Validated:** sampled EDA estimates (fast≈full); blocking union recall at 277K scale; numeric/house-number dominance; normalization safety (largest clusters 253/526/521 — plausible); transliteration ~zero-risk merges (~130–170/100K, 1 cross-script group of size 2 in 3M rows).

**Ruled out:**

> **Postcode blocking is excluded from the production candidate union unless a future experiment demonstrates otherwise.**

Also ruled out: exact-name as sole strategy (21.6%); name-only nearest-neighbor matching (hard negs outscore positives); aggressive normalization as a key; stripping suffixes/stopwords by default; Latin-only / English-only assumptions; collapsing Unicode to ASCII.

**Uncertain (Stage 2 must resolve):** recall of each blocker against the FULL ~10M haystack (frozen numbers are at 277K pool); whether TF-IDF top-k survives ANN approximation; smallest candidate set preserving sufficient recall; identity/composition of the 1.27% union-missed positives at full scale; France zero-shot blocking behavior (no GT — proxy evaluation needed).

## 6. DO NOT CHANGE (without dated amendment below)

1. Normalization v1 (`normalize_basic`) — no new stripping, folding, or key definitions.
2. Baseline blocker definitions (§3) and baseline metrics (§4) — Stage 2 reproduces them first, then benchmarks alternatives *against* them.
3. Postcode exclusion from the production union.
4. Raw-value preservation; country open-set; transliteration as feature-only.
5. Multilingual rules (§28): no English-only defaults, no script collapsing, suffixes as features.
6. Offline-first evaluation: blocking experiments do not consume leaderboard submissions (only ~15 total; reserved for entity-level prediction changes).

## 7. Stage 2 interface

**May consume:** frozen `normalize_basic`/`tokenize`/script/char-n-gram functions; frozen baseline blocker code; `eda/06–07` metrics + missed-positive identities; GT pairs; `logs/experiment_log.csv` (append-only, strict schema: Exp / Change / Local recall / Candidates-S1 / F0.5 / Submission).

**Must produce:** a blocking **benchmark harness** (not the final blocker): reproduce frozen baseline → benchmark scalable ANN generators independently + in combination → per-blocker: recall, burden, P50/P95/max per S1, S2/S3 recall separately, country recall separately, missed-positive identities — measured on the **large haystack**, not small pools.

**Must NOT do:** modify normalization; build pair features; train a matcher; submit to the leaderboard for blocking-only changes.

**Objective:** the smallest candidate set that preserves sufficient positive recall — recall and burden jointly, since every extra candidate is another false-positive opportunity for the F0.5-scored matcher.

## Amendments

_(none yet — append dated entries here, never edit §§1–7 silently)_
