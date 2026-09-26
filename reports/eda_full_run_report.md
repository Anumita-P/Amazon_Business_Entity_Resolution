# EDA Full-Run Report — Real Data (Stage 1, final)

- **Run:** `scripts/run_eda.py` on the full real dataset, `fast_mode=False`
- **Outputs branch:** `outputs/real-data-full-run1` (`fa6164d`), 37 artifacts (16 CSVs + 20 figures + casebook HTML + summary + run history)
- **Provenance:** 2026-09-26T10:02 UTC · runtime **1985.1s (~33 min)** · config hash `90d125f76e75` (byte-identical to the fast run — same config, no tuning between runs) · code `600edca` · validation **0 errors, 0 warnings**
- **Data:** train S1 2,206,821 / S2 5,034,616 / S3 5,285,603 · test S1 1,732,544 / S2 4,887,273 / S3 5,082,316 · GT pairs 7,638,365 (S1–S2 3,693,619 · S1–S3 3,944,746)
- **Companion:** `reports/eda_fast_mode_report.md` (same structure, fast-mode numbers). This report supersedes it.

## TL;DR

1. **Fast mode is vindicated.** Every pool-independent metric matches within ±0.7 pts (most within ±0.2). The fast report's conclusions stand unchanged.
2. **Blocking works at scale.** Union recall **98.7% @ ~99 candidates/S1** against a 5× bigger haystack (fast: 99.8% against a small pool — optimistic, as predicted).
3. **Names alone cannot separate matches from non-matches.** Mined name-hard negatives reach *higher* name similarity (token_set mean 0.88) than true positives (0.86). Addresses + house numbers do the separating: house-agree 66.5% pos vs ≤4% neg; numeric-conflict 11.9% pos vs 79–87% neg.
4. **A real S2-vs-S3 gap exists in addresses** (not names): addr_exact 12.5% S1–S2 vs 4.2% S1–S3. S3 is the noisy source; it needs more normalization help.
5. **Postcode blocking is dead** (4.3% recall, confirmed at full scale). Drop or redesign.
6. **Cross-script positives are 7.2%** of matches (all India–India, S1-Latin × candidate non-Latin), invisible to every current representation (all sims 0.0). Devanagari transliteration bridges about half the gap (ratio p50 0.53); other scripts (Tamil/Telugu/Gurmukhi/Kannada) get nothing (p50 0.09 — passthrough).
7. **France remains the zero-shot country** (~14–15% of test, 0% of train) with the lowest train-vocab coverage; char n-grams (0.73–0.82 coverage) degrade less than word tokens (0.16–0.20).

## 1. Fast-vs-full verification (the headline table)

Pool-independent metrics (should match — and do):

| Metric | Fast | Full | Δ |
|---|---|---|---|
| GT multi-match (2+) | 89% | 89.0% | 0.0 |
| GT 4+ matches | ~48% | 48.0% | ~0 |
| GT mixed S2+S3 | ~80.5% | 80.5% | ~0 |
| GT singleton | ~5.6% | 5.6% | 0.0 |
| Pos name_exact | 21.3% | 21.9% | +0.5 |
| Pos name_ratio p50 | 0.875 | 0.875 | 0.0 |
| Pos addr_exact | 8.1% | 8.2% | +0.1 |
| Pos addr Jaccard p50 | 0.625 | 0.625 | 0.0 |
| Pos numeric-conflict | 11.7% | 11.9% | +0.2 |
| Pos house-agree | ~66% | 66.5% | ~0 |
| exact_norm_name recall | 22.0% | 21.6% | −0.3 |
| exact_postcode recall | 4.9% | 4.3% | −0.6 |
| Cross-script positives | 6.5% | 7.2% | +0.7 |
| Deva translit ratio p50 | 0.52 | 0.53 | +0.01 |
| char 2–4 pos-Latin p50 | 0.726 | 0.727 | +0.001 |
| Translit cross-script groups | 0 | 1 (size 2, test-S3 addr) | ~0 |

Pool-dependent metrics (expected to drop — bigger haystack is harder; all drops are in the predicted direction):

| Metric | Fast (pool ~55K) | Full (pool 277K) | Δ |
|---|---|---|---|
| Union recall | 99.8% | **98.7%** | −1.0 |
| addr_tfidf_top20 | 92.5% | 89.6% | −2.9 |
| name_tfidf_top20 | 85.4% | 79.9% | −5.5 |
| exact_rare_token | 47.8% | 34.7% | −13.1 |
| Rescued-by-none | 14/5603 (0.25%) | 352/27643 (1.27%) | +1.0 |

Reading: the full-mode recall numbers are the honest ones. A ~1 pt union drop for a 5× haystack is a good trade — blocking generalizes.

## 2. Ground truth (02, FULL counts — final)

- 89.0% of S1 entities have 2+ matches; 48.0% have 4+; max buckets hit 60. Singleton (abstain) rate 5.6%.
- 80.5% of matched entities draw from **both** S2 and S3; S2-only 6.5%, S3-only 7.5%.
- Implication (unchanged): top-1 retrieval is invalid; the final layer must emit ranked lists **or abstain**; macro-F0.5's 1.0/0.0 singleton scoring forces a precision-biased, per-entity threshold (Stage 6/7).

## 3. Positives vs hard negatives (03/04, n=30K pos / 11K neg)

The money table (full-mode means):

| Feature | Positives | name_hard | addr_hard | hybrid | random |
|---|---|---|---|---|---|
| name token_set | 0.86 | **0.88** | 0.32 | 0.76 | 0.33 |
| addr Jaccard | 0.60 | 0.02 | 0.31 | 0.14 | 0.01 |
| numeric-conflict | 0.12 | 0.85 | 0.79 | 0.87 | 0.87 |
| house-agree | 0.66 | 0.005 | 0.04 | 0.02 | 0.001 |
| country_equal | 1.00 | 0.94 | 1.00 | 0.98 | 0.54 |

- **Names are necessary but not sufficient**: the miner finds non-matches with higher name overlap than real matches. Any model must combine name + address + numeric evidence; name-only scoring will drown in false positives.
- **Numerics are the cleanest single signal**: house-agree and numeric-conflict separate pos/neg by ~60–80 pts across ALL neg types.
- Full-mode negatives are *harder* than fast (name_hard token_set 0.88 vs 0.79) — the bigger mining pool finds closer confusers. Good: the model will train against the right difficulty.
- 4.4% of positives have a missing candidate address (vs 0% S1-side): the model must handle address-missing gracefully, never as auto-reject.

## 4. Source-pair and country gaps (03 splits — the new finding)

- **Names: no S2-vs-S3 gap** (exact 21.4% vs 22.3%, p50 0.875 both). **Addresses: big gap** — addr_exact 12.5% S1–S2 vs 4.2% S1–S3; addr Jaccard mean 0.675 vs 0.524. S2 addresses look cleaned/aligned to S1; S3 is raw. S3 pairs will need stronger address normalization and should be over-represented in training pairs.
- **Country gap**: US positives are easier (name_exact ~25–26%) than India (~16%). India carries all the script complexity (§6).
- S1–S3-India postcode overlap is ~0.15% vs ~7–8% for US pairs — another reason postcode blocking fails.

## 5. Blocking (06/07, 8000 S1 × 277K pool — final)

| Blocker | Recall | Burden/S1 |
|---|---|---|
| union_all | **98.7%** | 98.9 |
| address_tfidf_top50 / top20 | 91.2% / 89.6% | 50 / 20 |
| name_tfidf_top50 / top20 | 83.9% / 79.9% | 50 / 20 |
| exact_rare_token | 34.7% | 3.8 |
| exact_norm_name | 21.6% | 1.0 |
| exact_postcode | 4.3% | 0.2 |

- 352/27643 positives (1.27%) rescued by **none**. Only 200 were rescued by exactly one blocker (addr_top50 85, name_top50 75, rare_token 37, postcode 4) — the rest need multiple, i.e. the union has little redundancy to spare. Do not drop any TF-IDF blocker.
- Candidate graph (09): one giant component (213K nodes) + singletons; top hubs (degree ≤174) are all generic/Devanagari India names ("…Private Limited" patterns). Hubs suggest stop-wording the most generic corporate tokens at retrieval time, not deleting them from features.

## 6. Multilingual (11/12/13/14/15/16 — final)

- S1 is 100% Latin (both splits). Non-Latin lives only in S2/S3 candidates: names 5–11% (Deva 3–6%, other-Unicode 2–5%), addresses 9–12% mixed-script.
- Zero Cyrillic/Arabic anywhere. "Other Unicode" is Indic scripts (Tamil, Telugu, Gurmukhi, Kannada per hub inspection).
- Cross-script positives: **2157/30000 (7.2%)**, all India–India S1-Latin × candidate non-Latin. Word and char similarities are exactly 0.0 — correctly measured, genuinely unsolved.
- Transliteration (builtin, additional feature only): merges ~130–170 groups/100K names at ~zero risk (1 cross-script group of size 2 in 3M sampled rows); Deva bridge p50 0.53, other-script p50 0.09 (passthrough — no mapping tables). Char 2–4 beats word tokens on Latin positives (p50 0.727 vs 0.667) and ties at 0.0 on cross-script (correct — shared-script n-grams can't exist).
- Token audit (15): Latin positives share an English corporate suffix 42.9% of the time vs 36.7% for name_hard — suffixes carry weak positive signal, must stay as features, never stripped (per standing §28 rules).
- France (08/14): 15.0% of test S1; token coverage vs train 0.16–0.20 (US/India 0.39–0.54); char-3gram coverage 0.73–0.82. French names are shorter (mean 19.4 vs 22–28) with Latin diacritics (é/è), not non-Latin scripts.

## 7. Data quality (01/05 — final)

- No duplicate entity IDs anywhere; GT referential integrity clean (0 unknown IDs, 0 self-references, 0 dup IDs in lists).
- Names never missing. Addresses missing 3.3% (train S2/S3) and 2.6–2.7% (test S2/S3) — all empty-string, forming the giant norm-address clusters (169K/176K); exclude empties before judging address collisions.
- Normalization is safe: norm_name groups 180K/392K/398K (train S1/S2/S3), largest clusters 253/526/521 — real business-name collisions, plausible sizes. name+address joint key is ~unique (≤5 collisions per table).

## 8. Suggested next steps (Stage 2+)

1. **Freeze Stage 1.** Tag the full-run outputs branch as the EDA reference; no more EDA re-runs unless a data question arises.
2. **Stage 2 — Blocking at full scale.** Implement the union blocker (addr TF-IDF top-50 + name TF-IDF top-50 + rare-token + exact-name; postcode dropped) against the FULL 10M candidate pool with an ANN index (TF-IDF vectors scale poorly — evaluate BM25/FAISS or char-n-gram HNSW). Target: ≥95% recall @ ≤200/S1 on a held-out anchor set. The 1.27% rescued-by-none pairs are the error-analysis seed set.
3. **Stage 3 — Pair features.** Build the comparison-vector generator from §3's table: name sims (ratio/token_set/wratio/JW), address sims (ratio/Jaccard/containments), numeric set features (exact/overlap/conflict/Jaccard), house-number agree, postcode overlap, suffix/stopword flags, script-pair flags, translit sims. Handle address-missing as explicit flags, not imputation.
4. **Stage 4 — Pair model.** Train a nonlinear classifier (GBM first) on GT pairs + mined hard negatives (name/addr/hybrid mix from §3). Oversample S3 and India pairs (harder slices, §4). Calibrate probabilities — they feed the abstention threshold.
5. **Stage 5 — Cross-script & France.** (a) Extend transliteration or add script-bridging features for non-Deva Indic scripts (currently p50 0.09 — the biggest open recall gap, 7.2% of positives). (b) Validate France zero-shot behavior early with a US/India→France split experiment; prefer char-n-gram and numeric features that transfer.
6. **Stage 6/7 — Decision layer.** Per-entity ranked-list emission + abstention threshold tuned for macro-F0.5 on a validation split with singletons included.

## Appendix — reproduce

```powershell
python scripts/run_eda.py --data-root "<dataset root>" 2>&1 | Tee-Object -FilePath eda_full_run.log
```

Expect ~33 min, 37 artifacts, `fast_mode: False`, config hash `90d125f76e75`, 0 validation errors. Figures: `outputs/real-data-full-run1:eda/figures/` (20 PNGs).
