# EDA Fast-Mode Report — Real Data (Stage 1)

> Human-written report from the fast-mode pipeline outputs (branch
> `outputs/real-data-fast-run1`). Numbers below come from `eda/*.csv`,
> `reports/eda_summary.md`, and `logs/run_history.jsonl` of that run.
> Sampled sections are labeled ESTIMATE (fast mode = ~5x smaller samples);
> FULL-data sections are labeled FINAL (identical to what the full run prints).

**Run provenance:** fast_mode=True, code `600edca`, config hash `0b211e0f448f`
(fast-mode hash — differs from full-mode by design), transliteration backend
`builtin`, runtime 1439s (~24 min), 0 errors, 37 artifacts.
**Scale:** train 2.21M / 5.03M / 5.29M (S1/S2/S3), test 1.73M / 4.89M / 5.08M,
ground truth 2.21M rows → **7.64M positive pairs** (3.69M S1–S2 + 3.94M S1–S3).

## TL;DR for the team

1. This is a **many-match task**: 89% of S1 entities have 2+ matches, 48% have
   4+, 80% match in *both* S2 and S3. Only 5.6% are singletons. Top-1
   prediction is invalid; per-entity abstention + multi-pick thresholding is
   required (matters for macro-F0.5).
2. **Names carry the match, addresses confirm it.** 21% of positives match
   exactly on normalized name (median similarity 0.875); only 8% match exactly
   on address (median token-Jaccard 0.625). No meaningful S1–S2 vs S1–S3 gap.
3. **Numeric conflict is the strongest precision signal measured:**
   11.7% of positives vs ~85–89% of hard negatives show conflicting address
   numbers; house-number agreement is 66% vs 1%. Missing numbers (4.5% of
   positives) must stay a separate feature from contradictions.
4. **Blocking already reaches 99.8% recall** (union, ~98 candidates/S1) with
   address-TF-IDF-top20 doing most of the work (92.5% @ 20). Two surprises:
   `exact_postcode` is nearly useless (4.9% recall — real postcodes are noisy
   or absent), and only 14/5603 sampled positives (0.25%) escape all blockers.
   Fix retrieval for those 14 before touching any classifier.
5. **France is the true open-set test:** token-vocab coverage vs train is only
   ~0.19 (vs 0.47–0.60 for US/India) while character n-gram coverage is 0.81+.
   Character-level features transfer; word-token features don't. No
   country-specific logic in the core pipeline.
6. **Multilingual is real but one-sided:** S1 is 100% Latin; all non-Latin
   text lives in S2/S3 (5–11% per field: Devanagari 3–6%, other Unicode
   2–5%, mixed addresses ~9–12%). 6.5% of positives are cross-script (all
   India–India, S1-Latin × S2/S3-non-Latin). Word tokens score exactly 0.0 on
   them; transliteration partially bridges Devanagari (similarity 0.52) but
   not other scripts (0.09). Latin-only matching would be blind to ~7% of
   positives and ~10% of S2/S3 rows.
7. **Transliteration is low-risk AND low-gain on this data:** +80–123 merges
   per 100K rows, **zero** cross-script false-merge groups. Keep it as an
   optional feature, never canonical — it earns its place only on the
   cross-script slice.

## 1. Ground-truth structure (FINAL)

| Matches per S1 | Share |
|---|---|
| 4+ | 48.0% |
| 3 | 24.1% |
| 2 | 17.0% |
| 1 | 5.4% |
| 0 (singleton) | 5.6% |

Patterns: mixed S2+S3 80.5%, S3-only 7.5%, S2-only 6.5%, singleton 5.6%.
Mean matches/S1 ≈ 3.5. GT integrity is perfect (0 unknown IDs, 0 self-refs,
0 dup IDs, 0 bad prefixes) — labels are trustworthy.

**Implication:** the decision layer must emit 0–N matches per entity with a
precision-biased per-entity threshold (Stage 6/7). Any top-K-with-fixed-K
policy is wrong by construction.

## 2. Name and address behavior (ESTIMATE, n=6000 positives)

- Names: exact 21.3%, ratio p50 0.875, token-set p50 1.0 (52.6% exactly 1.0).
  S1–S2 vs S1–S3 splits are near-identical (exact 21.0% vs 21.7%) — keep a
  `source_pair` feature anyway (cheap insurance), but expect no large gap.
- Addresses: exact 8.1%, token-Jaccard p50 0.625, either-side-missing 4.5%.
- Missingness (FINAL): names never missing; country never missing; addresses
  missing **only** in S2/S3 at 2.6–3.4%. S1 is pristine. So "address missing"
  is itself a (weak) source-pair signal, and absence ≠ contradiction must hold
  in features.
- Normalization collisions (FINAL): train normalized-name groups
  180K/392K/398K with largest clusters 253/526/521 — exact-name matching needs
  address/numeric backup. Note: the largest *address* clusters (169K/176K)
  are exactly the empty-string groups — exclude empties before judging
  address-collision risk.

## 3. Numeric agreement (ESTIMATE) — the precision workhorse

| Class | Conflict | Any overlap |
|---|---|---|
| positive | 11.7% | 75.0% |
| name_hard | 87.0% | 0.4% |
| address_hard | 85.0% | 4.0% |
| hybrid_hard | 88.6% | 1.6% |
| random | 88.4% | 0.3% |

House-number 2×2: positives agree 3946 / conflict 704 (n=6000); hard negatives
agree 77 / conflict 6932 (n=8000). A learned numeric-conflict penalty should
be among the strongest Stage-5 features.

## 4. Blocking (ESTIMATE, 1600 anchors / 55.6K pool / 5603 positives)

| Blocker | Overall recall | Avg/S1 |
|---|---|---|
| exact_norm_name | 22.0% | 0.8 |
| exact_rare_name_token | 47.8% | 6.0 |
| exact_postcode | **4.9%** | 0.2 |
| name_tfidf_top20 / top50 | 85.4% / 87.9% | 20 / 50 |
| address_tfidf_top20 / top50 | **92.5%** / 94.1% | 20 / 50 |
| union_all | **99.8%** | 97.8 |

Rescue map: 5570 covered by multiple blockers; uniquely rescued:
address_top50 14, name_top50 3, rare_token 2; **rescued by none: 14**.
S1–S2 vs S1–S3 recall gaps are small (≤4pt) except postcode (useless both).

**Implications:** (a) `address_tfidf_top20` is the recall/burden sweet spot —
keep top50 only if the full run confirms its +1.6pt holds; (b) drop or
redesign `exact_postcode` (4.9% recall means real postcodes rarely match
exactly — normalize/approximate instead of exact-join); (c) the 14 unrescued
positives are the recall ceiling gap — inspect them in `07` + casebook before
any classifier work; (d) graph check: mean S1 degree 97.8, top hub degree 85
(generic "…Private Limited / Solutions" and Devanagari generic names) — hubs
are generic-name records, so downweight common tokens (IDF already does).

## 5. Country shift (FINAL counts, ESTIMATE stats)

- Train: US 60% / India 40% in all three sources. Test: India ~47%, US ~38%,
  **France ~14–15%** in all three test tables (~259K / 703K / 732K rows).
- France vs train vocab (ESTIMATE): **token coverage 0.18–0.20** (US 0.51–0.60,
  India 0.46–0.48) but **char-3-gram coverage 0.81–0.87**. The smoking gun for
  the architecture: word tokens don't transfer to unseen countries, character
  n-grams do.
- Shape shift: India addresses average 60–78 chars vs US 32–39; France ~40–50.
  Length/token-count features must be robust to this (they are — they're
  relative, not absolute).

## 6. Multilingual findings (FINAL mix, ESTIMATE pair stats)

**Script mix (FINAL, FULL data):**
- S1 (train+test): 100% Latin names; addresses 100% Latin except 21 mixed rows
  globally. The reference source is monolingual.
- S2 names: 90.6% Latin / 5.1% Devanagari / 3.9% other-Unicode / 0.4% mixed.
- S3 names: 94.7% Latin / 2.6% Devanagari / 2.0% other-Unicode / 0.7% mixed.
- S2/S3 addresses: ~87% Latin / ~9–12% mixed (script-mixing inside one
  address, e.g. Latin numbers + non-Latin street) / ~3% empty — with essentially
  zero pure-Devanagari addresses. No Cyrillic/Arabic anywhere.
- Non-Latin rows are concentrated in India-tagged S2/S3 records (14: India-S2
  names 27.6% non-ASCII rows in the FULL-rate column). France's 16–28%
  non-ASCII rate is Latin accents (é/è/ç) — same script, still matchable.
- The 2–5% "other Unicode" names (likely other Indic scripts) are the
  coverage gap of the builtin transliterator — flag, don't solve, in EDA.

**Cross-script positives (ESTIMATE, n=6000):** 6.5% overall (387 pairs:
S1–S2 8.3%, S1–S3 4.7%), **all** India–India with S1-Latin × candidate
Devanagari/other/mixed. There are no Devanagari–Devanagari positives *by
construction* (S1 has no Devanagari) — the empty `ml_deva_deva_positive`
bucket is expected, not a gap.

| Pair slice | name similarity p50 | translit similarity p50 |
|---|---|---|
| Latin–Latin | 0.88–0.90 | 0.80–0.87 |
| Latin–Devanagari | 0.10 | **0.52–0.53** (partial bridge) |
| Latin–mixed | 0.60–0.66 | 0.65–0.74 |
| Latin–other-Unicode | 0.10 | 0.09 (no bridge — passthrough) |

Word-token and char-n-gram similarities are exactly 0.0 on cross-script pairs
(16, n=387) — both are script-bound, which is correctness, not failure. On
Latin–Latin positives, char_2-4 (p50 0.726) slightly beats word tokens (0.667)
with hard negatives at 0.21 vs 0.17 — keep both. Cost (17K pairs,
single-thread): word 0.04s vs char 0.45–0.63s (~10–14x) — fine for pair
features, budget for retrieval.

**Transliteration recall-vs-risk (ESTIMATE, 100K rows/table):** +80–123 merges
per 100K names, **0 cross-script collision groups on all six tables**. The
merges are same-script accent folds (café→cafe). Verdict: safe to keep as an
optional agreement feature for the cross-script slice; it contributes nothing
elsewhere.

**Token audit (ESTIMATE):** cross-script positives have word-Jaccard p50 0.0
and token-set p50 0.11 vs 0.667/1.0 on Latin–Latin. English-suffix sharing
(~0.74 on Latin positives) must stay a measured feature, never a strip rule;
Indic-suffix sharing is 0.0 because baseline normalization strips combining
marks — a documented normalization effect, listed explicitly (empty
`ml_indic_suffix_positive` bucket) rather than hidden.

## 7. Casebook pointers (review before Stage 2/3)

All 25-cap buckets are full except the five principled empties
(`ml_deva_deva_positive`, `ml_indic_suffix_positive`,
`ml_stopword_only_overlap`, `neg_exact_address`, `pos_country_mismatch` —
note: zero country-mismatch positives in 6000 suggests country agreement is
near-perfect on matches; verify in the full run). Priority review order:
`ml_latin_deva_positive` → `ml_same_after_transliteration` →
`pos_conflicting_numbers` → `hardneg_high_score` → `large_candidate_group` →
`france_test_nearest_train`.

## 8. Decisions for later stages

| # | Signal | Response |
|---|---|---|
| 1 | 21% exact-name positives, 180–398K norm-name collision groups | Exact name = high-confidence feature/blocker, never sole matcher |
| 2 | Numeric conflict 12% pos vs ~87% hard-neg | Strong learned conflict penalty; missing ≠ conflict |
| 3 | 79% of positives non-exact on names | Address/char-n-gram retrieval + nonlinear pair model |
| 4 | No material S1–S2 vs S1–S3 gap | Keep `source_pair` feature; single calibration likely OK |
| 5 | 5.6% singletons, 89% multi (48% with 4+) | Per-entity 0–N decisions, precision-biased threshold, macro-F0.5 tuning |
| 6 | France token-cover 0.19, char-cover 0.85 | Universal char/token features; no country-gated logic |
| 7 | Generic-name hubs (degree ≤85) | IDF downweighting; tighten broad blocks |
| 8 | Union recall 99.8%, 14 unrescued | Close retrieval gap before classifier work |
| 9 | 6.5% cross-script positives, S1 Latin-only | Keep raw+normalized Unicode + char + token + optional-translit signals |
| 10 | Translit: +~0.1% merges, 0 cross-script groups | Optional agreement feature for the cross-script slice only |
| 11 | Postcode blocker 4.9% recall | Drop exact-postcode join; approximate/normalized postcode instead |

## 9. Caveats: fast mode vs the running full mode

- FINAL already: §§1 counts, §2 missingness/collisions, §5 country counts,
  §6 script mix. These will reproduce **byte-identically** in the full run.
- Estimates to confirm: all rates/medians on sampled pairs (§§2–4, §6 pair
  stats), blocking recall decimals, rare-slice cells (treat n<50 with care).
- The full run (5x samples, ~30–40 min) tightens intervals and fills rare
  buckets; no conclusion above should flip unless a rare slice surprises us —
  that check is the point of running it.
