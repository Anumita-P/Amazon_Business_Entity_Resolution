# Stage 2 Benchmark Report — run1 (large haystack)

- **Outputs branch:** `outputs/stage2-run1` (`42f97e5`), 37 files under `output/stage2/`
- **Scale:** full train haystack 10,320,219 pool rows (S2+S3); 2000 anchors → 7312 positives; pool-hit ceiling 1.0000
- **Runtime:** ~3.4 h wall (18:57→22:21); experiment log EXP-002…007 on arena branch
- **Code:** harness `bcdec04`; Stage-1 config verified frozen (`90d125f76e75`) at startup

## TL;DR

1. **EXP-001 reproduces the frozen baseline EXACTLY**: 0.987266 @ 98.852/S1 (all 6 decimals match the Stage-1 full run). Harness validated.
2. **Production ceiling for this blocker family: 94.05% @ ~102/S1** (non-postcode union at 10.3M scale) — a 4.7 pt drop from the 277K-pool number for a 37× haystack.
3. **Address TF-IDF is the workhorse** (top-50: 83.6%); name top-50 adds 6.7 pts of unique positives. Exact blockers are ~dead weight at top-50 (+5 positives for +4.3 burden).
4. **Rare-token collapsed** (34.7% → 4.0%): absolute DF≤25 does not survive 10M scale. Needs a relative-DF redesign or removal.
5. **IVF-SVD is unusable as designed** (best 41.6% name / 25.8% addr vs 70.0%/85.6% exact) — SVD-128 destroys char n-gram signal, catastrophically on S3 addresses (11%). But it rescues 61/435 union-missed → judge future ANNs on *complement* recall, not standalone.
6. **Hard slice = S3-India** (10.4% miss rate vs 3.5% US). **Country gap persists**: union 90.6% India vs 96.2% US.
7. **Exact sparse queries are too slow for production** (600–1100 s per field-k for 2000 queries); the final blocker needs WAND-style inverted-index retrieval, not dense blocks.

## 1. EXP-001 — frozen baseline reproduction: EXACT

| Metric | Frozen (Stage 1) | EXP-001 | Δ |
|---|---|---|---|
| union recall | 0.987266 | 0.987266 | 0.000000 |
| burden/S1 | 98.852 | 98.852 | 0.000 |
| non-postcode union | 0.9871 | 0.9871 | 0.0000 |

Same code, same config, same sampling → bit-identical metrics. The benchmark's yardstick is trustworthy.

## 2. EXP-002/003 — exact sparse TF-IDF at 10.3M scale

| Blocker | Overall | S1–S2 | S1–S3 | India | US | q-time |
|---|---|---|---|---|---|---|
| name top-20/50/100 | 59.2 / 66.1 / 70.0% | 65.0 | 67.1 | 57.0 | 72.1 | ~600 s |
| addr top-20/50/100 | 79.9 / 83.6 / 85.6% | 88.6 | 78.9 | 78.8 | 86.8 | ~1080 s |

- vs 277K pool (frozen): name top-50 83.9→66.1 (−17.8), addr top-50 91.2→83.6 (−7.6). Names degrade 2.3× faster — shorter strings, denser collision space.
- **S3 address gap confirmed at scale**: 88.6 vs 78.9 (−9.7). No S2/S3 gap for names. Matches the Stage-1 "S3 is the noisy source" finding.
- **Country gap**: name −15.1 pts India vs US; addr −8.0. Cross-script + transliteration-variant positives concentrate in India.
- Diminishing k: 20→50 buys +7.0/+3.7 pts; 50→100 buys +3.9/+1.9. top-50 is the sane operating point.
- Fit cost: name 460 s (480M nnz), addr 1737 s (772M nnz). Index builds are one-time; queries dominate.

## 3. EXP-004/005 — IVF ANN: negative result (with one salvage)

Best-per-field: name np4_top100 41.6%, addr np4_top100 25.8% — 28–60 pts below exact. Verdict: **SVD-128 dense retrieval cannot replace sparse TF-IDF here.** Contributing evidence:

- S3-address collapse: IVF-addr S2 40.6% vs **S3 11.5%** (exact: 88.6/78.9). SVD keeps coarse boilerplate geometry but erases the fine char n-grams that distinguish noisy S3 variants.
- India names hit hardest (IVF-name India 27.6% vs US 50.9%).
- Queries are fast (13–146 s vs 600–1100 s) — speed without recall is worthless, but it proves the harness can measure the tradeoff.
- Salvage: IVF-name uniquely rescues **61 of the 435 union-missed positives** (14%). Lesson for the next ANN candidate: report *complement* recall over the exact union's misses, not just standalone recall.

Harness caveats (both recorded, neither changes the verdict): (a) nprobe 8/16 scans hit the 100K/query cap (scan_p50 = 100000 exactly), so large-nprobe numbers are cap-truncated — that is the entire "nprobe inversion" artifact; (b) the IVF vectorizer fit its vocabulary on the 200K SVD sample, not the full pool (memory-bounded training), so IVF loss conflates vocab sampling + SVD damage. A follow-up should reuse the full-pool vectorizer for a clean ablation.

## 4. EXP-006 — unions + overlap/redundancy

| Blocker | Recall | Burden | P50/P95/Max |
|---|---|---|---|
| exact_norm_name | 21.8% | 9.7 | 1 / 57 / 342 |
| exact_rare_token | 4.0% | 0.7 | 0 / 0 / 25 (1907/2000 anchors: zero candidates) |
| name_tfidf_top50 | 66.1% | 50 | 50 / 50 / 50 |
| address_tfidf_top50 | 83.6% | 50 | 50 / 50 / 50 |
| frozen_exact_union | 24.7% | 10.3 | 2 / 57 / 342 |
| **exact_tfidf_union_top50** | **93.98%** | 97.5 | 98 / 100 / 100 |
| **full_union_top50** | **94.05%** | 101.8 | 98 / 115 / 392 |

Union splits: S2 94.6% / S3 93.5%; India 90.6% / US 96.2%.

Overlap reading (7312 positives):

- addr_top50 ∩ name_top50: both 4076, addr-only 2040, name-only 756 (Jaccard 0.59) — genuinely complementary, keep both.
- Unique rescues (member-only): addr 1953, name 487, exact_name 4, rare 1. The exact blockers contribute **5 positives** to the union (Jaccard exact-vs-TFIDF-union 0.999) at +4.3 burden — redundant at top-50, keep only as a cheap safety net or drop.
- 74 positives hit by all four members; 2445 by exactly one member; **435 missed by all** (5.95%).

Harness caveat: separate-k runs are not strictly nested (top-50 uniquely "rescues" 5 vs top-20/100 in exp002) because each k ran its own argpartition with different tie-breaking. Effect: 5/7312 — negligible; a follow-up can derive smaller-k sets by truncating the top-100 run.

## 5. Missed-positive autopsy (435 positives, 321 anchors)

| Slice | Miss rate |
|---|---|
| S3-India | **10.4%** (155/1493) |
| S2-India | 8.3% (118/1424) |
| S3-US | 3.9% (88/2240) |
| S2-US | 3.4% (74/2155) |

Misses concentrate exactly where Stage 1 said the problem is: India + S3 noise (+ cross-script, unmeasured here but same population). Max 5 missed per anchor — no single catastrophic anchor; the tail is broad, not spiky. Identities: `exp006_unions/missed_positives.csv`.

## 6. Production direction (recommendations, not decisions)

1. **Baseline production union**: addr TF-IDF top-50 + name TF-IDF top-50 (≈94% @ ~98/S1). Exact-name as a cheap add-on; rare-token only after a relative-DF redesign; postcode stays excluded.
2. **Retrieval engine must be inverted-index based** (WAND/MaxScore over the frozen char TF-IDF): same ranking the benchmark validated, at ~100× the query speed of dense blocks. Do NOT "fix" speed with SVD-style dense compression — EXP-004/005 killed that.
3. **Next ANN candidates** (if any): sparse-aware methods (SPLADE-style learned sparse, HNSW over high-dim sparse) or BM25 variants — evaluated on complement recall over the 435 misses + burden, never standalone recall alone.
4. **Slice work**: S3-India needs retrieval help (address normalization variants, transliteration-bridged keys) before the matcher can even see those candidates.
5. **Burden budget**: every +1/S1 is a false-positive lottery ticket for the F0.5 matcher. top-50 union (≈98/S1) is the starting budget; top-100 (+2–4 pts recall, +100/S1) needs matcher-side justification in Stage 4.

## 7. Reproduce

```powershell
python scripts/run_blocking_benchmark.py --data-root "<dataset>" --experiments all 2>&1 | Tee-Object stage2_run.log
```

Expect ~3.4 h, ~12 GB peak RAM (one float32 CSR + query blocks), pool cache reused on reruns. Results: `output/stage2/exp00{1..6}_*/` (metrics / per_positive / missed_positives / overlap.json / pairwise_overlap / runtime.json) + `logs/stage2_history.jsonl` + experiment-log rows. NOTE: never commit `output/stage2/cache/` (multi-GB regenerable files; now gitignored).
