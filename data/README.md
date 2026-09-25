# Data directory

## Where does the challenge data go?

The pipeline reads the challenge `dataset/` folder. You have **three** options
(no source-code edits needed in any case):

### Option A — copy/symlink `dataset/` into the repo root (recommended default)

```text
business_entity_resolution/
├── dataset/                 <- put it here
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
```

Then simply run:

```bat
python scripts/run_eda.py
```

Windows symlink (run CMD as Administrator, from the repo root):

```bat
mklink /D dataset "C:\Users\panum\Downloads\amazon\...\dataset"
```

### Option B — pass the path explicitly

```bat
python scripts/run_eda.py --data-root "C:\Users\panum\Downloads\amazon\Amazon ML challenge\student_resource\dataset"
```

### Option C — environment variable

```bat
set BER_DATA_ROOT=C:\Users\panum\Downloads\amazon\Amazon ML challenge\student_resource\dataset
python scripts/run_eda.py
```

Precedence: `--data-root` > `BER_DATA_ROOT` > `config/config.yaml:data_root`.

## Notes

- All files are **tab-separated**; the code always reads with `sep="\t", dtype=str`.
- `dataset/` is git-ignored — raw competition data must never be committed.
- No test files yet? The pipeline still runs: supervised sections use train, and
  test-dependent parts are skipped with a clear note.
- To smoke-test the plumbing without real data:
  `python scripts/make_synthetic_data.py --out dataset-synth` then
  `python scripts/run_eda.py --data-root dataset-synth`.
