# Experiment logging — team workflow (READ THIS FIRST)

We have **3 people, 3 days, ~5 leaderboard submissions/day**. This folder is how
we avoid losing track. There are **two** logs with different jobs:

| File | Written by | Purpose |
|---|---|---|
| `logs/run_history.jsonl` | pipeline (automatic) | machine-readable record of **every execution**: command, config hash, git hash, dataset mtimes, artifacts, runtime, and whether the run looks identical to the previous one |
| `logs/experiment_log.csv` | humans via `scripts/log_experiment.py` | human-readable record of **what changed, what was rerun, what was submitted** |

## Golden rules

1. **Never edit or delete old rows.** Append only. History is sacred.
2. **Rerun ≠ new experiment.** If nothing changed (same code/config/data), record
   `experiment_type = rerun` and set `parent_experiment_id` to the original run.
3. **Every leaderboard upload gets a row** with `experiment_type = submission`,
   `public_submission = yes`, `submission_number = 1..5`, and the score filled in
   after the portal returns it (`public_leaderboard_score`).
4. During EDA, model columns stay blank/`NA` — that is expected.

## What counts as a NEW experiment?

Anything substantive: normalization change, blocker change, top-k change, new
feature, model/threshold change, negative-sampling change, hyperparameter change,
different validation split. When in doubt, make it `new` and link the parent.

## How to log (humans)

Interactive (asks you questions):

```bat
python scripts/log_experiment.py
```

Non-interactive:

```bat
python scripts/log_experiment.py --team-member Asha --experiment-type new --day 1 ^
  --description "TF-IDF name top-k 20->50" --hypothesis "Rescue transliterated positives" ^
  --hyperparameters "retrieval.name_topk=[20,50]" --yes
```

Rerun of an earlier experiment (nothing changed):

```bat
python scripts/log_experiment.py --rerun EXP-003 --team-member Asha --notes "Re-ran after laptop reboot"
```

Submission record (score filled in after the portal responds):

```bat
python scripts/log_experiment.py --team-member Asha --experiment-type submission --day 2 ^
  --parent-experiment-id EXP-017 --public-submission yes --submission-number 3 ^
  --description "Day-2 submission 3 from EXP-017" --yes
```

Review recent history:

```bat
python scripts/log_experiment.py --list 15
```

## How to read the automatic run log

`logs/run_history.jsonl` is one JSON object per line. Useful fields:

- `command`, `timestamp_utc`, `git_commit`, `config_hash`
- `dataset_files`: per-file `{exists, mtime_utc, size_bytes}`
- `identical_to_previous_run`: `true` means "same command + config + code + data
  mtimes as the last run" → log it as a **rerun**, not a new experiment
- `generated_artifacts`, `runtime_seconds`, `status`

Quick peek (PowerShell):

```powershell
Get-Content logs\run_history.jsonl -Tail 2
```

## Submission-day checklist (5/day)

At the end of each day the log should show, per submission: which experiment it
came from (`parent_experiment_id`), who submitted it, the submission number, and
the leaderboard score. If a row is missing any of those, the next person to touch
the log fills it in — never leave a submission unrecorded.
