# Bucket 34 checkpoint index

Branch: `bucket34-checkpoints`

Paused: 2026-07-12

## Verified state

- Baseline score: `183.0379625777926`
- Current score: `185.0347274703821`
- Verified gain: `+1.9967648925895`
- Target: `+10`
- Remaining: `+8.0032351074105`

## Active exact models

- Task143: cost `495`, gain `+0.337914`, public/ARC-GEN wrong `0/0`, fresh `1000/1000`
- Task214: cost `343`, gain `+0.425668`, public/ARC-GEN wrong `0/0`, fresh `200/200`
- Task215: cost `190`, gain `+0.633509`, public/ARC-GEN wrong `0/0`, fresh `1000/1000`
- Task224: cost `1175`, gain `+0.343741`, public/ARC-GEN wrong `0/0`, fresh `500/500`
- Task225: cost `792`, gain `+0.255933`, public/ARC-GEN wrong `0/0`, fresh `200/200`

## Resume policy

Only promote new models with approximately `+1` point or close to it. Prefer larger structural gains. Keep the existing exact models unless a stronger candidate passes all promotion gates.

## Important files

- `RESUME_STATUS.json`: exact score state, model hashes, validation summaries, and unfinished work.
- `manifest_resume.json`: normalized manifest with consistent totals.
- `score_report_resume.json`: per-task baseline/current costs and gains.
- `task224/build_scalar_ngram7.py`: deterministic builder for the large active Task224 model.
- `task224/exact_scalar_ngram7_50.json`: exact cost report.
- `task224/verify_scalar_ngram7_500.json`: stronger fresh validation report.
- `task215/build_periodic_scaled.py`: unvalidated high-value candidate; do not promote without full gates.
- `task048/`: rejected DFA investigations retained to avoid repeating the state-explosion path.
- `task081/`, `task160/`, `task206/`: rejected structural candidates retained with failure evidence.
- `task226/prototype_cumsum.py`: exact NumPy cumulative-state formulation; not promoted because estimated ONNX cost was not competitive.

## Binary preservation

The active Task143, Task214, Task215, and Task225 ONNX binaries were committed earlier on this branch. Task224 is reproducible from its deterministic builder and exact model SHA-256 recorded in `RESUME_STATUS.json`:

`b6290ce1b0049d45ba6cd0fff32c7f864008730c38376276bb19ac17ba59e199`

A complete local resume ZIP was also created with SHA-256:

`9d122ae0ef801b71b2dcefcdd1f3990605122c4213a133666a99c81007b04c91`
