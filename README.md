# NeuroGolf ONNX Optimizer

Production-oriented optimizer for task-kit ONNX models. It searches from the official baseline or current `experiments/candidate.onnx`, writes generated models only under `taskXXX/experiments/optimizer_runs/<run>/`, validates with `taskXXX/validate_task.py`, caches by SHA256, and promotes only validated lower-cost candidates.

## Commands

Mine existing experiments:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/mine_existing.py --all --fresh-min 0.90 --promote
```

Run one task:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/run_one.py --task task066 --trials 1000 --beam-size 20 --max-combo-size 5 --fresh-min 0.90 --promote
```

Run all tasks:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/run_all.py --trials-per-task 5000 --beam-size 32 --max-combo-size 6 --fresh-min 0.90 --workers 8 --promote
```

Inspect results:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/inspect_results.py --top 50
```

## Promotion Rule

A candidate is accepted only when `public_wrong == 0`, `arc_gen_wrong == 0`, `fresh_right / fresh_total >= --fresh-min`, and `candidate_cost < current_incumbent_cost`. `--require-exact` and `--require-fresh-exact` make this stricter. `--public-arc-only` ignores fresh results. `--dry-run` validates and logs but never copies to `experiments/candidate.onnx`.

Before promotion, the optimizer validates the existing candidate and compares cost. Existing candidates are backed up as `candidate_prev_<timestamp>.onnx`.

## Search

The optimizer runs guided brute force:

- cleanup and dedup mutations first
- task analysis tags prioritize Where, table, dtype, concat, and conv families
- validation cache avoids repeated work by task and model hash
- accepted and near-miss candidates are kept in a beam
- promising mutations are combined up to the configured combo size
- near-miss combos are repaired by testing subsets

High-risk families are conservative and validation-driven. Analysis-only mutations are logged as build failures instead of crashing the campaign.
