# NeuroGolf Optimizer Agent Documentation

This directory is written for future agents that will modify and run the optimizer. It is not marketing documentation. It is the operating manual for pushing the optimizer from a small accepted gain system toward a valid 7900+ NeuroGolf score.

## Read Order

1. `STATUS_TRIED_TESTED_ROADMAP.md`
   - Current score state, accepted gain proof, failed attempts, readiness answer, and roadmap.
2. `EVALUATOR_CONTRACT.md`
   - Official validator rules. Read this before changing diagnostics, synthesis, or promotion.
3. `ARCHITECTURE.md`
   - Main modules and execution flow.
4. `SEARCH_AND_REWRITE_SYSTEM.md`
   - Mutation families, queueing, repair loops, and how to add reusable structural rewrites.
5. `SYNTHESIS_AND_RULE_SEARCH.md`
   - DSL synthesis, waterfall search, official-compatible Python verification, and compiler limits.
6. `ONNX_COST_ENGINEERING.md`
   - Scorer behavior, static shape requirements, operator/dtype traps, and high-value graph patterns.
7. `CAMPAIGNS_DIAGNOSTICS_AND_RETENTION.md`
   - Campaign accounting, pair diagnostics, repair lab, retention policy, and useful-trial metrics.

## Non-Negotiable Goal

The optimizer is successful only when it produces accepted score gain under the official validator:

```text
public_wrong == 0
arc_gen_wrong == 0
fresh_right / fresh_total >= fresh_min
candidate_cost < current_incumbent_cost
```

Default:

```text
fresh_min = 0.90
```

Coverage, candidate count, Python-only exactness, relaxed grid exactness, and near-miss count are diagnostics only. They are not success metrics.

## Workspace Contract

Allowed writes for optimizer work:

```text
.chatgpt/operations/neurogolf_optimizer/**
taskXXX/experiments/**
```

Do not modify task source files:

```text
taskXXX/task/taskXXX.onnx
taskXXX/task/arcgen_task.py
taskXXX/validate_task.py
taskXXX/baseline_stats.json
```

Existing `taskXXX/experiments/candidate.onnx` files must not be overwritten unless the new model is officially accepted and lower cost.

## Current Reality

The optimizer now has one small accepted gain:

```text
task233: incumbent 71718 -> 71679
accepted delta = 0.0005439444473971378
projected score = 7157.530543944447
```

This proves the optimizer can find and promote a valid candidate, but it is not remotely 7900+ ready.

The remaining gap to 7900 is about:

```text
7900 - 7157.530543944447 = 742.469456055553
```

That scale cannot be reached with pruning-only or local cleanup. It requires many rule-level structural rewrites: compact generated graphs that solve task rules directly.

## Top-Level Entrypoints

Single task:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/run_one.py \
  --task task233 \
  --families concat_pad \
  --trials 5 \
  --fresh-min 0.90 \
  --promote
```

Campaign:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/score_gain_campaign.py \
  --tasks task014,task396,task066,task233 \
  --trials-per-task 200 \
  --fresh-min 0.90 \
  --workers 2 \
  --validation-workers 1 \
  --time-limit-seconds 900
```

Strict diagnostic comparison:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/pair_runner.py \
  --task task031 \
  --model path/to/model.onnx \
  --sources public,arc \
  --comparison official_public_arc \
  --json
```

Near-miss lab:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/near_miss_lab.py \
  --task task014 \
  --model task014/experiments/attempt2.onnx \
  --fresh-min 0.90 \
  --repair \
  --dry-run \
  --keep-rejected
```

Required health checks:

```bash
PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer
PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
```

Current passing test count after the cache-profile and symbolic-repair patch:

```text
55 tests OK
```

Cross-task pattern mining:

```bash
PYTHONPATH=.chatgpt/operations python3.12 -m neurogolf_optimizer.pattern_miner \
  --all \
  --max-patterns 50
```

This writes a machine-readable ranked pattern database and Markdown report under:

```text
.chatgpt/operations/neurogolf_optimizer/runs/pattern_miner_*/
```

## Core Agent Rules

- Use official validator results for acceptance.
- Use `pair_runner --comparison official_public_arc` for public/ARC diagnostics.
- Use `fresh_grid` comparison only for fresh diagnostics.
- Do not count dynamic-shape or cost-null candidates as useful.
- Do not delete diagnostic rejected models needed for repair.
- Prefer reusable optimizer families over task-specific one-off repairs.
- A task-specific probe is acceptable only if the result is generalized into optimizer logic and tested.
- If a candidate fails official validation, record the exact mismatch and use it to improve diagnostics or repair logic.
