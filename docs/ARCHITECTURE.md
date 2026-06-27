# Optimizer Architecture

The optimizer is a local system for generating, validating, repairing, retaining, and promoting ONNX candidates under the official NeuroGolf validator.

## Main Modules

```text
run_one.py
  Single-task CLI wrapper around NeuroGolfOptimizer.optimize_task.

score_gain_campaign.py
  Multi-task campaign runner with progress accounting, time limits, resume, and report output.

optimizer.py
  Main optimization loop. Chooses incumbent, discovers mutations, builds candidates, validates candidates, runs automatic near-miss repair, promotes accepted candidates, and writes reports.

models.py
  Dataclasses for config, validation results, mutations, candidates, task context, and analysis.

config.py
  Common CLI flags and conversion into OptimizerConfig.

validate.py
  Official validator subprocess wrapper. Parses JSON emitted by taskXXX/validate_task.py.

pair_runner.py
  Lightweight ORT runner and official-compatible pair diagnostic engine.

near_miss_lab.py
  Shared repair implementations plus a standalone CLI. Main optimizer now calls try_repair_model from this module.

mutation_base.py
  Base mutation family interface and safe build wrapper.

mutations/**
  Reusable graph rewrite families.

synthesis/**
  Rule/program DSL, Python verification, ONNX compiler, waterfall synthesis, and task data loading.

search/**
  Beam/evolutionary queue helpers and subset near-miss repair helpers.

reporting.py
  JSON/JSONL/Markdown reporting helpers.

scoring.py
  Acceptance, near-miss categories, score delta, and fresh-rate helpers.
```

## Task Data Flow

1. `run_one.py` or `score_gain_campaign.py` builds `OptimizerConfig`.
2. `NeuroGolfOptimizer.make_context(task_id)` creates a `TaskContext`.
3. `choose_incumbent` selects:
   - `taskXXX/experiments/candidate.onnx` if it exists and is officially accepted versus baseline.
   - otherwise `taskXXX/task/taskXXX.onnx`.
4. `_families` chooses mutation families based on graph analysis and CLI filters.
5. `_discover_mutations` calls each family `discover`.
6. `_make_queue` builds mutation sets:
   - singles
   - beam combinations
   - evolutionary sampled combinations
7. `_build_candidate` applies mutations to a parent model.
8. `_validate_batch` runs the official validator wrapper.
9. Accepted candidates may be promoted.
10. Near misses trigger automatic repair attempts.
11. Reports and state files are updated incrementally.

## Current Search Loop Invariants

The optimizer tracks:

```text
visited_hashes
failed_build_signatures
accepted rows
near-miss rows
trial count
candidate index
incumbent model path
incumbent cost
```

Duplicates are not useful trials. Known failed build signatures are skipped.

The queue is sorted by:

```text
sum(estimated_savings)
sum(priority)
shorter mutation chains
```

This ordering is intended to spend validation budget on higher-value structural rewrites first.

## Candidate Lifecycle

A candidate has:

```text
parent_model
model_path
mutations
validation
build_error
model_hash
cost
delta
accepted
near_miss
near_miss_category
retention_reason
model_file_exists
```

Every accepted, near-miss, failed build, and repair candidate should end up in JSONL output.

## Promotion Path

Promotion happens in `_promote`:

1. Skip if not accepted, dry-run, or `--promote` absent.
2. Lock `taskXXX/experiments`.
3. Revalidate candidate with official validator.
4. Validate current `experiments/candidate.onnx` if present.
5. Copy old candidate to `candidate_prev_<timestamp>.onnx`.
6. Copy new candidate to `experiments/candidate.onnx`.
7. Copy new candidate to run `best.onnx`.
8. Write accept notes.

No accepted candidate should be promoted only from cached or stale validation.

## Validation Cache

`ValidationCache` keys results by:

```text
task
model hash
validation profile
```

The validation profile includes validator mode, quick/full command mode, quick flag, and timeout. This prevents quick validation results from being reused for stricter/full validation runs.

## Campaign Architecture

`score_gain_campaign.py` runs requested tasks and writes:

```text
score_gain_runs/<run_id>/progress.jsonl
score_gain_runs/<run_id>/task_status.json
score_gain_runs/<run_id>/score_gain_report.json
score_gain_runs/<run_id>/accepted.json
score_gain_runs/<run_id>/near_misses.json
score_gain_runs/<run_id>/rejected_summary.json
score_gain_runs/<run_id>/SCORE_GAIN_REPORT.md
```

Every requested task must finish as one of:

```text
completed
accepted
near_miss
timeout
interrupted
error
skipped
```

The report must not silently omit a requested task.

## Automatic Near-Miss Repair in Main Optimizer

When a validated candidate is near-miss, `optimizer.py` now calls:

```python
near_miss_lab.try_repair_model(...)
```

The repair candidates are adapted into normal `Candidate` objects and go through:

```text
official validation result
accepted/near-miss classification
retention
promotion
accepted score accounting
```

This closes the old gap where repair lived only in `near_miss_lab.py`.
