# Campaigns, Diagnostics, and Retention

This file covers the operational layer: running campaigns, diagnosing failures, preserving candidates, and proving that every requested task is accounted for.

## Campaign Runner

Main script:

```text
score_gain_campaign.py
```

Important flags:

```text
--tasks task014,task396,...
--trials-per-task N
--fresh-min 0.90
--workers N
--validation-workers N
--time-limit-seconds N
--task-time-limit-seconds N
--heartbeat-seconds N
--resume
```

Outputs:

```text
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/progress.jsonl
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/task_status.json
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/score_gain_report.json
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/accepted.json
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/near_misses.json
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/rejected_summary.json
.chatgpt/operations/neurogolf_optimizer/score_gain_runs/<run_id>/SCORE_GAIN_REPORT.md
```

## Terminal Status Contract

Every requested task must end as one of:

```text
completed
accepted
near_miss
timeout
interrupted
error
skipped
```

The report must include:

```text
completed_tasks
missing_tasks
running_at_interrupt
accepted_so_far
near_misses_so_far
```

No requested task may silently disappear from `score_gain_report.json`.

## Accounting Bug Fixed

Old behavior:

```text
task_order.json contained 11 tasks
score_gain_report.json contained 10 summaries
task233 was missing
errors reported as 0
```

Fixed behavior:

```text
non-terminal running/pending tasks are finalized to timeout/interrupted
task_status.json is written incrementally
progress.jsonl records start/completion/heartbeat/interruption
```

Proof probe:

```text
run_id: accounting_fix_probe
completed_tasks: ['task014']
missing_tasks: []
task014 status: completed
```

## Diagnostics

Use `pair_runner.py` for fast mismatch localization:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/pair_runner.py \
  --task task014 \
  --model task014/experiments/attempt2.onnx \
  --sources public,arc \
  --comparison official_public_arc \
  --json
```

Use official validator wrapper for acceptance:

```python
from neurogolf_optimizer.validate import validate_model
```

Do not infer acceptance from pair runner alone; pair runner does not measure fresh or cost.

## Near-Miss Repair

Standalone command:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/near_miss_lab.py \
  --task task014 \
  --model task014/experiments/attempt2.onnx \
  --fresh-min 0.90 \
  --repair \
  --dry-run \
  --keep-rejected
```

Shared function:

```python
near_miss_lab.try_repair_model(...)
```

Main optimizer now calls this automatically for near-miss candidates.

Current repair families:

```text
symbolic missing row/column line patch from pair_runner first failing coordinates
mask outside expected shape to no-color
clear channel 0 outside expected shape
tiny coordinate patch from pair_runner first failing coordinates
cleanup safe-build
```

Symbolic line repair is intentionally tried before dense repairs. It currently emits compact row/column/channel masks and an adjacent-extent selector, but task014 evidence shows this selector is still too broad and too expensive:

```text
task014 attempt2 -> symbolic missing_last_column repair
public_wrong=0, arc_gen_wrong=3, fresh=200/200, cost=85000
not accepted
```

Future repairs must restore more precise selector logic from task-derived extent, bbox, and color signals, and must stay below incumbent cost.

## Retention Policy

Default behavior:

```text
keep diagnostic rejected
delete only hopeless build/validation failures
```

Important flags:

```text
--delete-hopeless-rejected
--keep-diagnostic-rejected
--keep-rejected
--keep-high-cost-probes
```

Logs should include:

```json
{
  "model_path": "...",
  "model_file_exists": true,
  "retention_reason": "public_exact|arc_exact|fresh_ge_0.90"
}
```

Diagnostic rejected models are needed for:

```text
pair_runner replay
near-miss repair
mutation history
failure mining
compiler bug reproduction
```

## Useful Trial Metrics

The optimizer writes `campaign_metrics.json` per task run.

Important fields:

```text
trials
build_success_count
build_fail_count
duplicate_count
validated_count
accepted_count
near_miss_count
synth_candidates_validated
synth_candidates_accepted
repair_candidates_validated
repair_candidates_accepted
useful_trial_rate
```

Target:

```text
useful_trial_rate >= 0.50
```

Low useful trial rate usually means:

```text
duplicates are being counted
dynamic-shape candidates are leaking through
compiler unsupported programs are still queued
known failed signatures are not being skipped
validation budget is spent on py_score=0 candidates
```
