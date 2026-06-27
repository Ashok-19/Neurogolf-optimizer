# Agent Upgrade Playbook

This file is the procedure future agents should follow when upgrading the optimizer. It exists to prevent agents from drifting into manual one-task optimization.

## Prime Directive

Improve the optimizer, not just a task.

A task-specific investigation is allowed only when it produces at least one of:

```text
new reusable mutation family
new synthesis DSL primitive
new compiler template
new official diagnostic feature
new repair family
new campaign/search infrastructure
new unit/integration test that prevents regression
```

If the output is only `taskXXX/experiments/some_manual_candidate.onnx`, the work is incomplete.

## Standard Upgrade Loop

1. Select a cross-task weakness.
2. Mine examples from multiple tasks or known near misses.
3. Define a reusable rewrite/search/repair capability.
4. Add or modify optimizer code under `.chatgpt/operations/neurogolf_optimizer`.
5. Add tests using synthetic ONNX models when possible.
6. Run compile and unit tests.
7. Run one or two sentinel task probes only to prove the capability fires.
8. If a sentinel produces accepted gain, let the optimizer promote it through official validation.
9. Record evidence in campaign/run outputs.
10. Update docs if the capability changes architecture or contracts.

## Do Not Do This

```text
Do not manually rewrite a task and stop.
Do not bypass official validation.
Do not promote from pair_runner alone.
Do not treat Python relaxed exactness as success.
Do not overwrite existing candidate.onnx except through promotion.
Do not delete diagnostic near misses.
Do not hide timeouts or interrupted tasks.
Do not call cleanup-only repair a real repair.
Do not report 7900 readiness without accepted projected score >= 7900.
```

## What Counts as a Structural Rewrite Capability

Good optimizer upgrade:

```text
detects a graph pattern across many models
has deterministic metadata
builds a valid static ONNX candidate
preflights shape and dtype compatibility
validates officially
records accepted/near-miss/rejected status
can run in campaigns without manual intervention
```

Weak optimizer upgrade:

```text
hard-codes a task id
assumes one node index from one graph
fixes only one public example
uses dense 30x30 masks for every correction
passes pair_runner but not validate_task.py
improves file size but not official cost
```

## Cross-Task Pattern Mining Procedure

A future miner should scan all available incumbents:

```text
task*/experiments/candidate.onnx if officially accepted
otherwise task*/task/task*.onnx
```

For each model, collect:

```text
op histogram
initializer sizes/dtypes/sparsity
large all-zero/all-false tensors
Concat zero-padding patterns
Pad/Slice/Gather/Scatter patterns
Where branch fanout and dead-branch potential
arange index tensors and work dimensions
one-hot construction patterns
large Constants
large Gather tables and used index ranges
stale value_info / dynamic dims
candidate output encoding structure
```

Rank patterns by:

```text
number of affected tasks
estimated cost reduction
operator legality confidence
shape inference confidence
risk of behavioral change
availability of sentinel tasks
```

The miner output should be machine-readable:

```json
{
  "pattern_id": "concat_zero_tail_uint8",
  "affected_tasks": ["task233", "..."],
  "example_nodes": ["con114", "con348"],
  "estimated_savings": 90,
  "rewrite_family": "concat_pad",
  "ort_dtype_support": "verified",
  "sentinel_task": "task233"
}
```

## New Mutation Family Checklist

Before implementing:

```text
What graph pattern does it match?
Which ONNX operators and dtypes are legal?
Can shape inference prove static outputs?
What is the estimated memory/param saving?
What can go wrong behaviorally?
How will pair_runner explain failures?
Which synthetic test catches the rewrite?
Which sentinel task proves it fires?
```

Implementation:

```text
add file under mutations/
register in mutations/__init__.py
add discovery metadata
add deterministic apply
clear stale value_info
return self.build(...)
add tests
```

Post-implementation:

```text
compileall
unittest discover
run one sentinel task with --families new_family
inspect results.jsonl and campaign_metrics.json
if accepted, verify promotion notes and candidate path
```

## New Synthesis Primitive Checklist

Before adding a primitive:

```text
What ARC rule subproblem does it solve?
Can Python verifier compare it in official_tensor mode?
Can compiler emit static ONNX?
Can output no-color semantics be guaranteed?
What is the expected cost class?
Can it compose in waterfall search?
```

Implementation:

```text
add DSL step behavior
add family generator
add verifier behavior
add ONNX compiler backend
add static preflight
add unit test proving official tensor exactness
add sentinel synthesis run
```

Do not queue a primitive that compiles to cost-null graphs.

## New Repair Family Checklist

Repair should start from official diagnostics:

```text
first failing coordinates
missing active channels
extra active channels
extra channel 0 outside expected shape
expected/predicted active channel histograms
expected grid shape
predicted tensor shape
```

Good repair families are symbolic:

```text
row/column condition
color condition
bbox/span condition
branch selector restoration
single scalar threshold adjustment
small Gather/ScatterElements patch with low params
```

Bad repair families are usually dense:

```text
full 30x30 correction masks
large one-hot replacement tensors
per-example public lookup tables
```

Dense repairs are allowed only when official cost is still lower than incumbent.

## Campaign Procedure for Capability Testing

Short sentinel:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/run_one.py \
  --task taskXXX \
  --families family_name \
  --trials 20 \
  --fresh-min 0.90 \
  --dry-run \
  --keep-rejected \
  --run-id family_probe
```

Small campaign:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/score_gain_campaign.py \
  --tasks taskA,taskB,taskC \
  --families family_name \
  --trials-per-task 50 \
  --fresh-min 0.90 \
  --workers 1 \
  --validation-workers 1 \
  --dry-run \
  --time-limit-seconds 300
```

Promotion campaign:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/score_gain_campaign.py \
  --tasks taskA,taskB,taskC \
  --families family_name \
  --trials-per-task 100 \
  --fresh-min 0.90 \
  --workers 1 \
  --validation-workers 1 \
  --promote \
  --time-limit-seconds 900
```

Never use `--promote` until the family has passed a dry-run probe.

## Evidence to Report

Every substantial optimizer upgrade should report:

```text
files changed
tests added
compile/test commands and results
sentinel tasks run
accepted candidates found
accepted delta
near misses generated
useful_trial_rate
remaining blocker
whether this affects 7900 readiness
```

The correct readiness wording is usually:

```text
This improves optimizer capability, but it is not 7900+ ready until accepted campaign gain proves it.
```

## Current Highest-Leverage Next Task

Build the cross-task graph pattern miner. The current optimizer has enough validation, retention, and campaign infrastructure to benefit from it.

Expected deliverables:

```text
pattern_miner.py or analysis extension
machine-readable pattern report
top rewrite opportunities across all 400 tasks
at least one new reusable mutation family from mined results
tests for the new family
short campaign proof
```

This is more likely to move score than another hand-run near-miss investigation.

