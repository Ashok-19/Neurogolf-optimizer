# NeuroGolf Optimizer Worklog

## 2026-06-27: Cross-task pattern miner scaffold

- Added `pattern_miner.py` to scan all selected task incumbents, prefer accepted `experiments/candidate.onnx` only when optimizer accept notes exist, aggregate graph-pattern hits, and write ranked JSON/Markdown reports.
- Added `broadcast_scalarize` and `where_simplify` mutation families with conservative shape guards. The final all-task miner report shows no current safe live hits for these families after guard tightening.
- Updated docs for the miner entrypoint and new rewrite families.
- Validation:
  - `PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer`
  - `PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q` -> 51 tests OK
- Pattern miner proof:
  - `pattern_miner_probe4`: 400 tasks scanned, 63 opportunity classes, 0 errors.
  - Top classes: Where/selector rewrites, dtype lowering, initializer/value-info refresh, bool-mask rewrites, scalar-label lowering, arange work dimensions.
- Campaign proof:
  - `miner_top_family_probe` dry-run on `task009,task074,task054` with `dtype_lowering,scalar_label_rewrite`.
  - 3 tasks completed, 0 errors, 0 accepted, 0 near misses, total accepted delta 0.
- Readiness:
  - Projected score remains 7157.530543944447.
  - This is not 7900+ ready.

## 2026-06-27: Cache profile and symbolic near-miss repair patch

- Fixed `ValidationCache` to key entries by `(task, model_hash, validation_profile)` instead of only `(task, model_hash)`.
  - Profile includes validator mode, quick/full command mode, quick flag, and timeout.
  - Existing SQLite cache rows are migrated to a `legacy` profile instead of being reused accidentally.
- Improved Where/selector mutation shape safety.
  - `where_ablation` now uses non-strict ONNX shape inference during discovery.
  - Broadcast branch replacements carry `selected_shape`, `output_shape`, and insert static `Expand` when needed.
  - task018 selector probe moved from shape-inference build failures to valid rejected candidates; no accepted gain.
- Added early symbolic near-miss repair.
  - Classifies missing row/column line failures from `pair_runner` first failing coordinates.
  - Emits row/column/channel mask patches with an adjacent-extent selector before dense repairs.
  - Emits opset-aware `ReduceSum` for opset 12 versus opset 13+ models.
- Validation:
  - `PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer`
  - `PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q` -> 55 tests OK
- Focused proof:
  - `task014/experiments/attempt2.onnx`: baseline near miss remains public_wrong=0, arc_gen_wrong=1, fresh=200/200, cost=7905.
  - `repair_symbolic_missing_last_column_0` builds and validates but is not accepted: public_wrong=0, arc_gen_wrong=3, fresh=200/200, cost=85000.
  - This proves the symbolic repair path works mechanically, but selector logic and cost are still insufficient.
- Readiness:
  - Accepted gain unchanged.
  - Projected score remains 7157.530543944447.
  - This is not 7900+ ready.

## 2026-06-27: Failure-guided internal selector/extent repair

- Added `repair_classification.py` with reusable pair-diagnostic failure classification.
  - Covers `missing_last_column`, `missing_column`, `missing_bottom_row`, `missing_row`, `missing_border`, `wrong_color_region`, `missing_object_pixels`, and `extra_active_black_outside_shape`.
- Added an internal-first repair path in `near_miss_lab.py`.
  - Runs before symbolic/final-output fallback repairs.
  - For dimension-sweep-style line loss, restores only the affected row/col arange and matching `pad_to_30_i64` extent.
  - Reuses an existing target-length arange initializer when available instead of cloning a new one.
  - For bottom-row extent loss with no shortened arange, tries a conservative scalar extent `+1` internal candidate.
- Added pre-validation repair cost/risk reporting.
  - Repair rows now include `repair_family`, `failure_class`, `internal_or_dense`, `source_mutation_family`, `estimated_repair_cost`, `actual_validation_cost`, `primary_repair`, and `cost_gate`.
  - Dense/full-output fallback repairs are marked non-primary.
  - Full `[1,10,30,30]` replacement initializers and very high estimated costs are reported in `cost_gate`.
- Enriched mutation provenance.
  - `dimension_sweep`: source arange initializer, old/new len, axis, related axis initializers, pad old/new values.
  - `where_ablation`: condition tensor, selected/dropped branch tensors, output/dropped shape, graph-output flag.
  - `where_combo`: explicit per-choice metadata.

Validation:

```text
PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer
passed

PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
Ran 61 tests in 26.552s
OK
```

Sentinel probes:

```text
task014/experiments/attempt2.onnx
incumbent cost: 7983
near miss input: public_wrong=0, arc_gen_wrong=1, fresh=200/200, cost=7905

repair_internal_missing_last_column_0_dimension:
  family: internal_selector_extent
  class: missing_last_column
  internal_or_dense: internal
  source_mutation_family: dimension_sweep_inferred
  result: public_wrong=0, arc_gen_wrong=0, fresh=200/200, cost=7983
  accepted: false
  blocker: exact but cost equals incumbent; needs cost below 7983

repair_symbolic_missing_last_column_0:
  internal_or_dense: final_output_symbolic
  primary_repair: false
  cost_gate: fallback_not_primary_internal_repair|estimated_cost_far_above_incumbent
  result: public_wrong=0, arc_gen_wrong=3, fresh=200/200, cost=85000
```

```text
task396/experiments/ablate_141_else.onnx
incumbent cost: 27662
near miss input: public_wrong=0, arc_gen_wrong=3, fresh=199/200, cost=27441

repair_internal_missing_bottom_row_0_scalar_0:
  family: internal_selector_extent
  class: missing_bottom_row
  internal_or_dense: internal
  source_mutation_family: where_ablation_inferred
  result: public_wrong=4, arc_gen_wrong=260, fresh=163/200, cost=27444
  accepted: false
  blocker: scalar +1 extent is too broad and activates pixels outside true output shapes

symbolic/dense fallbacks:
  marked primary_repair=false
  high-cost symbolic bottom-row repairs cost 84667
  dense channel-0 fallback repairs report full_output_dense_initializer
```

Focused optimizer probe:

```text
python3.12 .chatgpt/operations/neurogolf_optimizer/run_one.py \
  --task task014 \
  --families dimension_sweep \
  --trials 50 \
  --fresh-min 0.90 \
  --dry-run \
  --keep-rejected \
  --run-id internal_selector_repair_probe \
  --timeout 1200

interrupted after roughly 9 minutes for runtime
state at interruption: 46 trials, 0 accepted, 13 near misses
```

Accepted gains:

```text
none
projected score remains 7157.530543944447
```

Why this moves toward 7900+:

- The repair system now repairs task014's known missing final column internally and exactly, without a dense final-output patch.
- The current blocker is cost, not correctness, for task014.
- The task396 probe shows the first scalar extent restoration is too broad, giving the next concrete selector-narrowing target.
- High-cost final-output repairs are no longer treated as primary score candidates.

## 2026-06-27: Internal repair gate/provenance/reporting correction

- Made the repair cost gate actionable.
  - Equal-cost or worse estimated repairs are now non-primary with `estimated_cost_not_below_incumbent`.
  - Non-internal fallback repairs remain non-primary with `fallback_not_primary_internal_repair`.
  - Full-output dense initializers remain non-primary with `full_output_dense_initializer`.
  - Non-primary repairs skip pair diagnostics and official validation unless `--keep-high-cost-probes` is set.
- Passed source mutation provenance from optimizer near-miss repair into `try_repair_model`.
  - Repair rows now include `source_mutations`, `source_mutation_families`, and `source_mutation_ids`.
  - Internal repair prefers actual mutation metadata when present; standalone near-miss lab can still use inferred provenance.
- Fixed cached validation reporting.
  - Cached `ValidationResult` objects now report the current requested `model_path` without changing the validation cache key or stored cache entry.

Validation:

```text
PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer
passed

PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
Ran 66 tests in 27.731s
OK
```

Sentinel probes:

```text
task014 normal repair probe
run_dir: task014/experiments/near_miss_lab/20260627_234111

repair_internal_missing_last_column_0_dimension:
  family: internal_selector_extent
  class: missing_last_column
  estimated_cost: 7983
  incumbent_cost: 7983
  primary_repair: false
  skipped: true
  skip_reason: estimated_cost_not_below_incumbent
  validation: null
  accepted: false

fallback symbolic/dense repairs:
  skipped as non-primary unless cleanup
  dense/full-output repairs report full_output_dense_initializer where applicable
```

```text
task014 with --keep-high-cost-probes
run_dir: task014/experiments/near_miss_lab/20260627_234140

repair_internal_missing_last_column_0_dimension:
  primary_repair: false
  cost_gate: estimated_cost_not_below_incumbent
  validation ran
  public_wrong: 0
  arc_gen_wrong: 0
  fresh: 200/200
  cost: 7983
  accepted: false
  blocker: equal to incumbent, not below it
```

```text
task396 normal repair probe
run_dir: task396/experiments/near_miss_lab/20260627_234235

near miss input:
  public_wrong: 0
  arc_gen_wrong: 3
  fresh: 199/200
  cost: 27441
  incumbent_cost: 27662

internal missing_bottom_row scalar repairs:
  primary_repair: true
  cost: 27444
  public_wrong: 4
  arc_gen_wrong: 260
  fresh: 163/200
  accepted: false
  blocker: scalar extent restoration is still too broad

symbolic/dense fallbacks:
  skipped as non-primary
  high-cost/full-output reasons include fallback_not_primary_internal_repair,
  estimated_cost_not_below_incumbent, estimated_cost_far_above_incumbent,
  and full_output_dense_initializer where applicable
```

Accepted gains:

```text
none
projected score remains 7157.530543944447
```

Remaining blocker:

- The cost/reporting behavior is now correct, but the internal repair family still needs a lower-cost exact task014 variant or a narrower task396 selector repair before it changes projected score.

## 2026-06-28: Feature 1B cost-positive internal repair completion

- Added a gated expected-extent internal repair variant for selector/extent near misses.
  - For missing bottom-row/row failures, the repair can replace a broad scalar `+1` with `Where(selector, Max(old_extent, expected_extent), old_extent)`.
  - The selector comes from source `where_ablation`/`where_combo` provenance when available, with a local scalar-Where inference fallback for standalone near-miss lab runs.
  - This is reusable selector restoration, not a dense final-output patch.
- Added bounded post-repair cleanup chains for internal repairs.
  - Variants record `repair_chain`, `post_repair_cleanup_steps`, `cost_before_cleanup`, `cost_after_cleanup`, and `accepted_after_cleanup`.
  - Cleanup is limited to prune/unreachable cleanup, unused initializer/value-info removal, and initializer dedup.
- Added compact repair cost diagnostics.
  - Repair rows now report source-vs-candidate params, memory, cost delta, largest initializer/intermediate contributors, full-volume intermediate count, and unused initializers.
  - Broad internal selector failures are labeled with `selector_too_broad` when pair diagnostics show extra active black/outside-shape behavior.
- Carried repair-chain and cost-diagnostic metadata through optimizer-created near-miss repair candidates.

Validation:

```text
PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer
passed

PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
Ran 72 tests in 25.968s
OK
```

Sentinel probes:

```text
task014 normal repair probe
run_dir: task014/experiments/near_miss_lab/20260628_002131

near miss input:
  public_wrong: 0
  arc_gen_wrong: 1
  fresh: 200/200
  cost: 7905
  incumbent_cost: 7983

repair_internal_missing_last_column_0_dimension:
  exact repair remains cost-gated
  estimated_cost: 7983
  primary_repair: false
  skipped: true
  skip_reason: estimated_cost_not_below_incumbent
  cost delta explanation: +78 total vs near miss
    params: +1
    memory: +77
    main contributors: 17x17 -> 17x18 class/inside/rect/shifted tensors

cleanup variants:
  repair_internal_missing_last_column_0_dimension_cleanup_prune
  repair_internal_missing_last_column_0_dimension_cleanup_dedup
  result: both remain estimated_cost 7983 and are skipped as equal-cost
```

```text
task396 normal repair probe
run_dir: task396/experiments/near_miss_lab/20260628_002204

near miss input:
  public_wrong: 0
  arc_gen_wrong: 3
  fresh: 199/200
  cost: 27441
  incumbent_cost: 27662

repair_internal_missing_bottom_row_0_gated_extent_0_gated_expected_0:
  family: internal_selector_extent
  strategy: gated_expected_extent
  accepted: true
  public_wrong: 0
  arc_gen_wrong: 0
  fresh: 200/200
  cost: 27446
  delta_points: 0.007839192355074687
  cost delta vs near miss: +5
    params: +1
    memory: +4
    nodes: +2

older broad scalar repair:
  cost: 27444
  public_wrong: 4
  arc_gen_wrong: 260
  fresh: 163/200
  labels: selector_too_broad, extra_active_black_outside_shape
```

Promotion:

```text
python3.12 .chatgpt/operations/neurogolf_optimizer/near_miss_lab.py \
  --task task396 \
  --model task396/experiments/ablate_141_else.onnx \
  --fresh-min 0.90 \
  --repair \
  --promote \
  --keep-rejected \
  --timeout 1200

promoted through existing near_miss_lab promotion path:
  repair_internal_missing_bottom_row_0_gated_extent_0_gated_expected_0
  task396/experiments/candidate.onnx now validates:
    public_wrong: 0
    arc_gen_wrong: 0
    fresh: 200/200
    cost: 27446
```

Accepted gains:

```text
task396: 27662 -> 27446
delta_points: 0.007839192355074687
```

Projected score:

```text
7157.530543944447 -> 7157.538383136803
```

Remaining blockers:

- task014 is still correctness-solved but not cost-positive; its exact internal repair costs +78 over the near miss and equals incumbent.
- The projected score moved, but the gain is still tiny relative to the 7900+ target; more near-miss conversions are needed.
