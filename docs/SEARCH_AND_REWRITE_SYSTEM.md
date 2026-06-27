# Search and Rewrite System

The optimizer must evolve from local graph cleanup into a structural rewrite search engine. This file describes the current system and how future agents should extend it.

## Mutation Family Interface

Every graph rewrite family implements:

```python
class BaseMutationFamily:
    name = "family_name"

    def discover(self, task: TaskContext, model: onnx.ModelProto) -> list[Mutation]:
        ...

    def apply(self, task: TaskContext, model: onnx.ModelProto, mutation: Mutation) -> BuildResult:
        ...
```

Discovery should be cheap and static. Application may clone and modify the graph, then must return through `self.build(...)`, which prunes and safe-builds.

## Mutation Requirements

A mutation should include:

```text
id: stable mutation id
family: family name
description: concise human/agent-readable description
priority: higher means try earlier
estimated_savings: rough cost reduction estimate
metadata: enough to reapply exactly
```

Never depend on node index alone when a stable name or structural signature is available. Node indexes are acceptable only for simple local rewrites and should be validated during `apply`.

## Current Mutation Families

```text
initializer_prune
  Removes unused initializers.

initializer_dedup
  Reuses duplicate initializer values.

identity_cleanup
  Removes Identity and trivial pass-through structure.

cast_simplify
  Removes or simplifies redundant casts when safe.

constant_rewrite
  Rewrites Constant nodes to cheaper forms when possible.

where_simplify
  Removes provably redundant Where nodes when branches are identical or the condition is a static all-true/all-false tensor.

where_ablation
  Replaces a Where with one branch and prunes dead graph.

where_combo
  Applies combinations of Where branch choices.

output_collapse
  Attempts final output simplification.

gather_shrink
  Shrinks gathered tables and remaps indices.

table_shrink
  Table-shrink wrapper around gather/table opportunities.

dimension_sweep
  Reduces inferred row/column work dimensions for arange-driven crop/output graphs.

dtype_lowering
  Lowers initializer dtype when consumer operators allow it.

broadcast_scalarize
  Replaces large uniform initializers with scalar initializers when every consumer supports ONNX multidirectional broadcasting and a non-scalar shape anchor remains.

concat_pad
  Replaces zero-padding Concats with Pad when ONNX Runtime accepts the dtype.

bool_mask_rewrite
  Simplifies bool masks and bool constants.

scalar_label_rewrite
  Replaces label-like tensors with cheaper scalar labels when possible.

synth_solver
  Queues compiler-backed DSL programs.

conv_rewrite
  Rewrites selected convolution patterns.
```

## Recently Fixed Generic Rewrite

`concat_pad` had the correct structural idea but emitted a float32 pad value for all dtypes. That failed or changed cost behavior for uint8/int64 tensors.

It now:

```text
uses the zero initializer dtype for Pad value
skips bool Pad rewrites because ORT rejects bool Pad in these task kits
```

Proof of generic value:

```text
task233: two uint8 zero-tail Concats -> Pad
official accepted: public=0, arc=0, fresh=199/200
cost: 71718 -> 71679
```

This should be treated as an optimizer capability, not a task-specific trick.

## Queue Construction

The optimizer builds candidate mutation sets from:

```text
single mutations
beam combinations
evolutionary sampled combinations
```

After de-duplication, queue items are sorted by:

```text
sum(estimated_savings)
sum(priority)
shorter chain length
```

This is only a heuristic. It should eventually be replaced or augmented by a learned or measured cost model that predicts:

```text
official exactness probability
fresh generalization probability
static scorer compatibility
cost reduction
repairability if near-miss
```

## Useful Trial Rules

A useful trial is a candidate that was built, retained long enough for diagnostics, and validated by the official wrapper.

Do not count:

```text
duplicate model hashes
known failed build signatures
missing/deleted candidates
py_score=0 synthesis candidates
known official encoding mismatch candidates
dynamic-shape candidates likely to score cost=null
```

Target for short campaigns:

```text
useful_trial_rate >= 0.50
```

If this is not achieved, report exact blockers.

## How to Add a New Rewrite Family

1. Identify a cross-task graph pattern, not a single-task patch.
2. Write a discovery pass that records enough metadata for deterministic application.
3. Estimate cost reduction conservatively.
4. Apply the rewrite on a cloned graph.
5. Remove stale value_info or let safe_build do it.
6. Run ONNX checker and strict shape inference through safe_build.
7. Add unit tests with a synthetic model.
8. Run at least one real-task probe only to prove the family fires.
9. Never promote unless official validator accepts.

## High-Value Rewrite Families Still Missing

These are needed for 7900+ scale:

```text
Rule-skeleton replacement
  Replace large exported graphs with compact hand-compiled rule templates.

Final-output symbolic composer
  Build output from detected anchors, spans, colors, and masks without full one-hot maps.

Selector branch restoration
  For near misses caused by branch ablation, restore only the missing selector logic, not full deleted branch cost.

Sparse correction compiler
  Express small cell corrections as cheap symbolic row/col/color conditions instead of dense 30x30 masks.

Dynamic-to-static shape compiler
  Convert dynamic crop/bbox logic into static masked graphs that official scorer can cost.

Operator substitution miner
  Search ONNX/ORT-supported operator alternatives for equivalent behavior with lower declared intermediates.

Graph macro miner
  Detect repeated subgraphs across tasks and replace with cheaper canonical macros.

Cost-model-guided mutation ordering
  Learn from prior validations which rewrites are worth official validation.
```

## Anti-Patterns

Do not spend campaigns on:

```text
random branch ablations without repair
cleanup-only near-miss handling
dynamic Slice/Pad crops that score cost=null
public-only constant outputs
large dense correction masks that exceed incumbent cost
one-hot full-volume intermediates when scalar/grid state is enough
```
