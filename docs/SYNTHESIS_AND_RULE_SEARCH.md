# Synthesis and Rule Search

Synthesis is the route from graph mutation toward true structural rewrites. The optimizer cannot reach 7900+ by pruning alone.

## Current Synthesis Components

```text
synthesis/dsl.py
  Program and Step representation.

synthesis/families.py
  Primitive candidate program generation.

synthesis/program_search.py
  Program search, official-compatible Python verification, and waterfall integration.

synthesis/verifier.py
  Python verifier with grid_relaxed and official_tensor modes.

synthesis/onnx_emit.py
  Compiler from DSL programs to ONNX.

synthesis/onnx_templates.py
  Static ONNX templates and shape/cost preflight helpers.

synthesis/waterfall.py
  Multi-step composition search over supported primitive steps.

mutations/synth_solver.py
  Mutation family that queues compiler-backed synthesis candidates.
```

## Verification Modes

Python verification supports:

```text
grid_relaxed
official_tensor
```

For programs intended to compile to ONNX and be validated on public/ARC, use:

```text
official_tensor
```

The program output grid is converted to the official one-hot/no-color tensor semantics before comparison. This catches no-color versus black padding errors before ONNX validation.

Fresh may use:

```text
grid_relaxed
```

because fresh validation uses grid conversion and missing-zero comparison.

## Ranking Contract

Program ranking must prioritize:

```text
official public tensor exact
official ARC tensor exact
fresh grid rate
estimated cost
complexity
```

Do not label a candidate `python_public_arc_exact` unless it is exact under official tensor encoding.

## Waterfall Search

`waterfall.py` composes primitive DSL steps into multi-step programs.

Supported linear steps currently include:

```text
Recolor
ReplaceColor
Crop
CropNonZeroBBox
CropColorBBox
Flip
Rotate
SelectColor
DrawRectangle
FillRectangle
DrawLine
Tile
```

It keeps imperfect intermediates because later steps can repair them.

Current limitations:

```text
It is still mostly enumerative.
It does not infer decomposed subgoals from failures.
It has weak cost modeling.
It has limited compiler coverage.
It does not yet synthesize rich conditional branch logic.
```

## ONNX Compiler Rules

Compiler outputs must:

```text
pass onnx.checker.check_model
pass strict shape inference
avoid dynamic scorer-critical shapes
avoid cost=null candidates
represent outside-output no-color as all-zero channels
preserve real black inside output grid as active channel 0
```

The crop/bbox compiler bug was fixed by emitting no-color outside valid output masks rather than active black.

## Known Synthesis Proofs

Task031 crop candidate:

```text
program: crop_nonzero_bbox
strict pair_runner public/ARC: 266/266
official wrapper: public=0, arc=0, fresh=200/200
cost=685640
not promoted because incumbent is far cheaper
```

Task150/task155 flip/rotate compiler:

```text
strict pair_runner public/ARC: 266/266
official wrapper: public=0, arc=0, fresh=200/200
not promoted because incumbent is tiny
```

These proofs show evaluator compatibility. They do not show cost competitiveness.

## Required Next Synthesis Upgrades

### Rule Decomposition

The synthesizer must infer subproblems:

```text
object/color selection
anchor detection
bbox/span computation
geometry transform
pattern completion
line/rectangle/stamp drawing
background/no-color composition
output shape decision
```

Each subproblem should have multiple ONNX representations ranked by cost.

### Conditional Program Search

Most high-value tasks need conditionals:

```text
if object A is left of object B
if dominant color is X
if shape is horizontal/vertical
if bbox width/height crosses a threshold
if marker color exists
```

The DSL needs cheap conditional constructs that compile to `Where`, `Greater`, `Equal`, `Reduce*`, and scalar state, not full dense maps.

### Static Shape Crop Family

Dynamic crops are risky because scorer shape inference may fail. Prefer:

```text
static 30x30 or max-HxW tensor
valid_area_mask
Where(valid_area_mask, value, no_color_zero_tensor)
```

or statically inferred task-family output bounds.

### Cost-Aware Compiler

The compiler should estimate:

```text
intermediate memory
initializer params
dtype cost
one-hot volume penalty
Pad/Slice/Concat declared tensors
```

before queuing the candidate.

### Failure-Guided Synthesis

When `pair_runner` reports failures, synthesis should create new candidates targeted at:

```text
missing row/column
extra outside shape activation
wrong color at sparse coordinates
wrong output shape
missing border/line
wrong flip/rotation branch
```

These should be symbolic repairs where possible, not dense 30x30 masks.

