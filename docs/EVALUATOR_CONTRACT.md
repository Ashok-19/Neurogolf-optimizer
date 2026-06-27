# Evaluator Contract

This file defines what the optimizer must optimize against. Future agents should treat this as the highest-priority technical contract inside the optimizer.

## Official Acceptance Rule

A candidate can be accepted only if all of these are true:

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

Promotion must revalidate the candidate immediately before copying it to `taskXXX/experiments/candidate.onnx`.

## Public/ARC Comparison

The official validator compares ONNX output tensors exactly:

```python
np.array_equal(user_output, benchmark["output"])
```

For public and ARC examples, relaxed grid comparison is wrong. It can hide official failures.

The output tensor is a one-hot-like tensor with shape usually:

```text
[1, 10, 30, 30]
```

Important distinction:

```text
inside expected output grid: color 0 is real black and must activate channel 0
outside expected output grid: no color is represented by all 10 channels equal to zero
```

Active channel 0 outside the expected output shape is an official tensor mismatch. It is not valid padding.

## Fresh Comparison

Fresh examples still use grid conversion plus missing-zero comparison. This is why a candidate can pass fresh but fail public/ARC.

Use:

```text
official_public_arc
```

for public/ARC diagnostics.

Use:

```text
fresh_grid
```

only for fresh diagnostics.

Use:

```text
both
```

only when intentionally comparing both behaviors for debugging.

## `pair_runner.py`

Required public/ARC command:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/pair_runner.py \
  --task task031 \
  --model path/to/model.onnx \
  --sources public,arc \
  --comparison official_public_arc \
  --json
```

For official tensor mismatches it reports:

```text
active wrong channels
missing active channels
extra active channel 0 outside expected output shape
too-many-colors cells
no-color cells
first failing coordinates
expected tensor active channels
predicted tensor active channels
```

This output is used by the near-miss repair system. Do not reduce it to a pass/fail boolean.

## Cost and Static Shape Contract

Official scorer compatibility requires static enough ONNX graphs. A candidate with:

```text
cost = null
memory = null
```

is not useful and must not be queued as an optimization candidate.

Preflight should include:

```text
onnx.checker.check_model
onnx.shape_inference.infer_shapes(strict_mode=True)
all node outputs have statically inferable shapes when possible
no dynamic dimensions in scorer-critical paths
```

The optimizer currently uses `static_shape_preflight` for synthesis. Any new compiler backend must keep this behavior.

## Promotion and Retention

Promotion is allowed only when the official validator accepts a lower-cost model.

Rejected models should be kept when they are diagnostic:

```text
synth Python public+ARC exact
public exact
ARC exact
near-miss category A/B/C/D
positive score gain but official failures
fresh >= 0.90
model has cost lower than incumbent
```

Logs must include:

```json
{
  "model_path": "...",
  "model_file_exists": true,
  "retention_reason": "..."
}
```

Do not leave logs pointing to deleted diagnostic models.

## Known Evaluator Bug That Was Fixed

Older `pair_runner.py` used relaxed grid comparison for public/ARC. That caused false confidence:

```text
task031 crop candidate:
Python relaxed score = 1.0
official public_wrong = 4
official arc_gen_wrong = 262
```

Root cause:

```text
outside real output shape: active black channel 0
official expected: no active channel
```

Fixed proof:

```text
task031_fix_probe candidate:
pair_runner official_public_arc = 266/266
official wrapper: public=0, arc=0, fresh=200/200, cost=685640
```

This candidate is not promoted because it is much more expensive than the incumbent.

