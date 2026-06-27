# ONNX Cost Engineering Notes

Use official ONNX and ONNX Runtime docs as source material for operator semantics and dtype support:

```text
ONNX docs: https://onnx.ai/onnx/intro/index.html
ONNX operators: https://onnx.ai/onnx/operators/
ONNX shape inference: https://onnx.ai/onnx/api/shape_inference.html
ONNX Runtime docs: https://onnxruntime.ai/docs/
```

Do not rely on memory of operator support. Check docs and verify locally with ONNX Runtime.


## Scorer Facts

Current scoring behavior:

```text
points = max(1, 25 - ln(memory + params))
cost = memory + params
```

Practical facts:

```text
MACs are effectively free.
Graph inputs and outputs are excluded from memory cost.
Intermediates and parameters dominate.
Declared/static tensor shapes matter.
Stale value_info can inflate cost.
Official validator/scorer is final.
```

This means a graph can be small on disk but expensive in score if it declares large intermediates.

## Design Bias

Prefer:

```text
scalar state
low-rank tensors
bool masks when operators support them
uint8 labels
small int indexes
Reduce/ArgMax/Gather/Where over dense lookup tables
static shape masked output over dynamic crop if scorer cannot infer shape
```

Avoid:

```text
large one-hot [1,10,30,30] intermediates
large broadcast chains
full 30x30 correction masks unless cost still wins
float32 state when bool/uint8/int32 works
dynamic shapes in scorer-critical paths
Constant nodes with large payloads as "free" replacements
```

## Operator and Dtype Traps

### Pad

`Pad` can replace zero-padding `Concat` for supported dtypes and static shapes.

Observed trap:

```text
ONNX Runtime rejects Pad on bool tensors in these task kits.
```

Therefore `concat_pad` skips bool zero-pad rewrites.

Observed successful case:

```text
uint8 vector concat zero-tail -> Pad
task233 accepted cost improvement
```

### Constant Nodes

Moving an initializer payload into a `Constant` node is not free.

Observed:

```text
task066 all-false bool mask initializers -> Constant nodes
official cost increased from 20081 to 20581
```

The scorer still charged payload/memory. Do not assume Constant nodes reduce params.

### Dynamic Crop

Dynamic `Slice`/`Pad` can pass ORT but fail scorer cost with:

```text
cost = null
memory = null
```

Prefer static masks and fixed shapes.

### One-Hot Output Encoding

Outside expected output shape:

```text
all ten channels must be zero
```

Inside expected output shape:

```text
real black color 0 must activate channel 0
```

This distinction must be preserved by crop, pad, flip, rotate, and synthesized output compilers.

## Cost-Aware Structural Patterns

### Zero-Tail Concat to Pad

Pattern:

```text
Concat(real_tensor, zero_initializer, axis=k)
```

Rewrite:

```text
Pad(real_tensor, pads, scalar_zero_same_dtype)
```

Requirements:

```text
zero initializer is all zeros
dtype is supported by ORT Pad
pads are static int64
pad value dtype matches input dtype
shape inference succeeds
```

### BBox Crop With Static No-Color Mask

Pattern:

```text
compute bbox
gather/crop area
pad outside area
```

Safe output:

```text
valid_area_mask = rows < output_h and cols < output_w
output = Where(valid_area_mask, cropped_value, no_color_zero_tensor)
```

Do not use active black padding outside expected output shape.

### Dimension Sweep

Pattern:

```text
arange index tensor drives row/col work dimension
Pad restores to 30x30
```

Rewrite:

```text
reduce arange length
adjust row/col shape initializers
adjust pad_to_30
```

This can generate high-value near misses. It needs symbolic repair to be broadly accepted.

### Selector Branch Optimization

Pattern:

```text
Where(condition, expensive_branch, cheap_branch)
```

Naive branch ablation often produces near misses. Future repair should restore only the missing scalar selector or sparse branch fragment, not the full removed branch.

## Static Shape Preflight Checklist

Before queuing a candidate:

```text
onnx.checker.check_model(model)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
all relevant value_info dimensions are concrete
output shape is official-compatible
official validator returns non-null cost/memory/params
```

If any step fails, do not count it as a useful trial.

