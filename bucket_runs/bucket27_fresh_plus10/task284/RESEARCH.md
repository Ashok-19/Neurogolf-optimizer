# Task284 research checkpoint

## Attempted int32 ScatterND rewrite

A fresh deterministic endpoint/geometry builder was created using int32 sparse indices to reduce the incumbent's large int64 index table.

Result: rejected by ONNX Runtime 1.24.4 CPU because ScatterND indices are not accepted as int32 by the runtime kernel used here, despite the broader ONNX schema.

## MaxUnpool idea

Flattened sparse foreground indices can be compact, but foreground-only rendering is insufficient because the output requires a positive background channel on every untouched cell. The background carrier removes the apparent gain.

Uploaded active baseline cost: 3089.
Approximate +1 target: <= 1136.
