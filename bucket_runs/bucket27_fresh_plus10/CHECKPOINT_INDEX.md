# Checkpoint index

This branch preserves all current bucket27 fresh-+10 work, including candidates below the later approximately +1 promotion threshold.

## Exact experimental candidates

- `task131/`: exact dynamic true-size single-canvas rewrite; 3894 -> 2679 cost; +0.3739933164; public 4/4, ARC-GEN 262/262, fresh 1000/1000. The ONNX model, deterministic builder, and validation report are present.
- `task117/`: exact canonical-quadrant reflect rewrite; 3572 -> 3401 cost; +0.0490561570; public 4/4, ARC-GEN 261/261, fresh 1000/1000. The deterministic builder, validation report, and exact ONNX bytes are present. The ONNX bytes are stored as `model.onnx.b64`; decode with `base64 -d model.onnx.b64 > model.onnx`.

## Research and failed paths

- `task004/RESEARCH.md`: exact local parity selector, stress-tested on 100,000 fresh cases.
- `task277/RESEARCH.md`: rejected heuristic classifiers and the remaining packed-connectivity direction.
- `task284/`: rejected int32 ScatterND builder and runtime explanation.
- `task361/RESEARCH.md`: compact center/rotation-state target.
- `RESUME_STATUS.md`: official accounting, promotion policy, operator findings, and next-target ranking.

## Promotion state

Official fresh promoted gain remains +0.000000 because the user later required roughly +1 gain per promotion. The active bucket was restored to the exact uploaded active models. Historical manifest gains are not counted.

## Resume procedure

1. Check out branch `agent/bucket27-fresh-plus10`.
2. Read `RESUME_STATUS.md` and this index.
3. Recreate task117's ONNX file if needed:

```bash
base64 -d bucket_runs/bucket27_fresh_plus10/task117/model.onnx.b64 \
  > bucket_runs/bucket27_fresh_plus10/task117/model.onnx
```

4. Continue only with approximately +1-or-better promotion candidates against the uploaded active models.
