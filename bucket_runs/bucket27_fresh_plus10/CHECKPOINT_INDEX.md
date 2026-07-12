# Checkpoint index

This branch preserves all current bucket27 fresh-+10 work, including candidates below the later approximately +1 promotion threshold.

## Exact experimental candidates

- `task131/`: exact dynamic true-size single-canvas rewrite; 3894 -> 2679 cost; +0.3739933164; public 4/4, ARC-GEN 262/262, fresh 1000/1000. The ONNX model, deterministic builder, and validation report are present.
- `task117/`: exact canonical-quadrant reflect rewrite; 3572 -> 3401 cost; +0.0490561570; public 4/4, ARC-GEN 261/261, fresh 1000/1000. Builder and validation report are present; the compact resume archive contains the ONNX binary.

## Research and failed paths

- `task004/RESEARCH.md`: exact local parity selector, stress-tested on 100,000 fresh cases.
- `task277/RESEARCH.md`: rejected heuristic classifiers and the remaining packed-connectivity direction.
- `task284/`: rejected int32 ScatterND builder and runtime explanation.
- `task361/RESEARCH.md`: compact center/rotation-state target.

## Promotion state

Official fresh promoted gain remains +0.000000 because the user later required roughly +1 gain per promotion. The active bucket was restored to the exact uploaded active models. Historical manifest gains are not counted.

## Resume bundle

`bucket27_resume_compact.zip.b64` is a base64-encoded ZIP containing the exact experimental ONNX files, builders, iteration scripts, validation JSON, research notes, and status documents. Decode with:

```bash
base64 -d bucket27_resume_compact.zip.b64 > bucket27_resume_compact.zip
unzip bucket27_resume_compact.zip
```

Expected ZIP SHA-256:

```text
a33f58c4df6cc1ceb7679a02c04259983df0cb3927641b540bd4ab15bcea8441
```
