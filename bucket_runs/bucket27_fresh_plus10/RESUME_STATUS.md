# Bucket27 fresh +10 resume checkpoint

## Promotion policy

The uploaded active models are the only baseline for this campaign.
Do not count historical manifest gain.
Only replace an incumbent when the new candidate gains approximately +1 point or more. Smaller exact candidates are research checkpoints only.

## Official active state

- Official fresh promoted gain: +0.000000
- Required fresh gain: +10.000000
- Active bucket models: restored to the exact uploaded baseline set
- `manifest.json`: not updated with experimental candidates

Uploaded active-model total: 175.57157329418018 points.
Required target: at least 185.57157329418018 points.

## Exact experimental checkpoints

### task131

Fresh structural rewrite: dynamic true-size single-canvas translation and direct label rendering.

- baseline cost: 3894
- candidate cost: 2679
- gain: +0.373993316387536
- public: 4/4, wrong 0
- bundled ARC-GEN: 262/262, wrong 0
- fresh: 1000/1000
- exact: yes
- promotion: no, below current approximately +1 threshold

### task117

Fresh structural rewrite: one canonical 7x7 source quadrant, body-cell clearing, and one reflect pad.

- baseline cost: 3572
- candidate cost: 3401
- gain: +0.0490561569891947
- public: 4/4, wrong 0
- bundled ARC-GEN: 261/261, wrong 0
- fresh: 1000/1000
- exact: yes
- promotion: no, below current approximately +1 threshold

## High-value next targets

1. task004: exact local parity rule is proven; need a terminal factorization below roughly 553 cost.
2. task131: eliminate padded scalar-label intermediate and use a terminal channel renderer; need below roughly 1433 cost.
3. task361: compact closure/rotation state plus terminal renderer; need roughly 931 cost or lower.
4. task277: exact packed bounded connectivity; need roughly 1045 cost or lower.
5. task284: coordinate-factor renderer with exact background handling; need roughly 1136 cost or lower.

## Operator/runtime findings

- Graph output memory is free.
- Dynamic runtime shapes can be used with scorer-valid positive placeholder dimensions; profiler-observed runtime sizes matter.
- Sparse initializers are not a usable optimization path under the current scorer.
- ORT CPU rejects int32 ScatterND indices in this environment.
- MaxUnpool flattened indices can address the full requested output tensor.
- Sparse foreground rendering still needs an exact background-channel strategy.
- TfIdfVectorizer is scorer-valid but duplicate n-grams are rejected by ORT, so it is not a universal free dense lookup.
- ai.onnx.ml LinearClassifier was identified as a potentially important attribute-stored transform, but rank/output behavior probing was not completed.

## Environment

- onnx==1.21.0
- onnxruntime==1.24.4
- numpy==2.4.4
