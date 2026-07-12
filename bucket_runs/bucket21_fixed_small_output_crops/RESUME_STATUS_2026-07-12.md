# Bucket21 optimization progress snapshot

Date: 2026-07-12
Bucket: `bucket21_fixed_small_output_crops.zip`
Tasks: 316, 325, 334, 346, 347, 351, 355, 365, 377, 384

## Current promotion policy

Only promote a task model when the verified score gain is at least +1.0, or approximately +0.8 when clearly near the target. Smaller improvements are checkpoints only and must not replace the bucket incumbent.

## Exact validated checkpoints

| Task | Baseline cost | Best checkpoint cost | Exact score gain | Validation | Status |
|---|---:|---:|---:|---|---|
| 346 | 778 | 697 | +0.1099411134 | public wrong 0; bundled ARC-GEN wrong 0; fresh 1000/1000 | exact checkpoint, not promotion |
| 365 | 2928 | 2852 | +0.0262990935 | public wrong 0; bundled ARC-GEN wrong 0; fresh 1000/1000 | exact checkpoint, not promotion |
| 377 | 2739 | 2306 | +0.1720784684 | public wrong 0; bundled ARC-GEN wrong 0; fresh 1000/1000 | exact checkpoint, not promotion |

Exact checkpoint total: **+0.3083186754**.

Task355 has a lower-cost float16 candidate at cost 395 versus 431 (+0.0872223252), but extended random testing exposed rare failures. It is an approximate/rejected experiment and must not be promoted.

## Best exact candidate files

- task346: `tasks/task346/work/candidate_rowmask_regex_crop10.onnx`
  - builder: `build_candidate_rowmask_regex_crop10.py`
  - architecture: crop to guaranteed 10x10 interior, encode selected-color rows as compact fixed-width decimal row masks, use free-attribute `RegexFullMatch` for exact ring recognition, terminal scalar render.
- task365: `tasks/task365/work/candidate_packed_dims.onnx`
  - builder: `build_candidate_packed_dims.py`
  - architecture: exact rectangle candidate selection with packed three-bit dimension tokens and lookup-based contiguous prefix length.
- task377: `tasks/task377/work/candidate_terminal_einsum_topk10.onnx`
  - builder: `build_candidate_terminal_einsum_topk10.py`
  - architecture: compact band/color state with terminal `Einsum` concentric-square renderer; TopK expanded to ten transitions to remove the full 30-position count branch.

## Important rejected branches

- task346 global color-count shortcut: not exact.
- task346 packed-bit alternative: exact but cost 829, worse than regex checkpoint.
- task325 diagonal-only terminal MaxUnpool: invalid because off-diagonal cells inside logical kxk output require explicit color-0 logits.
- task325 no-hole Euler shortcut: rare failures on ring-hole cases.
- task325 exact packed-row/TfIdfVectorizer Euler lookup: promising +1-scale direction, not completed. Target cost must be below about 412 for +1 versus baseline 1118.
- task351/task384 direct logical-sized output: evaluator requires physical 30x30 tensor.
- task351/task384 terminal MaxRoiPool: stretches selected ROI rather than preserving top-left strict crop semantics.
- task365 integer MatMul candidate: exact but cost 3026, worse than incumbent checkpoint.
- task377 boundary-delta representation: exact but cost 2892, worse than terminal-Einsum checkpoint.
- task377 outer-bbox-center shortcut: invalid because nested rectangles can be arbitrarily offset rather than centered.

## Highest-value next directions

1. Task325 exact packed-row Euler/component solver using free-attribute `TfIdfVectorizer` or another table operator. Must keep the complete cost below ~412 to achieve +1.
2. Search scorer catalog for free-attribute operators or hidden-state operators (`RNN`, `TfIdfVectorizer`, regex/string operators) that collapse spatial state to scalars/vectors.
3. Do not spend time shaving task346/365/377 further unless a redesigned architecture estimates at least +0.8 before implementation.
4. Re-read `scorer/onnx_operator_catalog.json` in the original bucket before selecting new operators.

## GitHub checkpoints already pushed

Repository: `Ashok-19/Neurogolf-optimizer`

- `628dac1fde649f67e0964facac7a9d3f54fa796b` — initial bucket21 exact checkpoint validation metadata.
- `70b3f23bebf599a5b5387be6396fb27d0567d60a` — task377 terminal Einsum checkpoint.
- `474cd801cb09538a9d0a24c69dfadc77d6f8dcd1` — task346 exact checkpoint.
- `d707e7ebf409ca0266df9893e875a1dd7a7f60da` — task365 exact checkpoint.
- `f4decc020cc17dc22898a7dd95785f8ee5f209ba` — task377 overwritten with the lower-cost TopK10 terminal-Einsum checkpoint.

No task currently meets the user's +0.8/+1 promotion threshold. The exact candidates above are resume checkpoints only.
