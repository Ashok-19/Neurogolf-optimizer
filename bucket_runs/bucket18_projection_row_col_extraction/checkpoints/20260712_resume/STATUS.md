# Bucket18 resume checkpoint — 2026-07-12

## Verified strongest models

| Task | Baseline cost | Strongest verified cost | Exact gain | Status |
|---|---:|---:|---:|---|
| 189 | 610 | 579 | +0.052156 | Exact; promoted before stricter threshold |
| 195 | 570 | 409 | +0.331921 | Public/ARC-GEN wrong 0; fresh 1000/1000 |
| 207 | 203 | 168 | +0.189242 | Public/ARC-GEN wrong 0; fresh 1000/1000 |
| 213 | 686 | 550 | +0.220959 | Public/ARC-GEN wrong 0; fresh 1000/1000 |

Verified cumulative gain: approximately **+0.794279**.

## Current promotion policy

For future work, only promote a new task candidate when the task gain is about +1 point or better. Small cost shaves may be retained as research artifacts but should not be pushed as promotions.

## GitHub promoted model commits

- Task189: `5f3aa1b2`
- Task213 stronger v2: `46611f78`
- Task195 stronger direct encoded Kronecker: `94d2554b`
- Task207 stronger direct colored-count: `9819a1f9`

## Exact but rejected major candidates

### Task201

Projection/crop rewrite was exact on public, bundled ARC-GEN, and fresh 1000/1000, but cost increased from 2729 to 2884.

### Task185

Weighted-projection + ROI rewrite was exact but cost increased from 1530 to 1558. Multiple ROI/cell-consensus variants were explored.

## Next high-value targets

1. Task201 — baseline 2729; +1 target below about 1004.
2. Task185 — baseline 1530; +1 target below about 563.
3. Task184 — baseline 1439; +1 target below about 529.
4. Task183 — baseline 582; +1 target below about 214.

Avoid returning to generic projection/ROI variants already tested unless a materially different state representation is found.
