# Bucket 34 resume checkpoint

Status: paused on 2026-07-12.

- Baseline total: 183.037962577793
- Current verified total: 185.034727470382
- Verified gain: +1.996764892590
- Target: +10
- Remaining: 8.003235107410

## Active exact promotions

| Task | Cost | Gain | Validation |
|---|---:|---:|---|
| 143 | 495 | +0.337914 | public 0 wrong, ARC-GEN 0 wrong, fresh 1000/1000 |
| 214 | 343 | +0.425668 | public 0 wrong, ARC-GEN 0 wrong, fresh 200/200 |
| 215 | 190 | +0.633509 | public 0 wrong, ARC-GEN 0 wrong, fresh 1000/1000 |
| 224 | 1175 | +0.343741 | public 0 wrong, ARC-GEN 0 wrong, fresh 500/500 |
| 225 | 792 | +0.255933 | public 0 wrong, ARC-GEN 0 wrong, fresh 200/200 |

## Resume policy

Only promote new candidates with about +1 point or close to +1. Prefer multi-point structural rewrites. Preserve the current exact models unless a stronger candidate passes the same or stronger validation.

## Important unfinished experiment

`tasks/task215/work/candidate_periodic_scaled.onnx` is unvalidated. Its builder is preserved, but it must not be promoted without the full gates.
