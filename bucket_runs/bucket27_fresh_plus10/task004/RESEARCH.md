# Task004 research checkpoint

## Exact local rule

For each non-background input pixel, the output chooses either the current pixel or its immediate left neighbor.
The exact selector is:

```text
choose_left <=> below_same_column == XOR(left_occupied, current_occupied)
```

This rule was stress-tested against 100,000 fresh generator samples with no observed failures.

Equivalent multilinear parity factorization:

```text
0.5 * (current + left)
+ 0.5 * (left - current)
  * (1 - 2L)
  * (1 - 2C)
  * (1 - 2B)
```

where L, C, and B are binary occupancies of the left, current, and below-same-column positions.

## Current blocker

The obvious ONNX implementations spend too much intermediate memory. A qualifying promotion needs cost near or below 553 to approach +1 point from the uploaded active baseline cost 1503. Continue searching for a terminal-output-only Conv/Einsum factorization rather than a multi-stage Boolean graph.
