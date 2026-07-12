# Task277 research checkpoint

## Rejected approaches

- large-window population score
- local count classifiers
- exclusion-rule variants
- duplicate-component autocorrelation heuristics without exact connectivity

These reached roughly 83%-95% depending on the variant, but generated cases contain conflicting local signatures. They must not be promoted.

## Exact remaining direction

The generator components are bounded small objects. The promising exact rewrite is compact/packed bounded connectivity or row-bitmask component state, followed by component-width comparison and one final renderer.

Uploaded active baseline cost: 2841.
Approximate +1 target: <= 1045.
