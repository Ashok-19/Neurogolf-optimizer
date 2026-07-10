# Task330 dual-threshold exactness proof

For each candidate shape anchor, the score convolution computes `s = interior_count + 10 * boundary_count`.
A valid six-cell isolated component has `s = 6`. Any missing interior cell has `s < 6`; any extra interior cell or occupied four-neighbor boundary has `s >= 7`.

The first quantized threshold is exactly `A = 255 * [s >= 6]`; the second is exactly `B = 255 * [s >= 7]`. Therefore `A - B = 255` exactly when `s = 6`, and zero otherwise. The cover convolution spreads only this difference over the matching shape footprint. Saturation makes overlapping covers idempotent.

The final signed renderer receives occupancy `O` and cover `C`, each in `{0,255}`:
- black logit: `128 - O`, positive iff inactive;
- blue logit: `O - C`, positive iff active and uncovered;
- red logit: `O + C - 256`, positive iff active and covered.

Thus the rewrite is structurally equivalent to the rule for every supported generated shape, not only the sampled fresh cases.
