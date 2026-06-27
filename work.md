# NeuroGolf Optimizer Worklog

## 2026-06-27: Cross-task pattern miner scaffold

- Added `pattern_miner.py` to scan all selected task incumbents, prefer accepted `experiments/candidate.onnx` only when optimizer accept notes exist, aggregate graph-pattern hits, and write ranked JSON/Markdown reports.
- Added `broadcast_scalarize` and `where_simplify` mutation families with conservative shape guards. The final all-task miner report shows no current safe live hits for these families after guard tightening.
- Updated docs for the miner entrypoint and new rewrite families.
- Validation:
  - `PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer`
  - `PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q` -> 51 tests OK
- Pattern miner proof:
  - `pattern_miner_probe4`: 400 tasks scanned, 63 opportunity classes, 0 errors.
  - Top classes: Where/selector rewrites, dtype lowering, initializer/value-info refresh, bool-mask rewrites, scalar-label lowering, arange work dimensions.
- Campaign proof:
  - `miner_top_family_probe` dry-run on `task009,task074,task054` with `dtype_lowering,scalar_label_rewrite`.
  - 3 tasks completed, 0 errors, 0 accepted, 0 near misses, total accepted delta 0.
- Readiness:
  - Projected score remains 7157.530543944447.
  - This is not 7900+ ready.

## 2026-06-27: Cache profile and symbolic near-miss repair patch

- Fixed `ValidationCache` to key entries by `(task, model_hash, validation_profile)` instead of only `(task, model_hash)`.
  - Profile includes validator mode, quick/full command mode, quick flag, and timeout.
  - Existing SQLite cache rows are migrated to a `legacy` profile instead of being reused accidentally.
- Improved Where/selector mutation shape safety.
  - `where_ablation` now uses non-strict ONNX shape inference during discovery.
  - Broadcast branch replacements carry `selected_shape`, `output_shape`, and insert static `Expand` when needed.
  - task018 selector probe moved from shape-inference build failures to valid rejected candidates; no accepted gain.
- Added early symbolic near-miss repair.
  - Classifies missing row/column line failures from `pair_runner` first failing coordinates.
  - Emits row/column/channel mask patches with an adjacent-extent selector before dense repairs.
  - Emits opset-aware `ReduceSum` for opset 12 versus opset 13+ models.
- Validation:
  - `PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer`
  - `PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q` -> 55 tests OK
- Focused proof:
  - `task014/experiments/attempt2.onnx`: baseline near miss remains public_wrong=0, arc_gen_wrong=1, fresh=200/200, cost=7905.
  - `repair_symbolic_missing_last_column_0` builds and validates but is not accepted: public_wrong=0, arc_gen_wrong=3, fresh=200/200, cost=85000.
  - This proves the symbolic repair path works mechanically, but selector logic and cost are still insufficient.
- Readiness:
  - Accepted gain unchanged.
  - Projected score remains 7157.530543944447.
  - This is not 7900+ ready.
