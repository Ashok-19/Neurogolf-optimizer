from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .config import repo_root_from_file
from .synthesis.grid import grid_equal_with_missing_zero, height, width
from .synthesis.onnx_templates import onehot_tensor_from_grid
from .synthesis.task_data import GridPair, load_all_available_pairs, load_arc_gen_pairs, load_public_pairs

ComparisonMode = str


def tensor_to_grid(output: Any) -> list[list[int]]:
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        out: list[list[int]] = []
        for r in range(arr.shape[1]):
            row: list[int] = []
            for c in range(arr.shape[2]):
                cols = [i for i in range(arr.shape[0]) if arr[i, r, c] > 0]
                row.append(cols[0] if len(cols) == 1 else (11 if cols else 10))
            while row and row[-1] == 10:
                row.pop()
            if row:
                out.append(row)
        while out and not out[-1]:
            out.pop()
        return out
    if arr.ndim == 2:
        return arr.astype(int).tolist()
    raise ValueError(f"unsupported output tensor shape {arr.shape}")


def classify_diff(pred: list[list[int]], expected: list[list[int]]) -> dict[str, Any]:
    ph, pw = height(pred), width(pred)
    eh, ew = height(expected), width(expected)
    mismatches = 0
    pred_colors = set()
    expected_colors = set()
    for row in pred:
        pred_colors.update(int(x) for x in row)
    for row in expected:
        expected_colors.update(int(x) for x in row)
    for r in range(max(ph, eh)):
        for c in range(max(pw, ew)):
            pv = pred[r][c] if 0 <= r < ph and 0 <= c < len(pred[r]) else 0
            ev = expected[r][c] if 0 <= r < eh and 0 <= c < len(expected[r]) else 0
            if int(pv) != int(ev):
                mismatches += 1
    if (ph, pw) != (eh, ew):
        kind = "shape_mismatch"
    elif pred_colors != expected_colors:
        kind = "wrong_color_set"
    elif mismatches:
        kind = "wrong_cells"
    else:
        kind = "exact"
    return {
        "kind": kind,
        "pred_shape": [ph, pw],
        "expected_shape": [eh, ew],
        "mismatched_cells": mismatches,
        "pred_colors": sorted(pred_colors),
        "expected_colors": sorted(expected_colors),
    }


def _active_channels(tensor: np.ndarray, r: int, c: int) -> list[int]:
    if tensor.ndim != 4 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        return []
    if r < 0 or c < 0 or r >= tensor.shape[2] or c >= tensor.shape[3]:
        return []
    return [int(ch) for ch in range(tensor.shape[1]) if tensor[0, ch, r, c] > 0]


def _active_channel_counts(tensor: np.ndarray) -> dict[str, int]:
    if tensor.ndim != 4 or tensor.shape[0] < 1:
        return {}
    return {str(ch): int(np.count_nonzero(tensor[0, ch] > 0)) for ch in range(tensor.shape[1])}


def classify_tensor_diff(pred_tensor: Any, expected_grid: list[list[int]]) -> dict[str, Any]:
    pred = np.asarray(pred_tensor)
    expected = onehot_tensor_from_grid(expected_grid)
    exact = np.array_equal(pred, expected)
    eh, ew = height(expected_grid), width(expected_grid)
    max_h = max(expected.shape[2], pred.shape[2] if pred.ndim == 4 else 0)
    max_w = max(expected.shape[3], pred.shape[3] if pred.ndim == 4 else 0)
    max_h = max(max_h, eh)
    max_w = max(max_w, ew)
    missing_active_channels = 0
    extra_active_channels = 0
    active_wrong_channels = 0
    extra_black_outside = 0
    too_many_colors_cells = 0
    no_color_cells = 0
    first_failing: list[dict[str, Any]] = []
    for r in range(max_h):
        for c in range(max_w):
            exp = _active_channels(expected, r, c)
            got = _active_channels(pred, r, c)
            if len(got) > 1:
                too_many_colors_cells += 1
            if r < eh and c < ew and not got:
                no_color_cells += 1
            exp_set, got_set = set(exp), set(got)
            missing = sorted(exp_set - got_set)
            extra = sorted(got_set - exp_set)
            if missing:
                missing_active_channels += len(missing)
            if extra:
                extra_active_channels += len(extra)
                if r < eh and c < ew:
                    active_wrong_channels += len(extra)
            if (r >= eh or c >= ew) and 0 in got_set:
                extra_black_outside += 1
            if exp != got and len(first_failing) < 20:
                reason = []
                if missing:
                    reason.append("missing_active_channels")
                if extra:
                    reason.append("extra_active_channels")
                if len(got) > 1:
                    reason.append("too_many_colors")
                if not got:
                    reason.append("no_color")
                if (r >= eh or c >= ew) and 0 in got_set:
                    reason.append("extra_active_channel_0_outside_expected_output_shape")
                first_failing.append(
                    {
                        "row": r,
                        "col": c,
                        "expected_tensor_active_channels": exp,
                        "predicted_tensor_active_channels": got,
                        "reason": reason,
                    }
                )
    return {
        "kind": "exact" if exact else "official_tensor_mismatch",
        "official_tensor_equal": bool(exact),
        "predicted_tensor_shape": list(pred.shape),
        "expected_tensor_shape": list(expected.shape),
        "expected_grid_shape": [eh, ew],
        "active_wrong_channels": active_wrong_channels,
        "missing_active_channels": missing_active_channels,
        "extra_active_channels": extra_active_channels,
        "extra_active_channel_0_outside_expected_output_shape": extra_black_outside,
        "too_many_colors_cells": too_many_colors_cells,
        "no_color_cells": no_color_cells,
        "first_failing_coordinates": first_failing,
        "expected_tensor_active_channels": _active_channel_counts(expected),
        "predicted_tensor_active_channels": _active_channel_counts(pred),
    }


def _load_pairs(task_dir: Path, sources: str) -> list[GridPair]:
    selected: list[GridPair] = []
    source_set = {s.strip() for s in sources.split(",") if s.strip()}
    if source_set & {"public", "train", "test"}:
        selected.extend(load_public_pairs(task_dir))
    if source_set & {"arc", "arc-gen", "arc_gen"}:
        selected.extend(load_arc_gen_pairs(task_dir))
    if source_set & {"all"}:
        selected = load_all_available_pairs(task_dir)
    return selected


def _compare_pair(pred_tensor: Any, pair: GridPair, comparison: ComparisonMode) -> tuple[bool, dict[str, Any] | None, bool, bool]:
    official_diff = classify_tensor_diff(pred_tensor, pair.output)
    official_exact = bool(official_diff["official_tensor_equal"])
    pred_grid = tensor_to_grid(pred_tensor)
    fresh_exact = grid_equal_with_missing_zero(pred_grid, pair.output)
    fresh_diff = None if fresh_exact else classify_diff(pred_grid, pair.output)
    if comparison == "official_public_arc":
        return official_exact, None if official_exact else official_diff, official_exact, fresh_exact
    if comparison == "fresh_grid":
        return fresh_exact, None if fresh_exact else fresh_diff, official_exact, fresh_exact
    if comparison == "both":
        if official_exact and fresh_exact:
            return True, None, official_exact, fresh_exact
        diff: dict[str, Any] = {
            "kind": "comparison_mismatch",
            "official_public_arc": official_diff,
            "fresh_grid": {"kind": "exact"} if fresh_exact else fresh_diff,
        }
        return False, diff, official_exact, fresh_exact
    raise ValueError(f"unsupported comparison mode {comparison}")


def run_pairs(task_dir: Path, model_path: Path, sources: str = "public,arc", comparison: ComparisonMode = "official_public_arc") -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {"ok": False, "error": f"onnxruntime import failed: {type(exc).__name__}: {exc}", "pairs": 0, "failures": []}
    pairs = _load_pairs(task_dir, sources)
    if not pairs:
        return {"ok": False, "error": "no task JSON pairs available", "pairs": 0, "failures": []}
    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
    except Exception as exc:
        return {"ok": False, "error": f"ORT session failed: {type(exc).__name__}: {exc}", "pairs": len(pairs), "failures": []}
    failures: list[dict[str, Any]] = []
    exact = 0
    official_exact_count = 0
    fresh_grid_exact_count = 0
    for pair in pairs:
        try:
            pred_tensor = sess.run(None, {input_name: onehot_tensor_from_grid(pair.input)})[0]
            passed, diff, official_exact, fresh_exact = _compare_pair(pred_tensor, pair, comparison)
            official_exact_count += int(official_exact)
            fresh_grid_exact_count += int(fresh_exact)
            if passed:
                exact += 1
                continue
            failures.append({"source": pair.source, "index": pair.index, **(diff or {"kind": "unknown_mismatch"})})
        except Exception as exc:
            failures.append({"source": pair.source, "index": pair.index, "kind": "runtime_error", "error": f"{type(exc).__name__}: {exc}"})
    return {
        "ok": True,
        "task": task_dir.name,
        "model": str(model_path),
        "sources": sources,
        "comparison": comparison,
        "pairs": len(pairs),
        "exact": exact,
        "wrong": len(failures),
        "official_public_arc_exact": official_exact_count,
        "fresh_grid_exact": fresh_grid_exact_count,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an ONNX candidate on public/ARC task pairs and report failing examples")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", default="public,arc")
    parser.add_argument("--comparison", choices=["official_public_arc", "fresh_grid", "both"], default="official_public_arc")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root_from_file()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    result = run_pairs(root / args.task, model_path, sources=args.sources, comparison=args.comparison)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{args.task}: exact={result.get('exact', 0)}/{result.get('pairs', 0)} wrong={result.get('wrong', 0)}")
        for failure in result.get("failures", [])[:20]:
            print(json.dumps(failure, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
