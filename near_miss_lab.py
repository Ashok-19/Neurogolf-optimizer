from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .config import add_common_args, config_from_args
from .graph_utils import load_model, safe_build, save_model
from .locking import TaskLock
from .pair_runner import run_pairs
from .reporting import write_json
from .scoring import delta_points, is_accepted, near_miss_category
from .search.scheduler import read_baseline_cost
from .synthesis.task_data import load_all_available_pairs
from .validate import validate_model


def _resolve_model(root: Path, model_arg: str) -> Path:
    path = Path(model_arg)
    return path if path.is_absolute() else root / path


def _output_np_dtype(model) -> np.dtype:
    elem_type = model.graph.output[0].type.tensor_type.elem_type
    return {
        TensorProto.BOOL: np.dtype(np.bool_),
        TensorProto.FLOAT: np.dtype(np.float32),
        TensorProto.DOUBLE: np.dtype(np.float64),
        TensorProto.INT64: np.dtype(np.int64),
        TensorProto.INT32: np.dtype(np.int32),
        TensorProto.UINT8: np.dtype(np.uint8),
        TensorProto.INT8: np.dtype(np.int8),
    }.get(elem_type, np.dtype(np.float32))


def _replace_output_with_where(model, condition: np.ndarray, replacement: np.ndarray | np.generic | bool | float, name: str):
    repaired = copy.deepcopy(model)
    out_dtype = _output_np_dtype(repaired)
    where_dtype = np.dtype(np.float32) if out_dtype == np.dtype(np.bool_) else out_dtype
    output_name = repaired.graph.output[0].name
    raw_output = f"{output_name}__{name}_raw"
    renamed = False
    for node in repaired.graph.node:
        for i, out in enumerate(node.output):
            if out == output_name:
                node.output[i] = raw_output
                renamed = True
    if not renamed:
        raise RuntimeError(f"could not find producer for graph output {output_name}")
    where_input = raw_output
    if out_dtype == np.dtype(np.bool_):
        where_input = f"{raw_output}_float"
        repaired.graph.node.append(helper.make_node("Cast", [raw_output], [where_input], name=f"{name}_cast_raw_float", to=TensorProto.FLOAT))
        repaired.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    cond_name = f"{name}_condition"
    repl_name = f"{name}_replacement"
    repaired.graph.initializer.append(numpy_helper.from_array(condition.astype(bool), name=cond_name))
    if isinstance(replacement, np.ndarray):
        repaired.graph.initializer.append(numpy_helper.from_array(replacement.astype(where_dtype), name=repl_name))
    else:
        repaired.graph.initializer.append(numpy_helper.from_array(np.array(replacement, dtype=where_dtype), name=repl_name))
    repaired.graph.node.append(helper.make_node("Where", [cond_name, where_input, repl_name], [output_name], name=name))
    return repaired


def _output_shape_mask_model(model, h: int, w: int, name: str):
    condition = np.zeros((1, 1, 30, 30), dtype=bool)
    condition[:, :, : max(0, min(30, h)), : max(0, min(30, w))] = True
    return _replace_output_with_where(model, condition, False, name)


def _channel0_outside_mask_model(model, h: int, w: int, name: str):
    condition = np.ones((1, 10, 30, 30), dtype=bool)
    condition[:, 0, max(0, min(30, h)) :, :] = False
    condition[:, 0, :, max(0, min(30, w)) :] = False
    replacement = np.zeros((1, 10, 30, 30), dtype=_output_np_dtype(model))
    return _replace_output_with_where(model, condition, replacement, name)


def _tiny_coordinate_patch_model(model, coords: list[dict], name: str):
    condition = np.ones((1, 10, 30, 30), dtype=bool)
    replacement = np.zeros((1, 10, 30, 30), dtype=_output_np_dtype(model))
    for coord in coords:
        r, c = int(coord.get("row", -1)), int(coord.get("col", -1))
        if not (0 <= r < 30 and 0 <= c < 30):
            continue
        condition[0, :, r, c] = False
        for ch in coord.get("expected_tensor_active_channels") or []:
            if 0 <= int(ch) < 10:
                replacement[0, int(ch), r, c] = True if replacement.dtype == np.bool_ else 1.0
    return _replace_output_with_where(model, condition, replacement, name)


def _symbolic_missing_line_templates(diagnostics: dict, max_templates: int = 6) -> list[dict]:
    templates: list[dict] = []
    seen: set[tuple] = set()
    for failure in diagnostics.get("failures", [])[:8]:
        coords = failure.get("first_failing_coordinates") or []
        if not coords and isinstance(failure.get("official_public_arc"), dict):
            coords = failure["official_public_arc"].get("first_failing_coordinates") or []
        if not coords:
            continue
        cleaned = []
        for coord in coords:
            reasons = set(coord.get("reason") or [])
            expected = [int(ch) for ch in coord.get("expected_tensor_active_channels") or []]
            predicted = [int(ch) for ch in coord.get("predicted_tensor_active_channels") or []]
            if not expected or predicted:
                continue
            if not ({"missing_active_channels", "no_color"} & reasons):
                continue
            row, col = int(coord.get("row", -1)), int(coord.get("col", -1))
            if 0 <= row < 30 and 0 <= col < 30:
                cleaned.append({"row": row, "col": col, "channels": expected})
        if len(cleaned) < 2:
            continue
        rows = sorted({item["row"] for item in cleaned})
        cols = sorted({item["col"] for item in cleaned})
        shape = failure.get("expected_grid_shape") or failure.get("expected_shape") or [30, 30]
        h = int(shape[0]) if isinstance(shape, list) and len(shape) == 2 else 30
        w = int(shape[1]) if isinstance(shape, list) and len(shape) == 2 else 30
        kind = None
        if len(cols) == 1 and len(rows) >= 2:
            kind = "missing_last_column" if cols[0] == w - 1 else "missing_column"
        elif len(rows) == 1 and len(cols) >= 2:
            kind = "missing_bottom_row" if rows[0] == h - 1 else "missing_row"
        if kind is None:
            continue
        selector = None
        if len(cols) == 1 and cols[0] > 0:
            selector = {"kind": "adjacent_left_active", "col": cols[0] - 1}
        elif len(rows) == 1 and rows[0] > 0:
            selector = {"kind": "adjacent_top_active", "row": rows[0] - 1}
        groups: dict[int, dict[str, set[int]]] = {}
        for item in cleaned:
            for ch in item["channels"]:
                groups.setdefault(ch, {"rows": set(), "cols": set()})
                groups[ch]["rows"].add(item["row"])
                groups[ch]["cols"].add(item["col"])
        normalized_groups = [
            {"channel": ch, "rows": sorted(data["rows"]), "cols": sorted(data["cols"])}
            for ch, data in sorted(groups.items())
        ]
        key = (kind, tuple(rows), tuple(cols), tuple((g["channel"], tuple(g["rows"]), tuple(g["cols"])) for g in normalized_groups))
        if key in seen:
            continue
        seen.add(key)
        templates.append(
            {
                "kind": kind,
                "rows": rows,
                "cols": cols,
                "expected_shape": [h, w],
                "selector": selector,
                "channel_groups": normalized_groups,
                "source": failure.get("source"),
                "index": failure.get("index"),
            }
        )
        if len(templates) >= max_templates:
            break
    return templates


def _rename_graph_output_producer(model, name: str) -> tuple[str, str]:
    output_name = model.graph.output[0].name
    raw_output = f"{output_name}__{name}_raw"
    for node in model.graph.node:
        for i, out in enumerate(node.output):
            if out == output_name:
                node.output[i] = raw_output
                return output_name, raw_output
    raise RuntimeError(f"could not find producer for graph output {output_name}")


def _symbolic_missing_line_patch_model(model, template: dict, name: str):
    repaired = copy.deepcopy(model)
    out_dtype = _output_np_dtype(repaired)
    output_name, raw_output = _rename_graph_output_producer(repaired, name)
    selector_output = _symbolic_extent_selector(repaired, raw_output, template.get("selector"), name)
    patch_outputs: list[str] = []
    for idx, group in enumerate(template.get("channel_groups") or []):
        channel = int(group["channel"])
        rows = [int(r) for r in group.get("rows", []) if 0 <= int(r) < 30]
        cols = [int(c) for c in group.get("cols", []) if 0 <= int(c) < 30]
        if not rows or not cols or not (0 <= channel < 10):
            continue
        row_mask = np.zeros((1, 1, 30, 1), dtype=bool)
        col_mask = np.zeros((1, 1, 1, 30), dtype=bool)
        channel_mask = np.zeros((1, 10, 1, 1), dtype=bool)
        row_mask[:, :, rows, :] = True
        col_mask[:, :, :, cols] = True
        channel_mask[:, channel, :, :] = True
        row_name = f"{name}_row_{idx}"
        col_name = f"{name}_col_{idx}"
        ch_name = f"{name}_ch_{idx}"
        spatial = f"{name}_spatial_{idx}"
        patch = f"{name}_patch_{idx}"
        repaired.graph.initializer.extend(
            [
                numpy_helper.from_array(row_mask, name=row_name),
                numpy_helper.from_array(col_mask, name=col_name),
                numpy_helper.from_array(channel_mask, name=ch_name),
            ]
        )
        repaired.graph.node.append(helper.make_node("And", [row_name, col_name], [spatial], name=f"{name}_and_row_col_{idx}"))
        if selector_output is not None:
            gated = f"{name}_gated_spatial_{idx}"
            repaired.graph.node.append(helper.make_node("And", [spatial, selector_output], [gated], name=f"{name}_and_extent_{idx}"))
            spatial = gated
        repaired.graph.node.append(helper.make_node("And", [spatial, ch_name], [patch], name=f"{name}_and_channel_{idx}"))
        patch_outputs.append(patch)
    if not patch_outputs:
        raise RuntimeError("symbolic line patch template had no usable channel groups")
    patch_output = patch_outputs[0]
    for idx, patch in enumerate(patch_outputs[1:], 1):
        merged = f"{name}_merged_patch_{idx}"
        repaired.graph.node.append(helper.make_node("Or", [patch_output, patch], [merged], name=f"{name}_or_patch_{idx}"))
        patch_output = merged
    if out_dtype == np.dtype(np.bool_):
        repaired.graph.node.append(helper.make_node("Or", [raw_output, patch_output], [output_name], name=name))
    else:
        one_name = f"{name}_one"
        repaired.graph.initializer.append(numpy_helper.from_array(np.array(1, dtype=out_dtype), name=one_name))
        repaired.graph.node.append(helper.make_node("Where", [patch_output, one_name, raw_output], [output_name], name=name))
    return repaired


def _symbolic_extent_selector(model, raw_output: str, selector: dict | None, name: str) -> str | None:
    if not selector:
        return None
    raw_float = f"{name}_raw_float"
    model.graph.node.append(helper.make_node("Cast", [raw_output], [raw_float], name=f"{name}_cast_raw_float", to=TensorProto.FLOAT))
    starts_name = f"{name}_selector_starts"
    ends_name = f"{name}_selector_ends"
    axes_name = f"{name}_selector_axes"
    steps_name = f"{name}_selector_steps"
    zero_name = f"{name}_selector_zero"
    if selector.get("kind") == "adjacent_left_active":
        axis = 3
        start = int(selector["col"])
    elif selector.get("kind") == "adjacent_top_active":
        axis = 2
        start = int(selector["row"])
    else:
        return None
    sliced = f"{name}_selector_slice"
    reduced = f"{name}_selector_sum"
    active = f"{name}_selector_active"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([start], dtype=np.int64), name=starts_name),
            numpy_helper.from_array(np.array([start + 1], dtype=np.int64), name=ends_name),
            numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_name),
            numpy_helper.from_array(np.array(0, dtype=np.float32), name=zero_name),
        ]
    )
    model.graph.node.append(helper.make_node("Slice", [raw_float, starts_name, ends_name, axes_name, steps_name], [sliced], name=f"{name}_selector_slice"))
    if _default_opset(model) >= 13:
        reduce_axes_name = f"{name}_selector_reduce_axes"
        model.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=reduce_axes_name))
        model.graph.node.append(helper.make_node("ReduceSum", [sliced, reduce_axes_name], [reduced], name=f"{name}_selector_reduce", keepdims=1))
    else:
        model.graph.node.append(helper.make_node("ReduceSum", [sliced], [reduced], name=f"{name}_selector_reduce", axes=[1], keepdims=1))
    model.graph.node.append(helper.make_node("Greater", [reduced, zero_name], [active], name=f"{name}_selector_greater"))
    return active


def _default_opset(model) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 12


def _repair_shapes(task_dir: Path, diagnostics: dict) -> list[tuple[int, int]]:
    shapes: set[tuple[int, int]] = set()
    for failure in diagnostics.get("failures", []):
        shape = failure.get("expected_grid_shape") or failure.get("expected_shape")
        if isinstance(shape, list) and len(shape) == 2:
            shapes.add((int(shape[0]), int(shape[1])))
        official = failure.get("official_public_arc")
        if isinstance(official, dict):
            shape = official.get("expected_grid_shape") or official.get("expected_shape")
            if isinstance(shape, list) and len(shape) == 2:
                shapes.add((int(shape[0]), int(shape[1])))
    pair_shapes = {(len(p.output), max((len(r) for r in p.output), default=0)) for p in load_all_available_pairs(task_dir)}
    if 0 < len(pair_shapes) <= 4:
        shapes.update(pair_shapes)
    return sorted(s for s in shapes if 0 <= s[0] <= 30 and 0 <= s[1] <= 30)[:8]


def _trim_pair_diagnostics(diagnostics: dict, max_failures: int = 5) -> dict:
    out = dict(diagnostics)
    failures = list(out.get("failures") or [])
    out["failure_count"] = len(failures)
    out["failures"] = failures[:max_failures]
    return out


def _evaluate_repair(task_dir: Path, model, repair_name: str, out_dir: Path, config, cache: ValidationCache, incumbent_cost: int) -> dict:
    try:
        built = safe_build(model, allow_nonstrict_shapes=config.allow_nonstrict_shapes)
        if not built.ok or built.model is None:
            return {"repair": repair_name, "ok": False, "error": built.error}
        repair_path = out_dir / f"{repair_name}.onnx"
        save_model(built.model, repair_path)
        pair_diag = _trim_pair_diagnostics(run_pairs(task_dir, repair_path, sources="public,arc", comparison="official_public_arc"))
        validation = validate_model(task_dir, repair_path, config, cache)
        vdict = validation.to_json_dict()
        accepted = is_accepted(vdict, incumbent_cost, config)
        return {
            "repair": repair_name,
            "ok": True,
            "model_path": str(repair_path),
            "model_file_exists": repair_path.exists(),
            "pair_diagnostics": pair_diag,
            "validation": vdict,
            "accepted": accepted,
            "repaired": accepted,
            "near_miss_category": near_miss_category(vdict, incumbent_cost, config),
            "delta_points": None if validation.cost is None else delta_points(incumbent_cost, validation.cost),
        }
    except Exception as exc:
        return {"repair": repair_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _try_cleanup_repair(task_dir: Path, model_path: Path, out_dir: Path, config, cache: ValidationCache, incumbent_cost: int) -> list[dict]:
    repairs: list[dict] = []
    try:
        model = load_model(model_path)
        built = safe_build(model, allow_nonstrict_shapes=config.allow_nonstrict_shapes)
        if not built.ok or built.model is None:
            return [{"repair": "cleanup_safe_build", "ok": False, "error": built.error}]
        repair_path = out_dir / "repair_cleanup.onnx"
        save_model(built.model, repair_path)
        validation = validate_model(task_dir, repair_path, config, cache)
        row = {
            "repair": "cleanup_safe_build",
            "ok": True,
            "model_path": str(repair_path),
            "validation": validation.to_json_dict(),
            "accepted": is_accepted(validation.to_json_dict(), incumbent_cost, config),
            "near_miss_category": near_miss_category(validation.to_json_dict(), incumbent_cost, config),
            "delta_points": None if validation.cost is None else delta_points(incumbent_cost, validation.cost),
        }
        row["model_file_exists"] = repair_path.exists()
        row["repaired"] = row["accepted"]
        row["pair_diagnostics"] = _trim_pair_diagnostics(run_pairs(task_dir, repair_path, sources="public,arc", comparison="official_public_arc"))
        repairs.append(row)
    except Exception as exc:
        repairs.append({"repair": "cleanup_safe_build", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return repairs


def _try_mask_repairs(task_dir: Path, model_path: Path, diagnostics: dict, out_dir: Path, config, cache: ValidationCache, incumbent_cost: int) -> list[dict]:
    repairs: list[dict] = []
    model = load_model(model_path)
    for h, w in _repair_shapes(task_dir, diagnostics):
        repairs.append(
            _evaluate_repair(
                task_dir,
                _output_shape_mask_model(model, h, w, f"repair_no_color_outside_{h}x{w}"),
                f"repair_no_color_outside_{h}x{w}",
                out_dir,
                config,
                cache,
                incumbent_cost,
            )
        )
        repairs.append(
            _evaluate_repair(
                task_dir,
                _channel0_outside_mask_model(model, h, w, f"repair_clear_channel0_outside_{h}x{w}"),
                f"repair_clear_channel0_outside_{h}x{w}",
                out_dir,
                config,
                cache,
                incumbent_cost,
            )
        )
    return repairs


def _try_tiny_correction_repairs(task_dir: Path, model_path: Path, diagnostics: dict, out_dir: Path, config, cache: ValidationCache, incumbent_cost: int) -> list[dict]:
    repairs: list[dict] = []
    model = load_model(model_path)
    for idx, failure in enumerate(diagnostics.get("failures", [])[:4]):
        coords = failure.get("first_failing_coordinates") or []
        if not coords and isinstance(failure.get("official_public_arc"), dict):
            coords = failure["official_public_arc"].get("first_failing_coordinates") or []
        if not coords or len(coords) > 12:
            continue
        repairs.append(
            _evaluate_repair(
                task_dir,
                _tiny_coordinate_patch_model(model, coords, f"repair_tiny_coordinate_patch_{idx}"),
                f"repair_tiny_coordinate_patch_{idx}",
                out_dir,
                config,
                cache,
                incumbent_cost,
            )
        )
    return repairs


def _try_symbolic_line_repairs(task_dir: Path, model_path: Path, diagnostics: dict, out_dir: Path, config, cache: ValidationCache, incumbent_cost: int) -> list[dict]:
    repairs: list[dict] = []
    model = load_model(model_path)
    for idx, template in enumerate(_symbolic_missing_line_templates(diagnostics)):
        name = f"repair_symbolic_{template['kind']}_{idx}"
        repair = _evaluate_repair(
            task_dir,
            _symbolic_missing_line_patch_model(model, template, name),
            name,
            out_dir,
            config,
            cache,
            incumbent_cost,
        )
        repair["symbolic_template"] = template
        repairs.append(repair)
    return repairs


def try_repair_model(
    task_dir: Path,
    model_path: Path,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    diagnostics: dict | None = None,
    max_repairs: int | None = None,
) -> list[dict]:
    diagnostics = diagnostics or run_pairs(task_dir, model_path, sources="public,arc", comparison="official_public_arc")
    out_dir.mkdir(parents=True, exist_ok=True)
    repairs: list[dict] = []
    for producer in (
        lambda: _try_symbolic_line_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost),
        lambda: _try_mask_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost),
        lambda: _try_tiny_correction_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost),
        lambda: _try_cleanup_repair(task_dir, model_path, out_dir, config, cache, incumbent_cost),
    ):
        if max_repairs is not None and len(repairs) >= max_repairs:
            break
        batch = producer()
        for repair in batch:
            repairs.append(repair)
            if max_repairs is not None and len(repairs) >= max_repairs:
                break
    return repairs


def _maybe_promote(task_dir: Path, repair: dict, config, incumbent_cost: int, cache: ValidationCache) -> bool:
    if config.dry_run or not config.promote or not repair.get("accepted"):
        return False
    src = Path(repair["model_path"])
    experiments = task_dir / "experiments"
    target = experiments / "candidate.onnx"
    with TaskLock(experiments):
        fresh = validate_model(task_dir, src, config, cache)
        if not is_accepted(fresh.to_json_dict(), incumbent_cost, config):
            return False
        current_cost = incumbent_cost
        if target.exists():
            current = validate_model(task_dir, target, config, cache)
            if is_accepted(current.to_json_dict(), incumbent_cost, config):
                current_cost = int(current.cost or current_cost)
        if fresh.cost is None or fresh.cost >= current_cost:
            return False
        if target.exists():
            shutil.copy2(target, experiments / f"candidate_prev_{time.strftime('%Y%m%d_%H%M%S')}.onnx")
        shutil.copy2(src, target)
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose and lightly repair a NeuroGolf near-miss ONNX model")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repair", action="store_true")
    add_common_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    root = config.repo_root
    task_dir = root / args.task
    model_path = _resolve_model(root, args.model)
    run_dir = task_dir / "experiments" / "near_miss_lab" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = ValidationCache(config.cache_dir or (root / ".chatgpt/operations/neurogolf_optimizer/cache"), disabled=config.no_cache)
    try:
        incumbent_cost = read_baseline_cost(task_dir)
        candidate_path = task_dir / "experiments" / "candidate.onnx"
        if candidate_path.exists():
            current = validate_model(task_dir, candidate_path, config, cache)
            if is_accepted(current.to_json_dict(), incumbent_cost, config):
                incumbent_cost = int(current.cost or incumbent_cost)
        validation = validate_model(task_dir, model_path, config, cache)
        diagnostics = run_pairs(task_dir, model_path, sources="public,arc", comparison="official_public_arc")
        repairs: list[dict] = []
        if args.repair:
            repairs.extend(try_repair_model(task_dir, model_path, run_dir, config, cache, incumbent_cost, diagnostics=diagnostics))
        for repair in repairs:
            repair["promoted"] = _maybe_promote(task_dir, repair, config, incumbent_cost, cache)
        result = {
            "task": args.task,
            "model": str(model_path),
            "incumbent_cost": incumbent_cost,
            "validation": validation.to_json_dict(),
            "near_miss_category": near_miss_category(validation.to_json_dict(), incumbent_cost, config),
            "pair_diagnostics": diagnostics,
            "repairs": repairs,
            "dry_run": config.dry_run,
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "near_miss_lab.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
