from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "neurogolf_optimizer"

from .cache import ValidationCache
from .config import add_common_args, config_from_args
from .graph_utils import (
    get_consumers,
    get_tensor_elem_type,
    get_value_info_shapes,
    initializer_signature,
    initializer_arrays,
    load_model,
    prune_unreachable,
    remove_unused_initializers,
    remove_unused_value_info,
    safe_build,
    save_model,
    set_initializer,
    tensor_dtype_size,
)
from .pair_runner import run_pairs
from .promotion import promote_candidate
from .repair_classification import FailureClass, classify_diagnostics
from .reporting import write_json
from .scoring import delta_points, is_accepted, near_miss_category
from .search.scheduler import read_baseline_cost
from .synthesis.task_data import load_all_available_pairs
from .validate import validate_model


def _normalize_source_mutations(source_mutations: list[Any] | None = None, source_provenance: list[dict] | None = None) -> list[dict]:
    raw = source_provenance if source_provenance is not None else source_mutations
    out: list[dict] = []
    for item in raw or []:
        if isinstance(item, dict):
            data = dict(item)
        elif hasattr(item, "to_json_dict"):
            data = dict(item.to_json_dict())
        else:
            data = {
                "id": getattr(item, "id", None),
                "family": getattr(item, "family", None),
                "description": getattr(item, "description", None),
                "metadata": getattr(item, "metadata", None) or {},
            }
        data.setdefault("metadata", {})
        out.append(
            {
                "id": data.get("id"),
                "family": data.get("family"),
                "description": data.get("description"),
                "metadata": data.get("metadata") or {},
            }
        )
    return out


def _source_provenance_summary(source_provenance: list[dict]) -> dict:
    families = [str(item.get("family")) for item in source_provenance if item.get("family")]
    ids = [str(item.get("id")) for item in source_provenance if item.get("id")]
    return {
        "source_mutations": source_provenance,
        "source_mutation_families": list(dict.fromkeys(families)),
        "source_mutation_ids": list(dict.fromkeys(ids)),
    }


def _matching_source_mutation(failure: FailureClass, source_provenance: list[dict], family: str | None = None) -> dict | None:
    axis_target = _failure_axis_and_target(failure)
    if axis_target is None:
        return None
    axis, target_len = axis_target
    reduced_len = target_len - 1
    for item in source_provenance:
        if family is not None and item.get("family") != family:
            continue
        meta = item.get("metadata") or {}
        if item.get("family") == "dimension_sweep":
            try:
                old_len = int(meta.get("old_len", -1))
                new_len = int(meta.get("new_len", -1))
            except (TypeError, ValueError):
                continue
            if meta.get("axis") == axis and old_len == target_len and new_len == reduced_len:
                return item
        elif item.get("family") in {"where_ablation", "where_combo"} and failure.kind in {"missing_bottom_row", "missing_row", "missing_border"}:
            return item
    return None


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


def _array_is_arange(arr: np.ndarray) -> bool:
    flat = np.asarray(arr).reshape(-1)
    return flat.size >= 2 and flat.size <= 30 and np.array_equal(flat, np.arange(flat.size, dtype=flat.dtype))


def _axis_from_name(name: str | None) -> str | None:
    if not name:
        return None
    low = name.lower()
    if "row" in low or "height" in low or low in {"r", "rr"}:
        return "row"
    if "col" in low or "width" in low or low in {"c", "cc"}:
        return "col"
    return None


def _axis_from_arange_shape(arr: np.ndarray) -> str | None:
    if arr.ndim == 4:
        if arr.shape[2] > 1 and arr.shape[3] == 1:
            return "row"
        if arr.shape[2] == 1 and arr.shape[3] > 1:
            return "col"
    if arr.ndim == 2:
        if arr.shape[0] > 1 and arr.shape[1] == 1:
            return "row"
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return "col"
    return None


def _node_axis_hint(node) -> str | None:
    axis = _axis_from_name(getattr(node, "name", ""))
    if axis:
        return axis
    for out in getattr(node, "output", []) or []:
        axis = _axis_from_name(out)
        if axis:
            return axis
    return None


def _failure_axis_and_target(failure: FailureClass) -> tuple[str, int] | None:
    if not failure.expected_shape:
        return None
    if failure.kind in {"missing_last_column", "missing_column", "wrong_output_width"}:
        return "col", int(failure.expected_shape[1])
    if failure.kind in {"missing_bottom_row", "missing_row", "wrong_output_height"}:
        return "row", int(failure.expected_shape[0])
    return None


def _unique_initializer_name(model, base: str) -> str:
    names = {init.name for init in model.graph.initializer}
    if base not in names:
        return base
    idx = 1
    while f"{base}_{idx}" in names:
        idx += 1
    return f"{base}_{idx}"


def _internal_dimension_restore_model(model, failure: FailureClass, _name: str, source_provenance: list[dict] | None = None) -> tuple[object | None, dict]:
    axis_target = _failure_axis_and_target(failure)
    if axis_target is None:
        return None, {"reason": "unsupported_failure_class"}
    axis, target_len = axis_target
    old_len = target_len - 1
    if old_len < 2 or target_len > 30:
        return None, {"reason": "invalid_extent", "axis": axis, "target_len": target_len}
    repaired = copy.deepcopy(model)
    arrays = initializer_arrays(repaired)
    consumers = get_consumers(repaired)
    changed_initializers: list[dict] = []
    cloned_initializers: list[dict] = []
    reused_initializers: list[dict] = []
    source_match = _matching_source_mutation(failure, source_provenance or [], "dimension_sweep")
    provenance_family = source_match.get("family") if source_match else "dimension_sweep_inferred"
    changed = False
    for init_name, arr in list(arrays.items()):
        if not np.issubdtype(arr.dtype, np.integer) or not _array_is_arange(arr):
            continue
        init_axis = _axis_from_name(init_name) or _axis_from_arange_shape(arr)
        if init_axis == axis and arr.size == old_len and arr.ndim == 4:
            if axis == "row":
                new_arr = np.arange(target_len, dtype=arr.dtype).reshape(1, 1, target_len, 1)
            else:
                new_arr = np.arange(target_len, dtype=arr.dtype).reshape(1, 1, 1, target_len)
            set_initializer(repaired, init_name, new_arr)
            changed_initializers.append({"name": init_name, "old_shape": list(arr.shape), "new_shape": list(new_arr.shape)})
            changed = True
        elif arr.ndim == 1 and arr.size == old_len:
            matching_consumers = [node for node in consumers.get(init_name, []) if _node_axis_hint(node) == axis]
            if not matching_consumers:
                continue
            reusable_name = None
            for candidate_name, candidate_arr in arrays.items():
                if (
                    candidate_arr.ndim == 1
                    and candidate_arr.size == target_len
                    and candidate_arr.dtype == arr.dtype
                    and _array_is_arange(candidate_arr)
                ):
                    reusable_name = candidate_name
                    break
            if reusable_name is None:
                new_name = _unique_initializer_name(repaired, f"{init_name}_{axis}_{target_len}")
                new_arr = np.arange(target_len, dtype=arr.dtype)
                repaired.graph.initializer.append(numpy_helper.from_array(new_arr, name=new_name))
                cloned_initializers.append(
                    {
                        "source": init_name,
                        "new_name": new_name,
                        "old_shape": list(arr.shape),
                        "new_shape": list(new_arr.shape),
                        "consumer_nodes": [node.name for node in matching_consumers],
                    }
                )
            else:
                new_name = reusable_name
                reused_initializers.append(
                    {
                        "source": init_name,
                        "reused_name": reusable_name,
                        "old_shape": list(arr.shape),
                        "consumer_nodes": [node.name for node in matching_consumers],
                    }
                )
            for node in matching_consumers:
                for i, inp in enumerate(node.input):
                    if inp == init_name:
                        node.input[i] = new_name
                        changed = True
    pad_change = None
    pad = initializer_arrays(repaired).get("pad_to_30_i64")
    if isinstance(pad, np.ndarray) and pad.shape == (8,):
        new_pad = pad.astype(np.int64).copy()
        pad_index = 6 if axis == "row" else 7
        new_pad[pad_index] = max(0, 30 - target_len)
        if not np.array_equal(pad, new_pad):
            set_initializer(repaired, "pad_to_30_i64", new_pad)
            pad_change = {"name": "pad_to_30_i64", "old": pad.astype(np.int64).tolist(), "new": new_pad.tolist()}
            changed = True
    if not changed or (not changed_initializers and not cloned_initializers and not reused_initializers):
        return None, {"reason": "no_axis_extent_to_restore", "axis": axis, "target_len": target_len}
    del repaired.graph.value_info[:]
    return repaired, {
        "strategy": "restore_axis_dimension",
        "axis": axis,
        "old_len": old_len,
        "new_len": target_len,
        "changed_initializers": changed_initializers,
        "cloned_initializers": cloned_initializers,
        "reused_initializers": reused_initializers,
        "pad_to_30_change": pad_change,
        "source_mutation_family": provenance_family,
        "source_mutation": source_match,
    }


def _scalar_one_for_dtype(dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer):
        return np.array(1, dtype=dtype)
    return np.array(1, dtype=np.float32)


def _scalar_array_for_elem_type(value: int | float, elem_type: int | None) -> np.ndarray:
    dtype = np.float32
    if elem_type == TensorProto.FLOAT16:
        dtype = np.float16
    elif elem_type == TensorProto.FLOAT:
        dtype = np.float32
    elif elem_type == TensorProto.DOUBLE:
        dtype = np.float64
    elif elem_type == TensorProto.INT64:
        dtype = np.int64
    elif elem_type == TensorProto.INT32:
        dtype = np.int32
    elif elem_type == TensorProto.UINT8:
        dtype = np.uint8
    elif elem_type == TensorProto.INT16:
        dtype = np.int16
    elif elem_type == TensorProto.UINT16:
        dtype = np.uint16
    elif elem_type == TensorProto.INT8:
        dtype = np.int8
    return np.array([value], dtype=dtype)


def _shape_numel(shape: list[int | None] | None) -> int | None:
    if not shape or any(dim is None for dim in shape):
        return None
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _scalar_bool_condition_candidates(model, less_index: int, source_match: dict | None = None, limit: int = 3) -> list[str]:
    shapes = get_value_info_shapes(model)
    out: list[str] = []
    condition = ((source_match or {}).get("metadata") or {}).get("condition_tensor") or ((source_match or {}).get("metadata") or {}).get("condition")
    if condition and condition in shapes and get_tensor_elem_type(model, condition) == TensorProto.BOOL and _shape_numel(shapes.get(condition)) == 1:
        out.append(str(condition))
    nodes = list(model.graph.node)
    for node in reversed(nodes[:less_index]):
        if node.op_type != "Where" or not node.input:
            continue
        cond = node.input[0]
        if cond in out:
            continue
        if get_tensor_elem_type(model, cond) != TensorProto.BOOL:
            continue
        if _shape_numel(shapes.get(cond)) != 1:
            continue
        out.append(cond)
        if len(out) >= limit:
            break
    return out[:limit]


def _internal_gated_expected_extent_models(
    model,
    failure: FailureClass,
    name_prefix: str,
    limit: int = 2,
    source_provenance: list[dict] | None = None,
) -> list[tuple[str, object, dict]]:
    axis_target = _failure_axis_and_target(failure)
    if axis_target is None or failure.kind not in {"missing_bottom_row", "missing_row", "missing_last_column", "missing_column"}:
        return []
    axis, target_len = axis_target
    if target_len <= 0 or target_len > 30:
        return []
    source_match = _matching_source_mutation(failure, source_provenance or [], "where_ablation")
    if source_match is None:
        source_match = _matching_source_mutation(failure, source_provenance or [], "where_combo")
    provenance_family = source_match.get("family") if source_match else "where_ablation_inferred"
    arrays = initializer_arrays(model)
    matches: list[tuple[int, object, np.ndarray]] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Less" or len(node.input) < 2:
            continue
        coord = arrays.get(node.input[0])
        if coord is None or _axis_from_arange_shape(coord) != axis:
            continue
        matches.append((idx, node, coord))
    out: list[tuple[str, object, dict]] = []
    for variant_idx, (node_idx, node, _coord) in enumerate(reversed(matches[-limit:])):
        old_extent = node.input[1]
        elem_type = get_tensor_elem_type(model, old_extent)
        if elem_type is None:
            continue
        for cond_idx, condition in enumerate(_scalar_bool_condition_candidates(model, node_idx, source_match=source_match)):
            repaired = copy.deepcopy(model)
            target_node = list(repaired.graph.node)[node_idx]
            target_name = _unique_initializer_name(repaired, f"{name_prefix}_{variant_idx}_{cond_idx}_target_extent")
            max_name = f"{name_prefix}_{variant_idx}_{cond_idx}_max_expected_extent"
            repaired_name = f"{name_prefix}_{variant_idx}_{cond_idx}_gated_extent"
            repaired.graph.initializer.append(numpy_helper.from_array(_scalar_array_for_elem_type(target_len, elem_type), name=target_name))
            repaired.graph.node.append(helper.make_node("Max", [old_extent, target_name], [max_name], name=max_name))
            repaired.graph.node.append(helper.make_node("Where", [condition, max_name, old_extent], [repaired_name], name=repaired_name))
            for i, inp in enumerate(target_node.input):
                if inp == old_extent:
                    target_node.input[i] = repaired_name
            del repaired.graph.value_info[:]
            out.append(
                (
                    f"{name_prefix}_{variant_idx}_gated_expected_{cond_idx}",
                    repaired,
                    {
                        "strategy": "gated_expected_extent",
                        "axis": axis,
                        "target_len": target_len,
                        "less_node": node.name,
                        "less_node_output": list(node.output),
                        "less_node_index": node_idx,
                        "old_extent": old_extent,
                        "selector_condition": condition,
                        "new_extent": repaired_name,
                        "source_mutation_family": provenance_family,
                        "source_mutation": source_match,
                    },
                )
            )
            if len(out) >= limit:
                return out
    return out


def _internal_scalar_extent_plus_one_models(
    model,
    failure: FailureClass,
    name_prefix: str,
    limit: int = 2,
    source_provenance: list[dict] | None = None,
) -> list[tuple[str, object, dict]]:
    axis_target = _failure_axis_and_target(failure)
    if axis_target is None:
        return []
    axis, _target_len = axis_target
    source_match = _matching_source_mutation(failure, source_provenance or [], "where_ablation")
    if source_match is None:
        source_match = _matching_source_mutation(failure, source_provenance or [], "where_combo")
    provenance_family = source_match.get("family") if source_match else "where_ablation_inferred"
    arrays = initializer_arrays(model)
    matches: list[tuple[int, object, np.ndarray]] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Less" or len(node.input) < 2:
            continue
        coord = arrays.get(node.input[0])
        if coord is None or _axis_from_arange_shape(coord) != axis:
            continue
        matches.append((idx, node, coord))
    out: list[tuple[str, object, dict]] = []
    for variant_idx, (node_idx, node, coord) in enumerate(reversed(matches[-limit:])):
        repaired = copy.deepcopy(model)
        target_node = list(repaired.graph.node)[node_idx]
        old_extent = target_node.input[1]
        one_name = _unique_initializer_name(repaired, f"{name_prefix}_{variant_idx}_one")
        plus_name = f"{name_prefix}_{variant_idx}_extent_plus_one"
        repaired.graph.initializer.append(numpy_helper.from_array(_scalar_one_for_dtype(coord.dtype), name=one_name))
        repaired.graph.node.insert(node_idx, helper.make_node("Add", [old_extent, one_name], [plus_name], name=plus_name))
        list(repaired.graph.node)[node_idx + 1].input[1] = plus_name
        del repaired.graph.value_info[:]
        out.append(
            (
                f"{name_prefix}_{variant_idx}",
                repaired,
                {
                    "strategy": "scalar_extent_plus_one",
                    "axis": axis,
                    "less_node": node.name,
                    "less_node_index": node_idx,
                    "old_extent": old_extent,
                    "new_extent": plus_name,
                    "coordinate_initializer": node.input[0],
                    "source_mutation_family": provenance_family,
                    "source_mutation": source_match,
                },
            )
        )
    return out


def _internal_selector_extent_candidates(
    model,
    failures: list[FailureClass],
    source_provenance: list[dict] | None = None,
) -> list[tuple[str, object | None, dict]]:
    candidates: list[tuple[str, object | None, dict]] = []
    for idx, failure in enumerate(failures[:6]):
        if failure.kind not in {"missing_last_column", "missing_bottom_row", "missing_column", "missing_row"}:
            continue
        base = f"repair_internal_{failure.kind}_{idx}"
        restored, restore_meta = _internal_dimension_restore_model(model, failure, f"{base}_dimension", source_provenance=source_provenance)
        if restored is not None:
            candidates.append((f"{base}_dimension", restored, {"failure": failure.to_json_dict(), **restore_meta}))
        elif failure.kind in {"missing_bottom_row", "missing_last_column"}:
            for repair_name, repaired, meta in _internal_gated_expected_extent_models(
                model,
                failure,
                f"{base}_gated_extent",
                source_provenance=source_provenance,
            ):
                candidates.append((repair_name, repaired, {"failure": failure.to_json_dict(), **meta}))
            for repair_name, repaired, meta in _internal_scalar_extent_plus_one_models(
                model,
                failure,
                f"{base}_scalar",
                source_provenance=source_provenance,
            ):
                candidates.append((repair_name, repaired, {"failure": failure.to_json_dict(), **meta}))
        else:
            candidates.append((f"{base}_skipped", None, {"failure": failure.to_json_dict(), **restore_meta}))
    return candidates


def _static_cost_estimate(model) -> int | None:
    try:
        shapes = get_value_info_shapes(model)
        graph_inputs = {vi.name for vi in model.graph.input}
        graph_outputs = {vi.name for vi in model.graph.output}
        init_names = {init.name for init in model.graph.initializer}
        params = 0
        for init in model.graph.initializer:
            try:
                params += int(numpy_helper.to_array(init).size)
            except Exception:
                continue
        memory = 0
        for name, shape in shapes.items():
            if name in graph_inputs or name in graph_outputs or name in init_names:
                continue
            if not shape or any(dim is None for dim in shape):
                continue
            elem_type = get_tensor_elem_type(model, name)
            if elem_type is None:
                continue
            numel = 1
            for dim in shape:
                numel *= int(dim)
            memory += int(numel * tensor_dtype_size(elem_type))
        return memory + params
    except Exception:
        return None


def _model_cost_breakdown(model) -> dict:
    shapes = get_value_info_shapes(model)
    graph_inputs = {vi.name for vi in model.graph.input}
    graph_outputs = {vi.name for vi in model.graph.output}
    init_names = {init.name for init in model.graph.initializer}
    consumers = get_consumers(model)
    params = 0
    initializers: list[dict] = []
    for init in model.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        size = int(arr.size)
        params += size
        initializers.append({"name": init.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "size": size})
    memory = 0
    intermediates: list[dict] = []
    full_volume_count = 0
    for name, shape in shapes.items():
        if name in graph_inputs or name in graph_outputs or name in init_names:
            continue
        if not shape or any(dim is None for dim in shape):
            continue
        elem_type = get_tensor_elem_type(model, name)
        if elem_type is None:
            continue
        numel = 1
        for dim in shape:
            numel *= int(dim)
        size = int(numel * tensor_dtype_size(elem_type))
        memory += size
        if shape == [1, 10, 30, 30] or shape == [1, 1, 30, 30]:
            full_volume_count += 1
        intermediates.append({"name": name, "shape": list(shape), "elem_type": int(elem_type), "bytes": size})
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": params,
        "memory": memory,
        "cost": params + memory,
        "largest_initializers": sorted(initializers, key=lambda x: x["size"], reverse=True)[:8],
        "largest_intermediates": sorted(intermediates, key=lambda x: x["bytes"], reverse=True)[:8],
        "full_volume_intermediate_count": full_volume_count,
        "unused_initializers": sorted(name for name in init_names if not consumers.get(name) and name not in graph_inputs)[:20],
    }


def _cost_diagnostics(source_model_path: str | Path | None, candidate_model) -> dict:
    candidate = _model_cost_breakdown(candidate_model)
    out = {"candidate": candidate}
    if source_model_path:
        try:
            source_built = safe_build(load_model(source_model_path), allow_nonstrict_shapes=True)
            if source_built.ok and source_built.model is not None:
                source = _model_cost_breakdown(source_built.model)
                out["source"] = source
                out["delta"] = {
                    "nodes": candidate["nodes"] - source["nodes"],
                    "initializers": candidate["initializers"] - source["initializers"],
                    "params": candidate["params"] - source["params"],
                    "memory": candidate["memory"] - source["memory"],
                    "cost": candidate["cost"] - source["cost"],
                    "full_volume_intermediate_count": candidate["full_volume_intermediate_count"] - source["full_volume_intermediate_count"],
                }
        except Exception as exc:
            out["source_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _dedup_initializers(model) -> tuple[object, list[str]]:
    repaired = copy.deepcopy(model)
    seen: dict[tuple[int, tuple[int, ...], str], str] = {}
    replacements: list[str] = []
    for init in list(repaired.graph.initializer):
        try:
            sig = initializer_signature(init)
        except Exception:
            continue
        existing = seen.get(sig)
        if existing is None:
            seen[sig] = init.name
            continue
        for node in repaired.graph.node:
            for i, inp in enumerate(node.input):
                if inp == init.name:
                    node.input[i] = existing
        replacements.append(f"{init.name}->{existing}")
    remove_unused_initializers(repaired)
    remove_unused_value_info(repaired)
    return repaired, replacements


def _post_repair_cleanup_models(model, base_repair_name: str, base_estimated_cost: int | None) -> list[tuple[str, object, dict]]:
    variants: list[tuple[str, object, dict]] = []
    pruned = prune_unreachable(copy.deepcopy(model))
    remove_unused_initializers(pruned)
    remove_unused_value_info(pruned)
    variants.append(
        (
            f"{base_repair_name}_cleanup_prune",
            pruned,
            {
                "repair_chain": [base_repair_name, "cleanup_prune"],
                "post_repair_cleanup_steps": ["prune_unreachable", "remove_unused_initializers", "remove_unused_value_info"],
                "cost_before_cleanup": base_estimated_cost,
            },
        )
    )
    deduped, replacements = _dedup_initializers(model)
    variants.append(
        (
            f"{base_repair_name}_cleanup_dedup",
            deduped,
            {
                "repair_chain": [base_repair_name, "cleanup_dedup"],
                "post_repair_cleanup_steps": ["initializer_dedup", "remove_unused_initializers", "remove_unused_value_info"],
                "initializer_dedup_replacements": replacements,
                "cost_before_cleanup": base_estimated_cost,
            },
        )
    )
    return variants


def _repair_cost_gate(model, incumbent_cost: int, internal_or_dense: str) -> dict:
    full_dense_initializers = []
    for init in model.graph.initializer:
        if tuple(int(dim) for dim in init.dims) == (1, 10, 30, 30):
            full_dense_initializers.append(init.name)
    graph_outputs = {out.name for out in model.graph.output}
    full_volume_intermediates = [
        name
        for name, shape in get_value_info_shapes(model).items()
        if shape == [1, 10, 30, 30] and name not in graph_outputs
    ]
    estimate = _static_cost_estimate(model)
    reasons: list[str] = []
    primary = internal_or_dense == "internal"
    if internal_or_dense != "internal":
        primary = False
        reasons.append("fallback_not_primary_internal_repair")
    if full_dense_initializers:
        primary = False
        reasons.append("full_output_dense_initializer")
    if estimate is not None and estimate >= incumbent_cost:
        primary = False
        reasons.append("estimated_cost_not_below_incumbent")
    if estimate is not None and estimate >= incumbent_cost * 3:
        primary = False
        reasons.append("estimated_cost_far_above_incumbent")
    return {
        "primary_repair": primary,
        "estimated_cost": estimate,
        "incumbent_cost": incumbent_cost,
        "full_dense_initializers": full_dense_initializers,
        "full_volume_intermediates": full_volume_intermediates[:12],
        "reason": "|".join(reasons) if reasons else None,
    }


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


def _repair_diagnostic_labels(pair_diag: dict | None, failure_class: str | None, internal_or_dense: str) -> list[str]:
    labels: list[str] = []
    for failure in (pair_diag or {}).get("failures", []) or []:
        coords = failure.get("first_failing_coordinates") or []
        reasons = [str(reason) for coord in coords for reason in (coord.get("reason") or [])]
        if any("extra_active_channel_0_outside_expected_output_shape" in reason for reason in reasons):
            labels.append("extra_active_black_outside_shape")
        if failure.get("extra_active_channels", 0):
            labels.append("extra_active_channels")
    if (
        internal_or_dense == "internal"
        and failure_class in {"missing_bottom_row", "missing_row", "missing_border"}
        and ("extra_active_black_outside_shape" in labels or "extra_active_channels" in labels)
    ):
        labels.append("selector_too_broad")
    return list(dict.fromkeys(labels))


def _evaluate_repair(
    task_dir: Path,
    model,
    repair_name: str,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    *,
    repair_family: str = "unknown",
    failure_class: str | None = None,
    internal_or_dense: str = "unknown",
    repair_metadata: dict | None = None,
) -> dict:
    repair_metadata = repair_metadata or {}
    repair_chain = repair_metadata.get("repair_chain") or [repair_name]
    cleanup_steps = repair_metadata.get("post_repair_cleanup_steps") or []
    base_row = {
        "repair": repair_name,
        "repair_family": repair_family,
        "failure_class": failure_class,
        "internal_or_dense": internal_or_dense,
        "source_model": repair_metadata.get("source_model"),
        "source_mutation_family": repair_metadata.get("source_mutation_family"),
        "repair_chain": repair_chain,
        "post_repair_cleanup_steps": cleanup_steps,
        "cost_before_cleanup": repair_metadata.get("cost_before_cleanup"),
        "repair_metadata": repair_metadata,
        **_source_provenance_summary(repair_metadata.get("source_mutations") or []),
    }
    try:
        built = safe_build(model, allow_nonstrict_shapes=config.allow_nonstrict_shapes)
        if not built.ok or built.model is None:
            return {**base_row, "ok": False, "error": built.error}
        cost_gate = _repair_cost_gate(built.model, incumbent_cost, internal_or_dense)
        cost_diag = _cost_diagnostics(repair_metadata.get("source_model"), built.model)
        repair_path = out_dir / f"{repair_name}.onnx"
        save_model(built.model, repair_path)
        if not cost_gate.get("primary_repair") and not getattr(config, "keep_high_cost_probes", False):
            return {
                **base_row,
                "repair": repair_name,
                "ok": True,
                "skipped": True,
                "skip_reason": cost_gate.get("reason"),
                "model_path": str(repair_path),
                "model_file_exists": repair_path.exists(),
                "estimated_repair_cost": cost_gate.get("estimated_cost"),
                "cost_after_cleanup": cost_gate.get("estimated_cost") if cleanup_steps else None,
                "accepted_after_cleanup": False if cleanup_steps else None,
                "cost_diagnostics": cost_diag,
                "cost_gate": cost_gate,
                "primary_repair": False,
                "pair_diagnostics": None,
                "repair_diagnostic_labels": [],
                "validation": None,
                "public_wrong": None,
                "arc_gen_wrong": None,
                "fresh_right": None,
                "fresh_total": None,
                "actual_validation_cost": None,
                "accepted": False,
                "repaired": False,
                "near_miss_category": None,
                "delta_points": None,
            }
        pair_diag = _trim_pair_diagnostics(run_pairs(task_dir, repair_path, sources="public,arc", comparison="official_public_arc"))
        validation = validate_model(task_dir, repair_path, config, cache)
        vdict = validation.to_json_dict()
        accepted = is_accepted(vdict, incumbent_cost, config)
        labels = _repair_diagnostic_labels(pair_diag, failure_class, internal_or_dense)
        return {
            **base_row,
            "repair": repair_name,
            "ok": True,
            "model_path": str(repair_path),
            "model_file_exists": repair_path.exists(),
            "estimated_repair_cost": cost_gate.get("estimated_cost"),
            "cost_after_cleanup": vdict.get("cost") if cleanup_steps else None,
            "accepted_after_cleanup": accepted if cleanup_steps else None,
            "cost_diagnostics": cost_diag,
            "cost_gate": cost_gate,
            "primary_repair": bool(cost_gate.get("primary_repair")),
            "pair_diagnostics": pair_diag,
            "repair_diagnostic_labels": labels,
            "validation": vdict,
            "public_wrong": vdict.get("public_wrong"),
            "arc_gen_wrong": vdict.get("arc_gen_wrong"),
            "fresh_right": vdict.get("fresh_right"),
            "fresh_total": vdict.get("fresh_total"),
            "actual_validation_cost": vdict.get("cost"),
            "accepted": accepted,
            "repaired": accepted,
            "near_miss_category": near_miss_category(vdict, incumbent_cost, config),
            "delta_points": None if validation.cost is None else delta_points(incumbent_cost, validation.cost),
        }
    except Exception as exc:
        return {**base_row, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _try_cleanup_repair(
    task_dir: Path,
    model_path: Path,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    repairs: list[dict] = []
    provenance_summary = _source_provenance_summary(source_provenance or [])
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
            "repair_family": "cleanup",
            "failure_class": None,
            "internal_or_dense": "cleanup",
            "source_model": str(model_path),
            "source_mutation_family": None,
            **provenance_summary,
            "repair_metadata": {},
            "estimated_repair_cost": None,
            "cost_gate": {"primary_repair": False, "reason": "cleanup_fallback"},
            "primary_repair": False,
            "ok": True,
            "model_path": str(repair_path),
            "validation": validation.to_json_dict(),
            "public_wrong": validation.public_wrong,
            "arc_gen_wrong": validation.arc_gen_wrong,
            "fresh_right": validation.fresh_right,
            "fresh_total": validation.fresh_total,
            "actual_validation_cost": validation.cost,
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


def _try_internal_selector_extent_repairs(
    task_dir: Path,
    model_path: Path,
    diagnostics: dict,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    repairs: list[dict] = []
    source_provenance = source_provenance or []
    provenance_summary = _source_provenance_summary(source_provenance)
    failures = classify_diagnostics(diagnostics)
    model = load_model(model_path)
    for repair_name, repaired, metadata in _internal_selector_extent_candidates(model, failures, source_provenance=source_provenance):
        failure = metadata.get("failure") or {}
        failure_class = failure.get("kind")
        metadata = {**metadata, **provenance_summary}
        if repaired is None:
            repairs.append(
                {
                    "repair": repair_name,
                    "repair_family": "internal_selector_extent",
                    "failure_class": failure_class,
                    "internal_or_dense": "internal",
                    "source_model": str(model_path),
                    "source_mutation_family": metadata.get("source_mutation_family"),
                    **provenance_summary,
                    "repair_metadata": metadata,
                    "ok": False,
                    "skipped": True,
                    "skip_reason": metadata.get("reason", "no_internal_candidate"),
                    "cost_gate": {"primary_repair": True, "reason": metadata.get("reason")},
                    "primary_repair": True,
                }
            )
            continue
        metadata = {**metadata, "source_model": str(model_path)}
        base_repair = _evaluate_repair(
            task_dir,
            repaired,
            repair_name,
            out_dir,
            config,
            cache,
            incumbent_cost,
            repair_family="internal_selector_extent",
            failure_class=failure_class,
            internal_or_dense="internal",
            repair_metadata=metadata,
        )
        repairs.append(base_repair)
        base_estimated_cost = base_repair.get("estimated_repair_cost")
        for cleanup_name, cleanup_model, cleanup_meta in _post_repair_cleanup_models(repaired, repair_name, base_estimated_cost):
            cleanup_metadata = {**metadata, **cleanup_meta, "source_model": str(model_path)}
            repairs.append(
                _evaluate_repair(
                    task_dir,
                    cleanup_model,
                    cleanup_name,
                    out_dir,
                    config,
                    cache,
                    incumbent_cost,
                    repair_family="internal_selector_extent",
                    failure_class=failure_class,
                    internal_or_dense="internal",
                    repair_metadata=cleanup_metadata,
                )
            )
    return repairs


def _try_mask_repairs(
    task_dir: Path,
    model_path: Path,
    diagnostics: dict,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    repairs: list[dict] = []
    provenance_summary = _source_provenance_summary(source_provenance or [])
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
                repair_family="shape_mask_fallback",
                failure_class="shape_or_no_color",
                internal_or_dense="final_output_symbolic",
                repair_metadata={"source_model": str(model_path), "source_mutation_family": None, **provenance_summary},
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
                repair_family="shape_mask_fallback",
                failure_class="extra_active_black_outside_shape",
                internal_or_dense="dense_final_output",
                repair_metadata={"source_model": str(model_path), "source_mutation_family": None, **provenance_summary},
            )
        )
    return repairs


def _try_tiny_correction_repairs(
    task_dir: Path,
    model_path: Path,
    diagnostics: dict,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    repairs: list[dict] = []
    provenance_summary = _source_provenance_summary(source_provenance or [])
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
                repair_family="tiny_coordinate_fallback",
                failure_class="coordinate_patch",
                internal_or_dense="dense_final_output",
                repair_metadata={"source_model": str(model_path), "source_mutation_family": None, "coordinate_count": len(coords), **provenance_summary},
            )
        )
    return repairs


def _try_symbolic_line_repairs(
    task_dir: Path,
    model_path: Path,
    diagnostics: dict,
    out_dir: Path,
    config,
    cache: ValidationCache,
    incumbent_cost: int,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    repairs: list[dict] = []
    provenance_summary = _source_provenance_summary(source_provenance or [])
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
            repair_family="symbolic_line_fallback",
            failure_class=template["kind"],
            internal_or_dense="final_output_symbolic",
            repair_metadata={"source_model": str(model_path), "source_mutation_family": None, "symbolic_template": template, **provenance_summary},
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
    source_mutations: list[Any] | None = None,
    source_provenance: list[dict] | None = None,
) -> list[dict]:
    diagnostics = diagnostics or run_pairs(task_dir, model_path, sources="public,arc", comparison="official_public_arc")
    normalized_source = _normalize_source_mutations(source_mutations, source_provenance)
    out_dir.mkdir(parents=True, exist_ok=True)
    repairs: list[dict] = []
    for producer in (
        lambda: _try_internal_selector_extent_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost, normalized_source),
        lambda: _try_symbolic_line_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost, normalized_source),
        lambda: _try_mask_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost, normalized_source),
        lambda: _try_tiny_correction_repairs(task_dir, model_path, diagnostics, out_dir, config, cache, incumbent_cost, normalized_source),
        lambda: _try_cleanup_repair(task_dir, model_path, out_dir, config, cache, incumbent_cost, normalized_source),
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
    metadata = {
        "repair_family": repair.get("repair_family"),
        "repair_name": repair.get("repair"),
        "repair_chain": repair.get("repair_chain"),
        "source_mutation_family": repair.get("source_mutation_family"),
        "near_miss_lab_repair": {
            "failure_class": repair.get("failure_class"),
            "internal_or_dense": repair.get("internal_or_dense"),
            "source_model": repair.get("source_model"),
            "actual_validation_cost": repair.get("actual_validation_cost"),
            "primary_repair": repair.get("primary_repair"),
        },
    }
    result = promote_candidate(task_dir, Path(repair["model_path"]), config, cache, incumbent_cost, metadata=metadata)
    if result.row is not None:
        repair["accept_ledger_row"] = result.row
    if result.reason:
        repair["promote_reason"] = result.reason
    return result.promoted


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
