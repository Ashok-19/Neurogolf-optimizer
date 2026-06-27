from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _vi(name: str):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 10, 30, 30])


def make_model(nodes, initializers=None) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, "synth_solver", [_vi("input")], [_vi("output")], initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def emit_identity() -> onnx.ModelProto:
    return make_model([helper.make_node("Identity", ["input"], ["output"], name="synth_identity")])


def _slice_channel(color: int, out: str):
    return helper.make_node("Slice", ["input", f"starts_{color}", f"ends_{color}", "axes_channel", "steps_1"], [out], name=f"slice_color_{color}")


def emit_recolor(color_map: dict[int, int]) -> onnx.ModelProto:
    return emit_compact_recolor(color_map)


def emit_compact_recolor(color_map: dict[int, int]) -> onnx.ModelProto:
    nodes = []
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps_1"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_channel"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero_scalar"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts_zero"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends_zero"),
    ]
    nodes.append(helper.make_node("Slice", ["input", "starts_zero", "ends_zero", "axes_channel", "steps_1"], ["zero_src"], name="slice_zero_src"))
    nodes.append(helper.make_node("Mul", ["zero_src", "zero_scalar"], ["zero_ch"], name="make_zero_ch"))
    per_out: list[str] = []
    for out_color in range(10):
        sources = [c for c in range(10) if int(color_map.get(c, c)) == out_color]
        if not sources:
            per_out.append("zero_ch")
            continue
        pieces = []
        for src in sources:
            inits.append(numpy_helper.from_array(np.array([src], dtype=np.int64), name=f"starts_{src}_{out_color}"))
            inits.append(numpy_helper.from_array(np.array([src + 1], dtype=np.int64), name=f"ends_{src}_{out_color}"))
            node_out = f"ch_{src}_to_{out_color}"
            nodes.append(helper.make_node("Slice", ["input", f"starts_{src}_{out_color}", f"ends_{src}_{out_color}", "axes_channel", "steps_1"], [node_out]))
            pieces.append(node_out)
        acc = pieces[0]
        for i, p in enumerate(pieces[1:], 1):
            out = f"sum_{out_color}_{i}"
            nodes.append(helper.make_node("Add", [acc, p], [out]))
            acc = out
        per_out.append(acc)
    nodes.append(helper.make_node("Concat", per_out, ["output"], axis=1, name="recolor_concat"))
    return make_model(nodes, inits)


def _indexing_inits(prefix: str) -> list:
    return [
        numpy_helper.from_array(np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1), name=f"{prefix}_row_base"),
        numpy_helper.from_array(np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30), name=f"{prefix}_col_base"),
        numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), name=f"{prefix}_full_shape"),
        numpy_helper.from_array(np.arange(29, -1, -1, dtype=np.int64), name=f"{prefix}_rev30"),
        numpy_helper.from_array(np.array(29, dtype=np.int64), name=f"{prefix}_idx29"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), name=f"{prefix}_idx0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name=f"{prefix}_one"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=f"{prefix}_zero_float"),
    ]


def _finite_extent_nodes(source: str, prefix: str) -> list:
    return [
        helper.make_node("ReduceMax", [source], [f"{prefix}_row_has"], axes=[1, 3], keepdims=0, name=f"{prefix}_row_has"),
        helper.make_node("ReduceMax", [source], [f"{prefix}_col_has"], axes=[1, 2], keepdims=0, name=f"{prefix}_col_has"),
        helper.make_node("ArgMax", [f"{prefix}_row_has"], [f"{prefix}_r0"], axis=1, keepdims=0, name=f"{prefix}_r0"),
        helper.make_node("ArgMax", [f"{prefix}_col_has"], [f"{prefix}_c0"], axis=1, keepdims=0, name=f"{prefix}_c0"),
        helper.make_node("Gather", [f"{prefix}_row_has", f"{prefix}_rev30"], [f"{prefix}_row_has_rev"], axis=1, name=f"{prefix}_row_rev"),
        helper.make_node("Gather", [f"{prefix}_col_has", f"{prefix}_rev30"], [f"{prefix}_col_has_rev"], axis=1, name=f"{prefix}_col_rev"),
        helper.make_node("ArgMax", [f"{prefix}_row_has_rev"], [f"{prefix}_r_last_rev"], axis=1, keepdims=0, name=f"{prefix}_r_last_rev"),
        helper.make_node("ArgMax", [f"{prefix}_col_has_rev"], [f"{prefix}_c_last_rev"], axis=1, keepdims=0, name=f"{prefix}_c_last_rev"),
        helper.make_node("Sub", [f"{prefix}_idx29", f"{prefix}_r_last_rev"], [f"{prefix}_r_last"], name=f"{prefix}_r_last"),
        helper.make_node("Sub", [f"{prefix}_idx29", f"{prefix}_c_last_rev"], [f"{prefix}_c_last"], name=f"{prefix}_c_last"),
        helper.make_node("Add", [f"{prefix}_r_last", f"{prefix}_one"], [f"{prefix}_height"], name=f"{prefix}_height"),
        helper.make_node("Add", [f"{prefix}_c_last", f"{prefix}_one"], [f"{prefix}_width"], name=f"{prefix}_width"),
    ]


def _expanded_index_nodes(prefix: str) -> list:
    return [
        helper.make_node("Expand", [f"{prefix}_row_base", f"{prefix}_full_shape"], [f"{prefix}_row_rel"], name=f"{prefix}_row_expand"),
        helper.make_node("Expand", [f"{prefix}_col_base", f"{prefix}_full_shape"], [f"{prefix}_col_rel"], name=f"{prefix}_col_expand"),
    ]


def _clip_index_nodes(raw: str, out: str, prefix: str) -> list:
    return [
        helper.make_node("Max", [raw, f"{prefix}_idx0"], [f"{out}_nonneg"], name=f"{out}_nonneg"),
        helper.make_node("Min", [f"{out}_nonneg", f"{prefix}_idx29"], [out], name=f"{out}_clip29"),
    ]


def _gather_2d_nodes(source: str, row_idx: str, col_idx: str, out: str, prefix: str) -> list:
    return [
        helper.make_node("GatherElements", [source, row_idx], [f"{prefix}_row_gathered"], axis=2, name=f"{prefix}_gather_rows"),
        helper.make_node("GatherElements", [f"{prefix}_row_gathered", col_idx], [out], axis=3, name=f"{prefix}_gather_cols"),
    ]


def _valid_mask_nodes(prefix: str, valid_h: str, valid_w: str, out: str) -> list:
    return [
        helper.make_node("Less", [f"{prefix}_row_base", valid_h], [f"{out}_row"], name=f"{out}_row"),
        helper.make_node("Less", [f"{prefix}_col_base", valid_w], [f"{out}_col"], name=f"{out}_col"),
        helper.make_node("And", [f"{out}_row", f"{out}_col"], [out], name=out),
    ]


def _emit_spatial_transform(kind: str) -> onnx.ModelProto:
    prefix = f"xf_{kind}"
    inits = _indexing_inits(prefix)
    nodes = _expanded_index_nodes(prefix)
    nodes.extend(_finite_extent_nodes("input", prefix))
    nodes.extend(
        [
            helper.make_node("Sub", [f"{prefix}_height", f"{prefix}_one"], [f"{prefix}_h_last"], name=f"{prefix}_h_last"),
            helper.make_node("Sub", [f"{prefix}_width", f"{prefix}_one"], [f"{prefix}_w_last"], name=f"{prefix}_w_last"),
        ]
    )
    if kind == "flip_h":
        raw_row = f"{prefix}_row_rel"
        nodes.append(helper.make_node("Sub", [f"{prefix}_w_last", f"{prefix}_col_rel"], [f"{prefix}_raw_col"], name=f"{prefix}_raw_col"))
        raw_col = f"{prefix}_raw_col"
        valid_h, valid_w = f"{prefix}_height", f"{prefix}_width"
    elif kind == "flip_v":
        nodes.append(helper.make_node("Sub", [f"{prefix}_h_last", f"{prefix}_row_rel"], [f"{prefix}_raw_row"], name=f"{prefix}_raw_row"))
        raw_row = f"{prefix}_raw_row"
        raw_col = f"{prefix}_col_rel"
        valid_h, valid_w = f"{prefix}_height", f"{prefix}_width"
    elif kind == "rot180":
        nodes.append(helper.make_node("Sub", [f"{prefix}_h_last", f"{prefix}_row_rel"], [f"{prefix}_raw_row"], name=f"{prefix}_raw_row"))
        nodes.append(helper.make_node("Sub", [f"{prefix}_w_last", f"{prefix}_col_rel"], [f"{prefix}_raw_col"], name=f"{prefix}_raw_col"))
        raw_row, raw_col = f"{prefix}_raw_row", f"{prefix}_raw_col"
        valid_h, valid_w = f"{prefix}_height", f"{prefix}_width"
    elif kind == "rot90":
        raw_row = f"{prefix}_col_rel"
        nodes.append(helper.make_node("Sub", [f"{prefix}_w_last", f"{prefix}_row_rel"], [f"{prefix}_raw_col"], name=f"{prefix}_raw_col"))
        raw_col = f"{prefix}_raw_col"
        valid_h, valid_w = f"{prefix}_width", f"{prefix}_height"
    elif kind == "rot270":
        nodes.append(helper.make_node("Sub", [f"{prefix}_h_last", f"{prefix}_col_rel"], [f"{prefix}_raw_row"], name=f"{prefix}_raw_row"))
        raw_row = f"{prefix}_raw_row"
        raw_col = f"{prefix}_row_rel"
        valid_h, valid_w = f"{prefix}_width", f"{prefix}_height"
    else:
        raise ValueError(kind)
    nodes.extend(_clip_index_nodes(raw_row, f"{prefix}_row_idx", prefix))
    nodes.extend(_clip_index_nodes(raw_col, f"{prefix}_col_idx", prefix))
    nodes.extend(_gather_2d_nodes("input", f"{prefix}_row_idx", f"{prefix}_col_idx", f"{prefix}_gathered", prefix))
    nodes.extend(_valid_mask_nodes(prefix, valid_h, valid_w, f"{prefix}_valid"))
    nodes.append(helper.make_node("Where", [f"{prefix}_valid", f"{prefix}_gathered", f"{prefix}_zero_float"], ["output"], name=f"{prefix}_no_color_mask"))
    return make_model(nodes, inits)


def emit_flip_h() -> onnx.ModelProto:
    return _emit_spatial_transform("flip_h")


def emit_flip_v() -> onnx.ModelProto:
    return _emit_spatial_transform("flip_v")


def emit_rotate180() -> onnx.ModelProto:
    return _emit_spatial_transform("rot180")


def emit_rotate90() -> onnx.ModelProto:
    return _emit_spatial_transform("rot90")


def emit_rotate270() -> onnx.ModelProto:
    return _emit_spatial_transform("rot270")


def onehot_tensor_from_grid(grid) -> np.ndarray:
    out = np.zeros([1, 10, 30, 30], dtype=np.float32)
    for r, row in enumerate(grid[:30]):
        for c, v in enumerate(row[:30]):
            if 0 <= int(v) < 10:
                out[0, int(v), r, c] = 1.0
    return out


def emit_constant_grid(grid) -> onnx.ModelProto:
    tensor = onehot_tensor_from_grid(grid)
    inits = [numpy_helper.from_array(tensor, name="const_output")]
    return make_model([helper.make_node("Identity", ["const_output"], ["output"], name="constant_output")], inits)


def emit_static_crop_then_pad(bbox: tuple[int, int, int, int]) -> onnx.ModelProto:
    r0, c0, r1, c1 = bbox
    h, w = max(0, r1 - r0), max(0, c1 - c0)
    inits = [
        numpy_helper.from_array(np.array([r0, c0], dtype=np.int64), name="crop_starts"),
        numpy_helper.from_array(np.array([r1, c1], dtype=np.int64), name="crop_ends"),
        numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="crop_axes"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="crop_steps"),
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 30 - h, 30 - w], dtype=np.int64), name="pads"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="pad_value"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "crop_axes", "crop_steps"], ["cropped"], name="static_crop"),
        helper.make_node("Pad", ["cropped", "pads", "pad_value"], ["output"], name="crop_pad", mode="constant"),
    ]
    return make_model(nodes, inits)


def _emit_crop_fixed(mask_source: str, output_h: int, output_w: int, extra_nodes, extra_inits) -> onnx.ModelProto:
    prefix = "crop_fixed"
    nodes = list(extra_nodes)
    inits = list(extra_inits)
    inits.extend(_indexing_inits(prefix))
    inits.extend(
        [
            numpy_helper.from_array(np.array(output_h, dtype=np.int64), name=f"{prefix}_max_h"),
            numpy_helper.from_array(np.array(output_w, dtype=np.int64), name=f"{prefix}_max_w"),
        ]
    )
    nodes.extend(_expanded_index_nodes(prefix))
    nodes.extend(_finite_extent_nodes(mask_source, prefix))
    nodes.extend(
        [
            helper.make_node("Sub", [f"{prefix}_r_last", f"{prefix}_r0"], [f"{prefix}_bbox_h0"], name=f"{prefix}_bbox_h0"),
            helper.make_node("Sub", [f"{prefix}_c_last", f"{prefix}_c0"], [f"{prefix}_bbox_w0"], name=f"{prefix}_bbox_w0"),
            helper.make_node("Add", [f"{prefix}_bbox_h0", f"{prefix}_one"], [f"{prefix}_bbox_h"], name=f"{prefix}_bbox_h"),
            helper.make_node("Add", [f"{prefix}_bbox_w0", f"{prefix}_one"], [f"{prefix}_bbox_w"], name=f"{prefix}_bbox_w"),
            helper.make_node("Min", [f"{prefix}_bbox_h", f"{prefix}_max_h"], [f"{prefix}_valid_h"], name=f"{prefix}_valid_h"),
            helper.make_node("Min", [f"{prefix}_bbox_w", f"{prefix}_max_w"], [f"{prefix}_valid_w"], name=f"{prefix}_valid_w"),
            helper.make_node("Add", [f"{prefix}_row_rel", f"{prefix}_r0"], [f"{prefix}_raw_row"], name=f"{prefix}_raw_row"),
            helper.make_node("Add", [f"{prefix}_col_rel", f"{prefix}_c0"], [f"{prefix}_raw_col"], name=f"{prefix}_raw_col"),
        ]
    )
    nodes.extend(_clip_index_nodes(f"{prefix}_raw_row", f"{prefix}_row_idx", prefix))
    nodes.extend(_clip_index_nodes(f"{prefix}_raw_col", f"{prefix}_col_idx", prefix))
    nodes.extend(_gather_2d_nodes("input", f"{prefix}_row_idx", f"{prefix}_col_idx", f"{prefix}_gathered", prefix))
    nodes.extend(_valid_mask_nodes(prefix, f"{prefix}_valid_h", f"{prefix}_valid_w", f"{prefix}_valid"))
    nodes.append(helper.make_node("Where", [f"{prefix}_valid", f"{prefix}_gathered", f"{prefix}_zero_float"], ["output"], name="crop_no_color_outside_valid_area"))
    return make_model(nodes, inits)


def emit_crop_nonzero_bbox_fixed(output_h: int, output_w: int) -> onnx.ModelProto:
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts_fg"),
        numpy_helper.from_array(np.array([10], dtype=np.int64), name="ends_fg"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_channel"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps_1"),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts_fg", "ends_fg", "axes_channel", "steps_1"], ["fg"], name="fg_channels"),
        helper.make_node("ReduceMax", ["fg"], ["mask"], axes=[1], keepdims=1, name="nonzero_mask"),
    ]
    return _emit_crop_fixed("mask", output_h, output_w, nodes, inits)


def emit_crop_color_bbox_fixed(color: int, output_h: int, output_w: int) -> onnx.ModelProto:
    inits = [
        numpy_helper.from_array(np.array([color], dtype=np.int64), name="starts_color"),
        numpy_helper.from_array(np.array([color + 1], dtype=np.int64), name="ends_color"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_channel"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps_1"),
    ]
    nodes = [helper.make_node("Slice", ["input", "starts_color", "ends_color", "axes_channel", "steps_1"], ["mask"], name="color_mask")]
    return _emit_crop_fixed("mask", output_h, output_w, nodes, inits)


def emit_color_select(color: int) -> onnx.ModelProto:
    return emit_recolor({c: color if c == color else 0 for c in range(10)})


def emit_mask_to_output(mask, color: int) -> onnx.ModelProto:
    return emit_constant_grid([[color if v else 0 for v in row] for row in mask])


def emit_two_color_output(mask, fg: int, bg: int) -> onnx.ModelProto:
    return emit_constant_grid([[fg if v else bg for v in row] for row in mask])


def emit_static_tile(tile) -> onnx.ModelProto:
    arr = np.array(tile, dtype=np.int64)
    reps = (30 // max(1, arr.shape[0]) + 1, 30 // max(1, arr.shape[1]) + 1)
    full = np.tile(arr, reps)[:30, :30]
    return emit_constant_grid(full.tolist())


def emit_mask_color(color: int) -> onnx.ModelProto:
    mapping = {c: 0 for c in range(10)}
    mapping[color] = color
    return emit_compact_recolor(mapping)


def _bbox_mask_nodes(target_color: int, fill: bool) -> tuple[list, list, str]:
    """Build a dynamic [1,1,30,30] bool mask for a color bbox border or fill."""
    nodes = []
    inits = [
        numpy_helper.from_array(np.array([target_color], dtype=np.int64), name="bbox_starts_color"),
        numpy_helper.from_array(np.array([target_color + 1], dtype=np.int64), name="bbox_ends_color"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_channel"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps_1"),
        numpy_helper.from_array(np.arange(29, -1, -1, dtype=np.int64), name="rev30"),
        numpy_helper.from_array(np.array(29, dtype=np.int64), name="idx_29"),
        numpy_helper.from_array(np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1), name="row_grid"),
        numpy_helper.from_array(np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30), name="col_grid"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero_scalar"),
    ]
    nodes.extend(
        [
            helper.make_node("Slice", ["input", "bbox_starts_color", "bbox_ends_color", "axes_channel", "steps_1"], ["bbox_src"], name="bbox_target_channel"),
            helper.make_node("ReduceMax", ["bbox_src"], ["row_has"], axes=[1, 3], keepdims=0, name="bbox_row_has"),
            helper.make_node("ReduceMax", ["bbox_src"], ["col_has"], axes=[1, 2], keepdims=0, name="bbox_col_has"),
            helper.make_node("ArgMax", ["row_has"], ["r0"], axis=1, keepdims=0, name="bbox_r0"),
            helper.make_node("ArgMax", ["col_has"], ["c0"], axis=1, keepdims=0, name="bbox_c0"),
            helper.make_node("Gather", ["row_has", "rev30"], ["row_has_rev"], axis=1, name="bbox_row_reverse"),
            helper.make_node("Gather", ["col_has", "rev30"], ["col_has_rev"], axis=1, name="bbox_col_reverse"),
            helper.make_node("ArgMax", ["row_has_rev"], ["r1_rev"], axis=1, keepdims=0, name="bbox_r1_rev"),
            helper.make_node("ArgMax", ["col_has_rev"], ["c1_rev"], axis=1, keepdims=0, name="bbox_c1_rev"),
            helper.make_node("Sub", ["idx_29", "r1_rev"], ["r1"], name="bbox_r1"),
            helper.make_node("Sub", ["idx_29", "c1_rev"], ["c1"], name="bbox_c1"),
            helper.make_node("Greater", ["bbox_src", "zero_scalar"], ["bbox_present_cells"], name="bbox_present_cells"),
            helper.make_node("ReduceMax", ["bbox_src"], ["bbox_any_value"], axes=[1, 2, 3], keepdims=0, name="bbox_any_value"),
            helper.make_node("Greater", ["bbox_any_value", "zero_scalar"], ["bbox_any"], name="bbox_any"),
            helper.make_node("GreaterOrEqual", ["row_grid", "r0"], ["row_ge_r0"], name="bbox_row_ge_r0"),
            helper.make_node("LessOrEqual", ["row_grid", "r1"], ["row_le_r1"], name="bbox_row_le_r1"),
            helper.make_node("And", ["row_ge_r0", "row_le_r1"], ["row_inside"], name="bbox_row_inside"),
            helper.make_node("GreaterOrEqual", ["col_grid", "c0"], ["col_ge_c0"], name="bbox_col_ge_c0"),
            helper.make_node("LessOrEqual", ["col_grid", "c1"], ["col_le_c1"], name="bbox_col_le_c1"),
            helper.make_node("And", ["col_ge_c0", "col_le_c1"], ["col_inside"], name="bbox_col_inside"),
            helper.make_node("And", ["row_inside", "col_inside"], ["bbox_fill_mask_unchecked"], name="bbox_inside"),
        ]
    )
    if fill:
        nodes.append(helper.make_node("And", ["bbox_fill_mask_unchecked", "bbox_any"], ["bbox_mask"], name="bbox_fill_mask"))
    else:
        nodes.extend(
            [
                helper.make_node("Equal", ["row_grid", "r0"], ["row_eq_r0"], name="bbox_row_eq_r0"),
                helper.make_node("Equal", ["row_grid", "r1"], ["row_eq_r1"], name="bbox_row_eq_r1"),
                helper.make_node("Or", ["row_eq_r0", "row_eq_r1"], ["row_edge"], name="bbox_row_edge"),
                helper.make_node("Equal", ["col_grid", "c0"], ["col_eq_c0"], name="bbox_col_eq_c0"),
                helper.make_node("Equal", ["col_grid", "c1"], ["col_eq_c1"], name="bbox_col_eq_c1"),
                helper.make_node("Or", ["col_eq_c0", "col_eq_c1"], ["col_edge"], name="bbox_col_edge"),
                helper.make_node("Or", ["row_edge", "col_edge"], ["bbox_edge"], name="bbox_edge"),
                helper.make_node("And", ["bbox_fill_mask_unchecked", "bbox_edge"], ["bbox_border_unchecked"], name="bbox_border_inside"),
                helper.make_node("And", ["bbox_border_unchecked", "bbox_any"], ["bbox_mask"], name="bbox_border_mask"),
            ]
        )
    return nodes, inits, "bbox_mask"


def _overlay_mask_as_color(nodes: list, inits: list, mask_name: str, color: int) -> onnx.ModelProto:
    inits.extend(
        [
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one_scalar"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="overlay_zero_scalar"),
        ]
    )
    per_out: list[str] = []
    for ch in range(10):
        inits.extend(
            [
                numpy_helper.from_array(np.array([ch], dtype=np.int64), name=f"overlay_start_{ch}"),
                numpy_helper.from_array(np.array([ch + 1], dtype=np.int64), name=f"overlay_end_{ch}"),
            ]
        )
        base = f"overlay_in_{ch}"
        val_name = "one_scalar" if ch == color else "overlay_zero_scalar"
        out = f"overlay_out_{ch}"
        nodes.append(helper.make_node("Slice", ["input", f"overlay_start_{ch}", f"overlay_end_{ch}", "axes_channel", "steps_1"], [base], name=f"overlay_slice_{ch}"))
        nodes.append(helper.make_node("Where", [mask_name, val_name, base], [out], name=f"overlay_where_{ch}"))
        per_out.append(out)
    nodes.append(helper.make_node("Concat", per_out, ["output"], axis=1, name="overlay_concat"))
    return make_model(nodes, inits)


def emit_bbox_rectangle(target_color, draw_color):
    nodes, inits, mask = _bbox_mask_nodes(int(target_color), fill=False)
    return _overlay_mask_as_color(nodes, inits, mask, int(draw_color))


def emit_fill_bbox(target_color, fill_color):
    nodes, inits, mask = _bbox_mask_nodes(int(target_color), fill=True)
    return _overlay_mask_as_color(nodes, inits, mask, int(fill_color))


def emit_static_line(color: int, row: int | None = None, col: int | None = None) -> onnx.ModelProto:
    mask = np.zeros((1, 1, 30, 30), dtype=bool)
    if row is not None:
        mask[:, :, max(0, min(29, int(row))), :] = True
    if col is not None:
        mask[:, :, :, max(0, min(29, int(col)))] = True
    nodes: list = []
    inits = [
        numpy_helper.from_array(mask, name="static_line_mask"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_channel"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps_1"),
    ]
    return _overlay_mask_as_color(nodes, inits, "static_line_mask", int(color))


def emit_line_between_markers(*args, **kwargs): return None
def emit_static_mask(mask, color=1): return emit_mask_to_output(mask, color)
def emit_equal_label_output(label_grid_expr): return emit_constant_grid(label_grid_expr)


def _dtype_size(elem_type: int | None) -> int:
    return {
        TensorProto.BOOL: 1,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.FLOAT16: 2,
        TensorProto.INT32: 4,
        TensorProto.UINT32: 4,
        TensorProto.FLOAT: 4,
        TensorProto.INT64: 8,
        TensorProto.UINT64: 8,
        TensorProto.DOUBLE: 8,
    }.get(elem_type or TensorProto.FLOAT, 4)


def _value_info_numel(vi) -> int | None:
    try:
        shape = vi.type.tensor_type.shape
        total = 1
        for dim in shape.dim:
            if dim.dim_value <= 0:
                return None
            total *= int(dim.dim_value)
        return total
    except Exception:
        return None


def _constant_tensor_elems(node) -> int:
    total = 0
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR:
            total += int(np.prod(attr.t.dims or [1]))
        elif attr.type == onnx.AttributeProto.TENSORS:
            for tensor in attr.tensors:
                total += int(np.prod(tensor.dims or [1]))
    return total


def estimate_model_score_cost(model: onnx.ModelProto) -> int:
    """Estimate NeuroGolf-facing cost from params plus inferred intermediate memory.

    This is deliberately conservative but shape-aware; it separates tiny gather/
    flip models from compiler outputs that allocate tens of full 10x30x30 tensors.
    """
    init_elems = 0
    for init in model.graph.initializer:
        try:
            init_elems += int(np.prod(init.dims or [1]))
        except Exception:
            init_elems += 1
    const_elems = sum(_constant_tensor_elems(node) for node in model.graph.node if node.op_type == "Constant")
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    graph_inputs = {x.name for x in inferred.graph.input}
    graph_outputs = {x.name for x in inferred.graph.output}
    init_names = {x.name for x in inferred.graph.initializer}
    memory = 0
    for vi in list(inferred.graph.value_info):
        if vi.name in graph_inputs or vi.name in graph_outputs or vi.name in init_names:
            continue
        numel = _value_info_numel(vi)
        if numel is None:
            continue
        memory += numel
    # If inference cannot see any intermediates, keep a small node penalty so
    # invalidly opaque graphs do not look free.
    if memory == 0 and inferred.graph.node:
        memory = len(inferred.graph.node) * 16
    return int(init_elems + const_elems + memory)


def estimate_model_cost(model: onnx.ModelProto) -> int:
    return estimate_model_score_cost(model)


def static_shape_preflight(model: onnx.ModelProto) -> tuple[bool, str | None]:
    try:
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    shapes: dict[str, list[int | None]] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not vi.type.HasField("tensor_type") or not vi.type.tensor_type.HasField("shape"):
            shapes[vi.name] = []
            continue
        dims: list[int | None] = []
        for dim in vi.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else None)
        shapes[vi.name] = dims
    for init in inferred.graph.initializer:
        shapes.setdefault(init.name, [int(d) for d in init.dims])
    missing: list[str] = []
    dynamic: list[str] = []
    for node in inferred.graph.node:
        for out in node.output:
            if not out:
                continue
            if out not in shapes:
                missing.append(out)
            elif not shapes[out] or any(d is None for d in shapes[out]):
                dynamic.append(out)
    if missing:
        return False, f"missing value_info for node outputs: {missing[:8]}"
    if dynamic:
        return False, f"dynamic dimensions for node outputs: {dynamic[:8]}"
    return True, None
