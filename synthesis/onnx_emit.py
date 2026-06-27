from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from .dsl import Program
from .onnx_templates import (
    emit_bbox_rectangle,
    emit_compact_recolor,
    emit_constant_grid,
    emit_crop_color_bbox_fixed,
    emit_crop_nonzero_bbox_fixed,
    emit_fill_bbox,
    emit_flip_h,
    emit_flip_v,
    emit_identity,
    emit_mask_color,
    emit_rotate90,
    emit_rotate180,
    emit_rotate270,
    emit_static_crop_then_pad,
    emit_static_line,
    emit_static_tile,
)


def _stage_model(step) -> onnx.ModelProto:
    if step.op == "Recolor":
        return emit_compact_recolor({int(k): int(v) for k, v in step.params["map"].items()})
    if step.op == "ReplaceColor":
        old, new = int(step.params["old"]), int(step.params["new"])
        mapping = {c: c for c in range(10)}
        mapping[old] = new
        return emit_compact_recolor(mapping)
    if step.op == "Crop":
        return emit_static_crop_then_pad(tuple(step.params["bbox"]))
    if step.op == "Flip":
        return emit_flip_h() if step.params["axis"] == "h" else emit_flip_v()
    if step.op == "Rotate":
        return {90: emit_rotate90, 180: emit_rotate180, 270: emit_rotate270}[int(step.params["deg"])]()
    if step.op == "CropNonZeroBBox":
        return emit_crop_nonzero_bbox_fixed(int(step.params.get("output_h", 30)), int(step.params.get("output_w", 30)))
    if step.op == "CropColorBBox":
        return emit_crop_color_bbox_fixed(int(step.params["color"]), int(step.params.get("output_h", 30)), int(step.params.get("output_w", 30)))
    if step.op in {"ObjectByColor", "CropBBox", "BoundingBox"} and "color" in step.params:
        return emit_crop_color_bbox_fixed(int(step.params["color"]), int(step.params.get("output_h", 30)), int(step.params.get("output_w", 30)))
    if step.op == "SelectColor":
        return emit_mask_color(int(step.params["color"]))
    if step.op == "DrawRectangle":
        return emit_bbox_rectangle(int(step.params["target_color"]), int(step.params.get("draw_color", step.params["target_color"])))
    if step.op == "FillRectangle":
        return emit_fill_bbox(int(step.params["target_color"]), int(step.params.get("fill_color", step.params["target_color"])))
    if step.op == "DrawLine":
        params = step.params
        if "row" in params or "col" in params:
            return emit_static_line(int(params.get("color", 1)), row=params.get("row"), col=params.get("col"))
    if step.op == "Tile" and "tile" in step.params:
        return emit_static_tile(step.params["tile"])
    raise NotImplementedError(f"ONNX emitter does not support step {step.op}")


def _compose_stage_models(models: list[onnx.ModelProto]) -> onnx.ModelProto:
    nodes = []
    inits = []
    for i, model in enumerate(models):
        prefix = f"stage{i}"
        prev_name = "input" if i == 0 else f"stage{i - 1}_output"
        out_name = "output" if i == len(models) - 1 else f"stage{i}_output"
        value_map = {"input": prev_name, "output": out_name}
        for init in model.graph.initializer:
            value_map[init.name] = f"{prefix}_{init.name}"
        for node in model.graph.node:
            for name in list(node.input) + list(node.output):
                if name and name not in value_map:
                    value_map[name] = f"{prefix}_{name}"
        for init in model.graph.initializer:
            copied = onnx.TensorProto()
            copied.CopyFrom(init)
            copied.name = value_map[init.name]
            inits.append(copied)
        for node in model.graph.node:
            copied = onnx.NodeProto()
            copied.CopyFrom(node)
            copied.name = f"{prefix}_{node.name or node.op_type}"
            copied.input[:] = [value_map.get(name, name) for name in node.input]
            copied.output[:] = [value_map.get(name, name) for name in node.output]
            nodes.append(copied)
    vi_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    vi_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "synth_waterfall_solver", [vi_in], [vi_out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def emit_program(program: Program) -> onnx.ModelProto:
    steps = program.steps
    if len(steps) == 1 and steps[0].op == "Input":
        return emit_identity()
    if len(steps) > 2 and steps[0].op == "Input":
        return _compose_stage_models([_stage_model(step) for step in steps[1:]])
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "Recolor":
        return emit_compact_recolor({int(k): int(v) for k, v in steps[1].params["map"].items()})
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "ReplaceColor":
        old, new = int(steps[1].params["old"]), int(steps[1].params["new"])
        mapping = {c: c for c in range(10)}
        mapping[old] = new
        return emit_compact_recolor(mapping)
    if len(steps) == 1 and steps[0].op == "Constant":
        return emit_constant_grid(steps[0].params["grid"])
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "Crop":
        return emit_static_crop_then_pad(tuple(steps[1].params["bbox"]))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "Flip":
        return emit_flip_h() if steps[1].params["axis"] == "h" else emit_flip_v()
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "Rotate":
        return {90: emit_rotate90, 180: emit_rotate180, 270: emit_rotate270}[int(steps[1].params["deg"])]()
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "CropNonZeroBBox":
        return emit_crop_nonzero_bbox_fixed(int(steps[1].params.get("output_h", 30)), int(steps[1].params.get("output_w", 30)))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "CropColorBBox":
        return emit_crop_color_bbox_fixed(int(steps[1].params["color"]), int(steps[1].params.get("output_h", 30)), int(steps[1].params.get("output_w", 30)))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op in {"ObjectByColor", "CropBBox", "BoundingBox"} and "color" in steps[1].params:
        return emit_crop_color_bbox_fixed(int(steps[1].params["color"]), int(steps[1].params.get("output_h", 30)), int(steps[1].params.get("output_w", 30)))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "SelectColor":
        return emit_mask_color(int(steps[1].params["color"]))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "DrawRectangle":
        return emit_bbox_rectangle(int(steps[1].params["target_color"]), int(steps[1].params.get("draw_color", steps[1].params["target_color"])))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "FillRectangle":
        return emit_fill_bbox(int(steps[1].params["target_color"]), int(steps[1].params.get("fill_color", steps[1].params["target_color"])))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "DrawLine":
        params = steps[1].params
        if "row" in params or "col" in params:
            return emit_static_line(int(params.get("color", 1)), row=params.get("row"), col=params.get("col"))
    if len(steps) == 2 and steps[0].op == "Input" and steps[1].op == "Tile":
        if "tile" in steps[1].params:
            return emit_static_tile(steps[1].params["tile"])
    raise NotImplementedError(f"ONNX emitter does not support {program.describe()}")
