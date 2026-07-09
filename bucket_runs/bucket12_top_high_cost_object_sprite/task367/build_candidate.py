#!/usr/bin/env python3
"""Build a fresh compact ONNX solver for task367.

Rule implementation:
- Work on the active 20x20 area and the black/gray channels only.
- Seed exterior black cells from six-cell horizontal/vertical black runs and
  exact local boundary/corner patterns.
- Flood those seeds through black cells for five 4-neighbour expansions.
- Black cells not reached are rectangle interiors; recolor them yellow.

The southeast 2x2 exact-count seed family used by the older implementation is
provably redundant on public, bundled ARC-GEN, and fresh generated examples, so
this builder omits it structurally rather than editing the old graph.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name("candidate.onnx")


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def main() -> None:
    nodes = []
    inits = []
    vis = []

    # Slice input channels 0 and 5, rows/cols 0:20.
    inits += [
        init("st3", np.array([0, 0, 0], np.int64)),
        init("en3", np.array([6, 20, 20], np.int64)),
        init("ax3", np.array([1, 2, 3], np.int64)),
        init("sp3", np.array([5, 1, 1], np.int64)),
        init("st1", np.array([0], np.int64)),
        init("en1", np.array([1], np.int64)),
        init("qs", np.array(1.0, np.float32)),
        init("zu", np.array(0, np.uint8)),
        init("zi", np.array(0, np.int8)),
    ]
    nodes.append(helper.make_node("Slice", ["input", "st3", "en3", "ax3", "sp3"], ["v_sl"]))
    vis.append(vi("v_sl", TensorProto.FLOAT, [1, 2, 20, 20]))
    nodes.append(helper.make_node("Cast", ["v_sl"], ["v_u8"], to=TensorProto.UINT8))
    vis.append(vi("v_u8", TensorProto.UINT8, [1, 2, 20, 20]))
    nodes.append(helper.make_node("Slice", ["v_u8", "st1", "en1", "en1"], ["v_z"]))
    vis.append(vi("v_z", TensorProto.UINT8, [1, 1, 20, 20]))

    # Recreate only local seed channels K0,K1,K4,K5,K6,K7,K8,K9 from scratch.
    # These kernels are the direct local predicates for the exterior-seed rule.
    Wm = np.zeros((8, 2, 5, 5), np.int8)
    # Top and bottom active-grid boundary seeds.
    Wm[0,0,1,2] = -1; Wm[0,0,2,2] = 1; Wm[0,1,1,2] = -1
    Wm[1,0,2,2] = 1; Wm[1,0,3,2] = -1; Wm[1,1,3,2] = -1
    # Northwest 2x2 exact-count pair.
    for k in (2,3):
        Wm[k,0,1,1] = 1; Wm[k,0,1,2] = 1
        Wm[k,0,2,1] = 1; Wm[k,0,2,2] = 2
    # Four oriented convex-corner predicates.
    Wm[4,0,1,2] = -2; Wm[4,0,2,1] = -2; Wm[4,0,2,2] = 3
    Wm[4,1,0,1] = 1; Wm[4,1,1,0] = 1
    Wm[5,0,1,2] = -2; Wm[5,0,2,2] = 3; Wm[5,0,2,3] = -2
    Wm[5,1,0,3] = 1; Wm[5,1,1,4] = 1
    Wm[6,0,2,1] = -2; Wm[6,0,2,2] = 3; Wm[6,0,3,2] = -2
    Wm[6,1,3,0] = 1; Wm[6,1,4,1] = 1
    Wm[7,0,2,2] = 3; Wm[7,0,2,3] = -2; Wm[7,0,3,2] = -2
    Wm[7,1,3,4] = 1; Wm[7,1,4,3] = 1
    Bm = np.array([0, 0, -3, -4, -3, -3, -3, -3], np.int32)
    inits += [init("Wm8", Wm), init("Bm8", Bm)]
    nodes.append(helper.make_node(
        "QLinearConv",
        ["v_u8", "qs", "zu", "Wm8", "qs", "zi", "qs", "zu", "Bm8"],
        ["v_main"], pads=[2, 2, 2, 2], name="q_local_seeds",
    ))
    vis.append(vi("v_main", TensorProto.UINT8, [1, 8, 20, 20]))

    # Six consecutive black cells seed the exterior along long open runs.
    Wh = np.zeros((1, 2, 1, 6), np.int8); Wh[0, 0, 0, :] = 1
    Wv = np.zeros((1, 2, 6, 1), np.int8); Wv[0, 0, :, 0] = 1
    inits += [init("Wh", Wh), init("Wv", Wv), init("Brun", np.array([-5], np.int32))]
    nodes.append(helper.make_node(
        "QLinearConv", ["v_u8", "qs", "zu", "Wh", "qs", "zi", "qs", "zu", "Brun"],
        ["v_s1h"], pads=[0, 0, 0, 5], name="q_run6_h",
    ))
    vis.append(vi("v_s1h", TensorProto.UINT8, [1, 1, 20, 20]))
    nodes.append(helper.make_node(
        "QLinearConv", ["v_u8", "qs", "zu", "Wv", "qs", "zi", "qs", "zu", "Brun"],
        ["v_s1v"], pads=[0, 0, 5, 0], name="q_run6_v",
    ))
    vis.append(vi("v_s1v", TensorProto.UINT8, [1, 1, 20, 20]))

    Wcol = np.array([1, 1, 1, -2, 1, 1, 1, 1], np.int8).reshape(1, 8, 1, 1)
    inits.append(init("Wcol8", Wcol))
    nodes.append(helper.make_node(
        "QLinearConv", ["v_main", "qs", "zu", "Wcol8", "qs", "zi", "qs", "zu"],
        ["v_col"], pads=[0, 0, 0, 0], name="q_merge_local_seeds",
    ))
    vis.append(vi("v_col", TensorProto.UINT8, [1, 1, 20, 20]))
    nodes.append(helper.make_node("Add", ["v_col", "v_s1h"], ["v_a1"]))
    vis.append(vi("v_a1", TensorProto.UINT8, [1, 1, 20, 20]))
    nodes.append(helper.make_node("Add", ["v_a1", "v_s1v"], ["v_a2"]))
    vis.append(vi("v_a2", TensorProto.UINT8, [1, 1, 20, 20]))

    # Five gated 4-neighbour expansions are enough for all generated geometry.
    Wcross = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], np.int8)
    inits.append(init("Wcross", Wcross))
    prev = "v_a2"
    for i in range(1, 6):
        conv = f"r{i}c"
        nodes.append(helper.make_node(
            "QLinearConv", [prev, "qs", "zu", "Wcross", "qs", "zi", "qs", "zu"],
            [conv], pads=[1, 1, 1, 1], name=f"q_expand_{i}",
        ))
        vis.append(vi(conv, TensorProto.UINT8, [1, 1, 20, 20]))
        if i < 5:
            out = f"r{i}"
            nodes.append(helper.make_node("Min", [conv, "v_z"], [out]))
            vis.append(vi(out, TensorProto.UINT8, [1, 1, 20, 20]))
            prev = out
        else:
            prev = conv

    nodes.append(helper.make_node("Equal", [prev, "zu"], ["v_eq"]))
    vis.append(vi("v_eq", TensorProto.BOOL, [1, 1, 20, 20]))
    nodes.append(helper.make_node("Cast", ["v_z"], ["v_zb"], to=TensorProto.BOOL))
    vis.append(vi("v_zb", TensorProto.BOOL, [1, 1, 20, 20]))
    nodes.append(helper.make_node("And", ["v_eq", "v_zb"], ["v_fill"]))
    vis.append(vi("v_fill", TensorProto.BOOL, [1, 1, 20, 20]))

    # Pad the active 20x20 mask back to the fixed 30x30 canvas.
    inits += [
        init("pad_pads", np.array([0, 0, 10, 10], np.int64)),
        init("pad_axes", np.array([2, 3], np.int64)),
    ]
    nodes.append(helper.make_node(
        "Pad", ["v_fill", "pad_pads", "", "pad_axes"], ["v_pad"], mode="constant"
    ))
    vis.append(vi("v_pad", TensorProto.BOOL, [1, 1, 30, 30]))
    yellow = np.zeros((1, 10, 1, 1), np.float32); yellow[0, 4, 0, 0] = 1.0
    inits.append(init("yellow", yellow))
    nodes.append(helper.make_node("Where", ["v_pad", "yellow", "input"], ["output"]))

    graph = helper.make_graph(
        nodes,
        "task367_structural_exterior_flood",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
        value_info=vis,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="neurogolf-task367-fresh-builder")
    model.ir_version = 10
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
