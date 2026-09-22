from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT / "ARC-GEN"))
sys.path.insert(0, str(TASK))
import arcgen_task


def solve_np(grid):
    a = np.asarray(grid, dtype=np.uint8)
    pos = np.argwhere(a != 0)
    color = int(a[tuple(pos[0])])
    rows = np.sort(pos[:, 0])
    cols = np.sort(pos[:, 1])
    cr = int(rows[1])
    cc = int(cols[1])
    step = int(rows[2] - cr)
    h, w = a.shape
    d = np.maximum(
        np.abs(np.arange(h)[:, None] - cr),
        np.abs(np.arange(w)[None, :] - cc),
    )
    return np.where(d % step == 0, color, 0).astype(np.uint8).tolist()


def check_np(fresh_n=5000):
    examples = json.loads((TASK / "task.json").read_text())
    for split in ("train", "test", "arc-gen"):
        for i, pair in enumerate(examples.get(split, [])):
            assert solve_np(pair["input"]) == pair["output"], (split, i)
    seen = {
        str(pair["input"])
        for split in ("train", "test", "arc-gen")
        for pair in examples.get(split, [])
    }
    fresh = 0
    for i in range(30000):
        if fresh >= fresh_n:
            break
        random.seed(10000 + i)
        try:
            pair = arcgen_task.generate()
        except Exception:
            continue
        key = str(pair["input"])
        if key in seen:
            continue
        seen.add(key)
        assert solve_np(pair["input"]) == pair["output"], i
        fresh += 1
    assert fresh == fresh_n, fresh
    print(f"numpy exact public/arc-gen + {fresh} fresh")


def init(name, value):
    return numpy_helper.from_array(np.asarray(value), name)


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def build(path: Path):
    nodes = []
    inits = []

    def I(name, value):
        inits.append(init(name, value))
        return name

    def N(op, inputs, output, **attrs):
        nodes.append(helper.make_node(
            op, inputs, [output], name=f"n{len(nodes):03d}_{op}", **attrs
        ))
        return output

    I("one_half_f", np.array(1.5015, np.float32))
    I("pad_u8", np.array(31, np.uint8))
    I("coord_w", (np.arange(30, dtype=np.float32) + np.float32(0.25)) / np.float32(3.0))
    I("coord_u8", np.arange(30, dtype=np.uint8).reshape(1, 1, 1, 30))
    I("bg_index", np.array([[[[0]]]], np.int64))
    I("zero_update", np.array([[[[0.0]]]], np.float32))

    # Exactly two channels occur in the real grid: background and the dynamic
    # drawing color. Spatial ReduceMax recovers channel presence, then one
    # scalar ScatterElements edit clears channel zero. This is cheaper than
    # counting every channel, comparing against three, and casting the result.
    N("ReduceMax", ["input"], "channel_present", axes=[2, 3], keepdims=1)
    N("ScatterElements", ["channel_present", "bg_index", "zero_update"],
      "center_vec", axis=1)

    # The one-hot input contains exactly one positive entry per real cell, so
    # its global sum is N^2. Attribute-form axes avoid charged axis tensors.
    N("Einsum", ["input"], "total_count", equation="nchw->n")
    N("Sqrt", ["total_count"], "size_f")

    # First and second coordinate moments recover center and spacing from the
    # three symmetric markers without a dense occupancy map.
    N("Einsum", ["input", "center_vec", "coord_w"], "row_sum_f",
      equation="nchw,ncij,h->n")
    N("Einsum", ["input", "center_vec", "coord_w"], "col_sum_f",
      equation="nchw,ncij,w->n")
    N("Einsum", ["input", "center_vec", "coord_w", "coord_w"],
      "row_sq_sum_f", equation="nchw,ncij,h,h->n")
    # coord_w is scaled by 1/3, so the first moments are the exact center
    # coordinates and no integer division outputs are needed.
    N("Cast", ["row_sum_f"], "cr_u8", to=TensorProto.UINT8)
    N("Cast", ["col_sum_f"], "cc_u8", to=TensorProto.UINT8)

    # With C=sum(coord/3) and Q=sum((coord/3)^2) over the three symmetric
    # markers, Q=C^2/3+2s^2/9, hence s^2=4.5Q-1.5C^2.
    N("Mul", ["row_sum_f", "row_sum_f"], "center_sq_f")
    N("Sum", ["row_sq_sum_f", "row_sq_sum_f", "row_sq_sum_f"], "three_q_f")
    N("Sub", ["three_q_f", "center_sq_f"], "variance_term_f")
    N("Mul", ["variance_term_f", "one_half_f"], "step_sq_f")
    N("Sqrt", ["step_sq_f"], "step_f")
    N("Cast", ["step_f"], "step_u8", to=TensorProto.UINT8)

    # Unsigned distance arithmetic avoids the unsupported int8 Where path.
    # Max(coord,center)-Min(coord,center) is the exact absolute distance with
    # the same three one-byte vector outputs as Sub+Abs+Cast.
    N("Cast", ["size_f"], "size_u8", to=TensorProto.UINT8)
    N("Less", ["coord_u8", "size_u8"], "valid_1d")
    N("Max", ["coord_u8", "cr_u8"], "row_hi_u8")
    N("Min", ["coord_u8", "cr_u8"], "row_lo_u8")
    N("Sub", ["row_hi_u8", "row_lo_u8"], "dr_u8_1d")
    N("Where", ["valid_1d", "dr_u8_1d", "pad_u8"], "dr_valid_1d")
    N("Transpose", ["dr_valid_1d"], "dr_u8", perm=[0, 1, 3, 2])
    N("Max", ["coord_u8", "cc_u8"], "col_hi_u8")
    N("Min", ["coord_u8", "cc_u8"], "col_lo_u8")
    N("Sub", ["col_hi_u8", "col_lo_u8"], "dc_u8_1d")
    N("Where", ["valid_1d", "dc_u8_1d", "pad_u8"], "dc_u8")

    # Exact Chebyshev ring rule. Prime sentinel 31 is nonzero modulo every
    # legal spacing 2..7, forcing all padded coordinates down the input branch.
    N("Max", ["dr_u8", "dc_u8"], "cheb_u8")
    N("Mod", ["cheb_u8", "step_u8"], "rem_u8", fmod=0)
    N("Cast", ["rem_u8"], "nonring", to=TensorProto.BOOL)

    # Final output memory is free. Off-ring and padded cells reuse input;
    # ring cells receive the recovered dynamic color vector.
    N("Where", ["nonring", "input", "center_vec"], "output")

    graph = helper.make_graph(
        nodes,
        "task137_scaled_moment_einsum_total",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
        ir_version=10,
        producer_name="neurogolf_structural_rewrite",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print(
        "saved", path,
        "nodes", len(nodes),
        "params", sum(int(np.prod(x.dims or [1])) for x in inits),
        "bytes", path.stat().st_size,
    )


if __name__ == "__main__":
    check_np()
    build(HERE / "candidate_scaled_moment_einsum_total.onnx")
