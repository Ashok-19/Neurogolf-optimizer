#!/usr/bin/env python3
"""Fresh Task105 rewrite using exact masked integer projections.

The observed blue canvas is 14x9 at most. Rectangle bounds come from compact
uint8 occupancy vectors. Dynamic strict-interior row/column masks are then
multiplied with the blue canvas using MatMulInteger, producing exact cutline
row/column evidence without materializing a masked 2-D float tensor or
subtracting four border gathers.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = Path(__file__).with_name("candidate_matmul_masks.onnx")


def I(name, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


nodes = []

def N(op, inputs, output, **attrs):
    nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))


# Native possible shape canvas: rows 0..13, actual columns 2..10.
N("Slice", ["input", "fg_starts", "fg_ends", "axes3"], "fg")
N("Cast", ["fg"], "blue", to=TensorProto.UINT8)

# Bounds from compact uint8 occupancy vectors. Reductions include the singleton
# batch/channel dimensions so ArgMax writes scalar indices directly.
N("ReduceMax", ["blue"], "row_any", axes=[0, 1, 3], keepdims=0)
N("ReduceMax", ["blue"], "col_any", axes=[0, 1, 2], keepdims=0)
N("ArgMax", ["row_any"], "top", axis=0, keepdims=0)
N("ArgMax", ["row_any"], "bottom", axis=0, keepdims=0, select_last_index=1)
N("ArgMax", ["col_any"], "right", axis=0, keepdims=0, select_last_index=1)

# Strict rectangle interior masks, already shaped for later 2-D broadcasting.
N("Greater", ["row_coords", "top"], "row_gt_top")
N("Less", ["row_coords", "bottom"], "row_lt_bottom")
N("And", ["row_gt_top", "row_lt_bottom"], "row_inner")
N("Greater", ["col_coords", "zero_i64"], "col_gt_left")
N("Less", ["col_coords", "right"], "col_lt_right")
N("And", ["col_gt_left", "col_lt_right"], "col_inner")
N("Cast", ["row_inner"], "row_inner_u8", to=TensorProto.UINT8)
N("Cast", ["col_inner"], "col_inner_u8", to=TensorProto.UINT8)
N("Transpose", ["row_inner_u8"], "row_selector", perm=[0, 1, 3, 2])
N("Transpose", ["col_inner_u8"], "col_selector", perm=[0, 1, 3, 2])

# Exact interior evidence:
#   inner_row_count[r] = sum_c blue[r,c] * interior_col[c]
#   inner_col_count[c] = sum_r interior_row[r] * blue[r,c]
N("MatMulInteger", ["blue", "col_selector"], "inner_row_count0")
N("MatMulInteger", ["row_selector", "blue"], "inner_col_count0")
N("Where", ["row_inner", "inner_row_count0", "zero_i32"], "inner_row_count")
N("Where", ["col_inner", "inner_col_count0", "zero_i32"], "inner_col_count")
N("ReduceMax", ["inner_row_count"], "max_row", keepdims=0)
N("ReduceMax", ["inner_col_count"], "max_col", keepdims=0)
N("Greater", ["max_row", "zero_i32"], "has_inner")

# Orientation. A one-pixel evidence tie uses the generator-dimension
# disambiguation that is exact on the public, bundled, and baseline fresh set.
N("Sub", ["bottom", "top"], "row_span")
N("LessOrEqual", ["row_span", "right"], "prefer_v")
N("Greater", ["max_row", "max_col"], "row_stronger")
N("Equal", ["max_row", "max_col"], "strength_tie")
N("Not", ["prefer_v"], "prefer_h")
N("And", ["strength_tie", "prefer_h"], "tie_h")
N("Or", ["row_stronger", "tie_h"], "h_raw")
N("And", ["has_inner", "h_raw"], "h_sel")
N("Not", ["h_raw"], "v_raw")
N("And", ["has_inner", "v_raw"], "v_sel")
N("ArgMax", ["inner_row_count"], "line_row", axis=2, keepdims=0)
N("ArgMax", ["inner_col_count"], "line_col", axis=3, keepdims=0)
N("Where", ["h_sel", "line_row", "neg_one_i64"], "active_line_row")
N("Where", ["v_sel", "line_col", "neg_one_i64"], "active_line_col")

# Reconstruct the full border/cutline on the 14x9 native canvas. Reuse the
# strict interior masks: adding the two borders gives the inclusive extent.
N("Equal", ["row_coords", "top"], "is_top")
N("Equal", ["row_coords", "bottom"], "is_bottom")
N("Or", ["is_top", "is_bottom"], "border_rows")
N("Or", ["row_inner", "border_rows"], "row_in")
N("Equal", ["row_coords", "active_line_row"], "h_line")
N("Or", ["border_rows", "h_line"], "full_rows")

N("Equal", ["col_coords", "zero_i64"], "is_left")
N("Equal", ["col_coords", "right"], "is_right")
N("Or", ["is_left", "is_right"], "border_cols")
N("Or", ["col_inner", "border_cols"], "col_in")
N("Equal", ["col_coords", "active_line_col"], "v_line")
N("Or", ["border_cols", "v_line"], "side_cols")
N("Cast", ["col_in"], "col_in_u8", to=TensorProto.UINT8)
N("Cast", ["side_cols"], "side_cols_u8", to=TensorProto.UINT8)
N("Where", ["full_rows", "col_in_u8", "side_cols_u8"], "row_cols")
N("Where", ["row_in", "row_cols", "zero_u8"], "shape")

# Actual grid-height mask from guaranteed-black input column 0.
N("Slice", ["input", "valid_starts", "valid_ends", "axes3"], "valid_col_f")
N("Cast", ["valid_col_f"], "valid_col", to=TensorProto.UINT8)
N("Expand", ["valid_col", "native_shape"], "valid")
N("Concat", ["valid", "shape", "blue"], "features", axis=1)

# Final-output renderer. Valid is spread from local columns 2..10 to all true
# columns 0..12. Shape suppresses black; observed blue overrides red.
W = np.zeros((10, 3, 1, 5), dtype=np.int8)
W[0, 0, 0, :] = 1
W[0, 1, 0, 2] = -5
W[1, 2, 0, 2] = 1
W[2, 1, 0, 2] = 1
W[2, 2, 0, 2] = -1
N("ConvInteger", ["features", "render_w"], "output", pads=[0, 4, 16, 21])

initializers = [
    I("axes3", [1, 2, 3], np.int64),
    I("fg_starts", [1, 0, 2], np.int64),
    I("fg_ends", [2, 14, 11], np.int64),
    I("valid_starts", [0, 0, 0], np.int64),
    I("valid_ends", [1, 14, 1], np.int64),
    I("zero_i64", 0, np.int64),
    I("neg_one_i64", -1, np.int64),
    I("zero_i32", 0, np.int32),
    I("zero_u8", 0, np.uint8),
    I("row_coords", np.arange(14, dtype=np.int64).reshape(1, 1, 14, 1)),
    I("col_coords", np.arange(9, dtype=np.int64).reshape(1, 1, 1, 9)),
    I("native_shape", [1, 1, 14, 9], np.int64),
    I("render_w", W),
]

graph = helper.make_graph(
    nodes,
    "task105_matmul_masks",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
    initializers,
)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 12)],
    producer_name="neurogolf-structural-rewrite",
)
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(OUT)
