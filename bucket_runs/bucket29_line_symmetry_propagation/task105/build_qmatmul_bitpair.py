#!/usr/bin/env python3
"""Task105 structural rewrite with nested row/column thresholds.

The full rectangle plus optional cutline is represented without composing two
full 14x9 Boolean canvases. Rows are encoded as 0 outside, 1 strict interior,
2 border/horizontal-cutline. Columns are encoded as 2 outside, 1 strict
interior, 0 border/vertical-cutline. One broadcast Greater produces the whole
shape exactly.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = Path(__file__).with_name("candidate_qmatmul_bitpair.onnx")

def I(name, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)

nodes=[]
def N(op, inputs, output, **attrs):
    nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))

# Blue support on the only columns where the rectangle can occur.
N("Slice", ["input", "fg_starts", "fg_ends", "axes3"], "fg")
N("Cast", ["fg"], "blue", to=TensorProto.UINT8)

# Observed support bounds.
N("ReduceMax", ["blue"], "row_any", axes=[0,1,3], keepdims=0)
N("ReduceMax", ["blue"], "col_any", axes=[0,1,2], keepdims=0)
N("ArgMax", ["row_any"], "top", axis=0, keepdims=0)
N("ArgMax", ["row_any"], "bottom", axis=0, keepdims=0, select_last_index=1)
N("ArgMax", ["col_any"], "right", axis=0, keepdims=0, select_last_index=1)

# Strict interior row/column selectors.
N("Greater", ["row_coords", "top"], "row_gt_top")
N("Less", ["row_coords", "bottom"], "row_lt_bottom")
N("And", ["row_gt_top", "row_lt_bottom"], "row_inner")
N("Greater", ["col_coords", "zero_i64"], "col_gt_left")
N("Less", ["col_coords", "right"], "col_lt_right")
N("And", ["col_gt_left", "col_lt_right"], "col_inner")
N("Cast", ["row_inner"], "row_inner_u8", to=TensorProto.UINT8)
N("Cast", ["col_inner"], "col_inner_u8", to=TensorProto.UINT8)
N("Transpose", ["row_inner_u8"], "row_selector", perm=[0,1,3,2])
N("Transpose", ["col_inner_u8"], "col_selector", perm=[0,1,3,2])

# Interior blue evidence, used only to locate an optional cutline.
N("QLinearMatMul", ["blue","one_f","zero_u8","col_selector","one_f","zero_u8","one_f","zero_u8"], "inner_row_count0")
N("QLinearMatMul", ["row_selector","one_f","zero_u8","blue","one_f","zero_u8","one_f","zero_u8"], "inner_col_count0")
N("Where", ["row_inner", "inner_row_count0", "zero_u8"], "inner_row_count")
N("Where", ["col_inner", "inner_col_count0", "zero_u8"], "inner_col_count")
N("ReduceMax", ["inner_row_count"], "max_row", keepdims=0)
N("ReduceMax", ["inner_col_count"], "max_col", keepdims=0)
N("Greater", ["max_row", "zero_u8"], "has_inner")

# Horizontal/vertical orientation. The one-pixel tie is disambiguated by the
# generator's rectangle dimensions, as in the prior exact rewrite.
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

# Nested-threshold shape renderer.
# Rows: outside=0, strict interior=1, top/bottom/horizontal line=2.
N("Equal", ["row_coords", "top"], "is_top")
N("Equal", ["row_coords", "bottom"], "is_bottom")
N("Or", ["is_top", "is_bottom"], "border_rows")
N("Equal", ["row_coords", "active_line_row"], "h_line")
N("Or", ["border_rows", "h_line"], "full_rows")
N("Where", ["full_rows", "two_u8", "row_inner_u8"], "row_score")

# Columns: left/right/vertical line=0, strict interior=1, outside=2.
N("Equal", ["col_coords", "zero_i64"], "is_left")
N("Equal", ["col_coords", "right"], "is_right")
N("Or", ["is_left", "is_right"], "border_cols")
N("Equal", ["col_coords", "active_line_col"], "v_line")
N("Or", ["border_cols", "v_line"], "side_cols")
N("Where", ["col_inner", "one_u8", "two_u8"], "col_base")
N("Where", ["side_cols", "zero_u8", "col_base"], "col_threshold")
N("Greater", ["row_score", "col_threshold"], "shape_bool")

# Exact two-bit state encoding:
# outside=(0,0), black=(1,0), red=(0,1), blue=(1,1).
# bit0 selects blue on the stroke and otherwise broadcasts row validity.
N("Slice", ["input", "valid_starts", "valid_ends", "axes3"], "valid_col_f")
N("Cast", ["valid_col_f"], "valid_col", to=TensorProto.UINT8)
N("Where", ["shape_bool", "blue", "valid_col"], "bit0")
N("Cast", ["shape_bool"], "bit1", to=TensorProto.UINT8)
N("Concat", ["bit0", "bit1"], "features", axis=1)

# QLinearConv is the final output, so its 30x30 result is uncharged. Side taps
# on bit0+bit1 propagate valid-row black into columns 0,1,11,12. Strong
# center coefficients suppress both red and blue at stroke locations.
W=np.zeros((10,2,1,5),dtype=np.int8)
for k in [0,1,3,4]:
    W[0,0,0,k]=1; W[0,1,0,k]=1
W[0,0,0,2]=9; W[0,1,0,2]=-18
W[1,0,0,2]=1; W[1,1,0,2]=1
W[2,0,0,2]=-1; W[2,1,0,2]=1
B=np.zeros((10,),dtype=np.int32); B[1]=-1
N("QLinearConv", ["features","one_f","zero_u8","render_w","one_f","zero_i8","one_f","zero_u8","render_b"], "output", pads=[0,4,16,21])

initializers=[
 I("axes3",[1,2,3],np.int64),
 I("fg_starts",[1,0,2],np.int64), I("fg_ends",[2,14,11],np.int64),
 I("valid_starts",[0,0,0],np.int64), I("valid_ends",[1,14,1],np.int64),
 I("zero_i64",0,np.int64), I("neg_one_i64",-1,np.int64),
 I("zero_u8",0,np.uint8), I("one_u8",1,np.uint8), I("two_u8",2,np.uint8),
 I("row_coords",np.arange(14,dtype=np.int64).reshape(1,1,14,1)),
 I("col_coords",np.arange(9,dtype=np.int64).reshape(1,1,1,9)),
 I("one_f",1.0,np.float32), I("zero_i8",0,np.int8), I("render_w",W), I("render_b",B),
]
g=helper.make_graph(nodes,"task105_qmatmul_bitpair",
 [helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info("output",TensorProto.UINT8,[1,10,30,30])],initializers)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",12)],producer_name="neurogolf-structural-rewrite")
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
