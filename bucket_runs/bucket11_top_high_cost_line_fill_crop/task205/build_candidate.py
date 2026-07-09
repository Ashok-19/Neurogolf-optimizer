#!/usr/bin/env python3
"""Build a fresh structural solver for NeuroGolf task205.

Rule:
1. Identify the rectangle color by row-concentration score.
2. Recover its row/column support using compact 1-D projections.
3. Identify the only non-rectangle color inside that support.
4. Render the cropped rectangle, extending that color across its marked rows/cols.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name('candidate.onnx')

nodes=[]; inits=[]; vis=[]

def init(name, arr, dtype=None):
    a=np.asarray(arr,dtype=dtype)
    inits.append(numpy_helper.from_array(a,name))
    return name

def vi(name, dtype, shape):
    vis.append(helper.make_tensor_value_info(name,dtype,shape))

def node(op, ins, out, **attrs):
    nodes.append(helper.make_node(op,ins,[out],name=out,**attrs)); return out

# Small constants only.
init('depth10', np.array(10,np.int64))
init('oh_values', np.array([0.,1.],np.float32))
init('half', np.array(0.5,np.float32))
init('three_fifths', np.array(0.6,np.float32))
init('thirteen_twentieths', np.array(0.65,np.float32))
init('zero_f', np.array(0.,np.float32))
init('one_f', np.array(1.,np.float32))
init('range30_i32', np.arange(30,dtype=np.int32))
init('last_i32', np.array(29,np.int32))

# Color with the strongest rectangular row concentration.
node('Einsum',['input','input'],'box_score',equation='nchw,ncha->nc'); vi('box_score',TensorProto.FLOAT,[1,10])
node('ArgMax',['box_score'],'box_idx',axis=1,keepdims=0); vi('box_idx',TensorProto.INT64,[1])
node('OneHot',['box_idx','depth10','oh_values'],'box_oh',axis=-1); vi('box_oh',TensorProto.FLOAT,[1,10])

# Column -> row -> refined-column support, matching the generator geometry.
node('Einsum',['input','box_oh'],'col_count',equation='nchw,nc->nw'); vi('col_count',TensorProto.FLOAT,[1,30])
node('ReduceMax',['col_count'],'col_max',axes=[1],keepdims=1); vi('col_max',TensorProto.FLOAT,[1,1])
node('Mul',['col_max','half'],'col_thr'); vi('col_thr',TensorProto.FLOAT,[1,1])
node('Greater',['col_count','col_thr'],'col_seed'); vi('col_seed',TensorProto.BOOL,[1,30])
node('Cast',['col_seed'],'col_seed_f',to=TensorProto.FLOAT); vi('col_seed_f',TensorProto.FLOAT,[1,30])

node('Einsum',['input','box_oh','col_seed_f'],'row_count',equation='nchw,nc,nw->nh'); vi('row_count',TensorProto.FLOAT,[1,30])
node('ReduceMax',['row_count'],'row_max',axes=[1],keepdims=1); vi('row_max',TensorProto.FLOAT,[1,1])
node('Mul',['row_max','three_fifths'],'row_thr'); vi('row_thr',TensorProto.FLOAT,[1,1])
node('Greater',['row_count','row_thr'],'row_mask'); vi('row_mask',TensorProto.BOOL,[1,30])
node('Cast',['row_mask'],'row_mask_f',to=TensorProto.FLOAT); vi('row_mask_f',TensorProto.FLOAT,[1,30])

node('Einsum',['input','box_oh','row_mask_f'],'col_ref_count',equation='nchw,nc,nh->nw'); vi('col_ref_count',TensorProto.FLOAT,[1,30])
node('ReduceMax',['col_ref_count'],'col_ref_max',axes=[1],keepdims=1); vi('col_ref_max',TensorProto.FLOAT,[1,1])
node('Mul',['col_ref_max','thirteen_twentieths'],'col_ref_thr'); vi('col_ref_thr',TensorProto.FLOAT,[1,1])
node('Greater',['col_ref_count','col_ref_thr'],'col_mask'); vi('col_mask',TensorProto.BOOL,[1,30])
node('Cast',['col_mask'],'col_mask_f',to=TensorProto.FLOAT); vi('col_mask_f',TensorProto.FLOAT,[1,30])

# Rectangle bounds.
node('ArgMax',['row_mask_f'],'top64',axis=1,keepdims=0); vi('top64',TensorProto.INT64,[1])
node('ArgMax',['row_mask_f'],'bottom64',axis=1,keepdims=0,select_last_index=1); vi('bottom64',TensorProto.INT64,[1])
node('ArgMax',['col_mask_f'],'left64',axis=1,keepdims=0); vi('left64',TensorProto.INT64,[1])
node('ArgMax',['col_mask_f'],'right64',axis=1,keepdims=0,select_last_index=1); vi('right64',TensorProto.INT64,[1])
for src,dst in [('top64','top'),('bottom64','bottom'),('left64','left'),('right64','right')]:
    node('Cast',[src],dst,to=TensorProto.INT32); vi(dst,TensorProto.INT32,[1])

# The marker color is the most common non-box color inside the rectangle.
node('Einsum',['input','row_mask_f','col_mask_f'],'inside_counts',equation='nchw,nh,nw->nc'); vi('inside_counts',TensorProto.FLOAT,[1,10])
node('Sub',['one_f','box_oh'],'not_box'); vi('not_box',TensorProto.FLOAT,[1,10])
node('Mul',['inside_counts','not_box'],'marker_score'); vi('marker_score',TensorProto.FLOAT,[1,10])
node('ArgMax',['marker_score'],'marker_idx',axis=1,keepdims=0); vi('marker_idx',TensorProto.INT64,[1])
node('OneHot',['marker_idx','depth10','oh_values'],'marker_oh',axis=-1); vi('marker_oh',TensorProto.FLOAT,[1,10])

# Absolute input rows/columns containing marker pixels.
node('Einsum',['input','marker_oh','col_mask_f'],'marker_rows_count',equation='nchw,nc,nw->nh'); vi('marker_rows_count',TensorProto.FLOAT,[1,30])
node('Greater',['marker_rows_count','zero_f'],'marker_rows'); vi('marker_rows',TensorProto.BOOL,[1,30])
node('Einsum',['input','marker_oh','row_mask_f'],'marker_cols_count',equation='nchw,nc,nh->nw'); vi('marker_cols_count',TensorProto.FLOAT,[1,30])
node('Greater',['marker_cols_count','zero_f'],'marker_cols'); vi('marker_cols',TensorProto.BOOL,[1,30])

# Output-coordinate maps. Invalid coordinates are clipped for Gather, then gated.
node('Add',['top','range30_i32'],'row_abs'); vi('row_abs',TensorProto.INT32,[30])
node('Min',['row_abs','last_i32'],'row_idx'); vi('row_idx',TensorProto.INT32,[30])
node('Add',['left','range30_i32'],'col_abs'); vi('col_abs',TensorProto.INT32,[30])
node('Min',['col_abs','last_i32'],'col_idx'); vi('col_idx',TensorProto.INT32,[30])
node('Gather',['marker_rows','row_idx'],'marker_row_g',axis=1); vi('marker_row_g',TensorProto.BOOL,[1,30])
node('Squeeze',['marker_row_g'],'marker_row_out',axes=[0]); vi('marker_row_out',TensorProto.BOOL,[30])
node('Gather',['marker_cols','col_idx'],'marker_col_g',axis=1); vi('marker_col_g',TensorProto.BOOL,[1,30])
node('Squeeze',['marker_col_g'],'marker_col_out',axes=[0]); vi('marker_col_out',TensorProto.BOOL,[30])

node('Sub',['bottom','top'],'height_m1'); vi('height_m1',TensorProto.INT32,[1])
node('Sub',['right','left'],'width_m1'); vi('width_m1',TensorProto.INT32,[1])
node('LessOrEqual',['range30_i32','height_m1'],'valid_rows'); vi('valid_rows',TensorProto.BOOL,[30])
node('LessOrEqual',['range30_i32','width_m1'],'valid_cols'); vi('valid_cols',TensorProto.BOOL,[30])
node('Cast',['valid_rows'],'valid_rows_f',to=TensorProto.FLOAT); vi('valid_rows_f',TensorProto.FLOAT,[30])
node('Cast',['valid_cols'],'valid_cols_f',to=TensorProto.FLOAT); vi('valid_cols_f',TensorProto.FLOAT,[30])

node('Not',['marker_row_out'],'nonmarker_row'); vi('nonmarker_row',TensorProto.BOOL,[30])
node('And',['valid_rows','nonmarker_row'],'valid_plain_rows'); vi('valid_plain_rows',TensorProto.BOOL,[30])
node('Cast',['valid_plain_rows'],'valid_plain_rows_f',to=TensorProto.FLOAT); vi('valid_plain_rows_f',TensorProto.FLOAT,[30])
node('Not',['marker_col_out'],'nonmarker_col'); vi('nonmarker_col',TensorProto.BOOL,[30])
node('And',['valid_cols','nonmarker_col'],'valid_plain_cols'); vi('valid_plain_cols',TensorProto.BOOL,[30])
node('Cast',['valid_plain_cols'],'valid_plain_cols_f',to=TensorProto.FLOAT); vi('valid_plain_cols_f',TensorProto.FLOAT,[30])

# Rank-2 separable final renderer. The final graph output is memory-free to scorer.
node('Sub',['box_oh','marker_oh'],'box_minus_marker'); vi('box_minus_marker',TensorProto.FLOAT,[1,10])
node('Unsqueeze',['marker_oh'],'marker_basis',axes=[1]); vi('marker_basis',TensorProto.FLOAT,[1,1,10])
node('Unsqueeze',['box_minus_marker'],'plain_basis',axes=[1]); vi('plain_basis',TensorProto.FLOAT,[1,1,10])
node('Concat',['marker_basis','plain_basis'],'color_basis',axis=1); vi('color_basis',TensorProto.FLOAT,[1,2,10])
node('Unsqueeze',['valid_rows_f'],'rows_all',axes=[0,1]); vi('rows_all',TensorProto.FLOAT,[1,1,30])
node('Unsqueeze',['valid_plain_rows_f'],'rows_plain',axes=[0,1]); vi('rows_plain',TensorProto.FLOAT,[1,1,30])
node('Concat',['rows_all','rows_plain'],'row_basis',axis=1); vi('row_basis',TensorProto.FLOAT,[1,2,30])
node('Unsqueeze',['valid_cols_f'],'cols_all',axes=[0,1]); vi('cols_all',TensorProto.FLOAT,[1,1,30])
node('Unsqueeze',['valid_plain_cols_f'],'cols_plain',axes=[0,1]); vi('cols_plain',TensorProto.FLOAT,[1,1,30])
node('Concat',['cols_all','cols_plain'],'col_basis',axis=1); vi('col_basis',TensorProto.FLOAT,[1,2,30])
node('Einsum',['color_basis','row_basis','col_basis'],'output',equation='nkc,nkh,nkw->nchw')

inp=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])
out=helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])
g=helper.make_graph(nodes,'task205_structural',[inp],[out],initializer=inits,value_info=vis)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',12)],ir_version=10)
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
