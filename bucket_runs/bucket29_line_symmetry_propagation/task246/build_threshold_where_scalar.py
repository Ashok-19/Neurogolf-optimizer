#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT=Path(__file__).with_name('candidate_threshold_where_scalar.onnx')

def I(name, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value,dtype=dtype), name)

nodes=[]
def N(op, ins, out, **attrs):
    nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))

# Four scalar coordinate contractions avoid materializing and splitting two
# 2-element coordinate tensors.
N('Einsum',['input','red_sel','row_coord'],'rred',equation='nchw,c,abhx->n')
N('Einsum',['input','green_sel','row_coord'],'rgreen',equation='nchw,c,abhx->n')
N('Einsum',['input','red_sel','col_coord'],'cred',equation='nchw,c,abxw->n')
N('Einsum',['input','green_sel','col_coord'],'cgreen',equation='nchw,c,abxw->n')

N('Min',['rred','rgreen'],'minr')
N('Max',['rred','rgreen'],'maxr')
N('Min',['cred','cgreen'],'minc')
N('Max',['cred','cgreen'],'maxc')

# Row score: red row=2, strict vertical rows=1, all others=0.
N('Equal',['row_coord','rred'],'red_row')
N('Greater',['row_coord','minr'],'row_gt')
N('Less',['row_coord','maxr'],'row_lt')
N('And',['row_gt','row_lt'],'v_rows')
N('Cast',['v_rows'],'v_u8',to=TensorProto.UINT8)
N('Where',['red_row','two_u8','v_u8'],'row_score')

# Column threshold: green col=0, strict between columns=1, elsewhere=2.
N('Equal',['col_coord','cgreen'],'green_col')
N('Greater',['col_coord','minc'],'col_gt')
N('Less',['col_coord','maxc'],'col_lt')
N('And',['col_gt','col_lt'],'h_strict')
N('Where',['h_strict','one_u8','two_u8'],'col_tmp')
N('Where',['green_col','zero_u8','col_tmp'],'col_threshold')

# Nested sets: row_score > col_threshold is exactly the required L path.
N('Greater',['row_score','col_threshold'],'path')
# Preserve the original one-hot input everywhere else; final output memory is free.
N('Where',['path','cyan','input'],'output')

red_sel=np.zeros((10,),np.float32);red_sel[2]=1
green_sel=np.zeros((10,),np.float32);green_sel[3]=1
row=np.arange(30,dtype=np.float32).reshape(1,1,30,1)
col=np.arange(30,dtype=np.float32).reshape(1,1,1,30)
cyan=np.zeros((1,10,1,1),np.float32);cyan[0,8,0,0]=1
inits=[I('red_sel',red_sel),I('green_sel',green_sel),I('row_coord',row),I('col_coord',col),I('zero_u8',0,np.uint8),I('one_u8',1,np.uint8),I('two_u8',2,np.uint8),I('cyan',cyan)]
g=helper.make_graph(nodes,'task246_threshold_where_scalar',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',12)],producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
