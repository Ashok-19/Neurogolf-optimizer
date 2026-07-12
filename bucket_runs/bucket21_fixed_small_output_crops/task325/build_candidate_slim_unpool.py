"""Fresh deterministic ONNX rewrite for task325.

The semantic state is the scalar 4-connected component count.  The exact
Euler characteristic calculation includes the only possible enclosed-hole
correction.  Rendering uses six possible diagonal cells rather than a 6x6
square mask.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name('candidate_slim_unpool.onnx')

def init(name, arr): return numpy_helper.from_array(np.asarray(arr), name=name)
def vi(name, dtype, shape): return helper.make_tensor_value_info(name, dtype, shape)
nodes=[]
def N(op, ins, outs, **attrs): nodes.append(helper.make_node(op, ins, outs, name=outs[0], **attrs))

# Exact scalar Euler state: components = V - Eh - Ev + F + H.
N('Einsum',['input','cyan'],['V'],equation='zchw,c->')
N('Einsum',['input','cyan','input','cyan','successor'],['Eh'],equation='zchw,c,zkhv,k,wv->')
N('Einsum',['input','cyan','input','cyan','successor'],['Ev'],equation='zchw,c,zkgw,k,hg->')
N('Einsum',['input','cyan','input','cyan','input','cyan','successor','input','cyan','successor'],['F'],equation='zchw,c,zkhv,k,zlgw,l,hg,zmgv,m,wv->')
N('Einsum',['input','cyan','input','cyan','input','cyan','successor','input','cyan','input','cyan','successor','input','cyan','successor','input','cyan','successor','input','cyan'],['H'],equation='zcad,c,zkae,k,zlaf,l,ab,zmbd,m,znbf,n,bg,zpgd,p,de,zqge,q,ef,zrgf,r->')
N('Sub',['V','Eh'],['euler_1'])
N('Sub',['euler_1','Ev'],['euler_2'])
N('Add',['euler_2','F'],['euler_3'])
N('Add',['euler_3','H'],['component_count'])

# Six possible output diagonal positions.  Only the prefix below count is active.
N('Less',['diag_ids','component_count'],['diag_active'])
N('Cast',['diag_active'],['diag_values'],to=TensorProto.FLOAT16)
N('MaxUnpool',['diag_values','diag_indices','output_shape'],['output'],kernel_shape=[1,1],strides=[1,1])

succ=np.zeros((30,30),np.float32)
for i in range(29): succ[i,i+1]=1
# Cyan is ARC color 8.  Flat index for [0,8,i,i] is 8*900 + 31*i.
idx=(8*900 + 31*np.arange(6,dtype=np.int64)).reshape(1,1,1,6)
inits=[
 init('cyan',np.eye(10,dtype=np.float32)[8]),
 init('successor',succ),
 init('diag_ids',np.arange(6,dtype=np.float32).reshape(1,1,1,6)),
 init('diag_indices',idx),
 init('output_shape',np.array([1,10,30,30],np.int64)),
]
g=helper.make_graph(nodes,'task325_scalar_euler_slim_unpool',[vi('input',TensorProto.FLOAT,[1,10,30,30])],[vi('output',TensorProto.FLOAT16,[1,10,30,30])],inits)
m=helper.make_model(g,ir_version=10,opset_imports=[helper.make_opsetid('',12)])
m=onnx.shape_inference.infer_shapes(m,strict_mode=True)
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
