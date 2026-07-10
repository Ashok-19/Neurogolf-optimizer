#!/usr/bin/env python3
"""Fresh structural rewrite for task245: decode compact labels, shift the red sprite, render."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT=Path(__file__).resolve().parents[3]
TASK=ROOT/'tasks/task245'
OUT=Path(__file__).with_name('candidate.onnx')

def solve_numpy(grid:list[list[int]])->list[list[int]]:
    a=np.array(grid,dtype=np.int64)
    red=a==2; green=a==3
    rr,rc=np.argwhere(red).min(axis=0)
    gr,gc=np.argwhere(green).min(axis=0)
    dr,dc=int(gr-rr),int(gc-rc)
    out=a.copy(); out[red]=0
    for r,c in np.argwhere(red): out[r+dr,c+dc]=2
    return out.tolist()

def init(name,arr): return numpy_helper.from_array(np.asarray(arr),name)
def vi(name,dtype,shape): return helper.make_tensor_value_info(name,dtype,shape)

def build(path=OUT):
    nodes=[]; inits=[]
    # One convolution crops the 10x10 task area and codes outside/black/red/green as 0/1/2/-1.
    w=np.zeros((1,10,2,2),np.float32); w[0,0,0,0]=1; w[0,2,0,0]=2; w[0,3,0,0]=-1
    inits.append(init('code_w',w))
    nodes.append(helper.make_node('Conv',['input','code_w'],['code'],kernel_shape=[2,2],dilations=[20,20]))
    inits.append(init('two_i8',np.array(2,np.int8)))
    nodes += [
      helper.make_node('Cast',['code'],['code_i8'],to=TensorProto.INT8),
      helper.make_node('Equal',['code_i8','two_i8'],['red_b']),
      helper.make_node('Cast',['red_b'],['red'],to=TensorProto.INT8),
      helper.make_node('ReduceMax',['code_i8'],['red_rows'],axes=[3],keepdims=0),
      helper.make_node('ReduceMax',['code_i8'],['red_cols'],axes=[2],keepdims=0),
      helper.make_node('ReduceMin',['code_i8'],['green_rows'],axes=[3],keepdims=0),
      helper.make_node('ReduceMin',['code_i8'],['green_cols'],axes=[2],keepdims=0),
      helper.make_node('ArgMax',['red_rows'],['rr'],axis=2,keepdims=0),
      helper.make_node('ArgMax',['red_cols'],['rc'],axis=2,keepdims=0),
      helper.make_node('ArgMin',['green_rows'],['gr'],axis=2,keepdims=0),
      helper.make_node('ArgMin',['green_cols'],['gc'],axis=2,keepdims=0),
      helper.make_node('Sub',['gr','rr'],['dr64']),
      helper.make_node('Sub',['gc','rc'],['dc64']),
      helper.make_node('Cast',['dr64'],['dr'],to=TensorProto.INT32),
      helper.make_node('Cast',['dc64'],['dc'],to=TensorProto.INT32),
      helper.make_node('Squeeze',['dr','squeeze_axes'],['drs']),
      helper.make_node('Squeeze',['dc','squeeze_axes'],['dcs']),
    ]
    inits.append(init('squeeze_axes',np.array([0,1],np.int64)))
    inits.append(init('vec',np.arange(-1,9,dtype=np.int32)))
    nodes += [
      helper.make_node('Sub',['vec','drs'],['src_r']),
      helper.make_node('Sub',['vec','dcs'],['src_c']),
      helper.make_node('Gather',['red','src_r'],['red_r'],axis=2),
      helper.make_node('Gather',['red_r','src_c'],['red_out'],axis=3),
      helper.make_node('Concat',['code_i8','red_out'],['feat'],axis=1),
    ]
    rw=np.zeros((10,2,1,1),np.int8); rw[0,0,0,0]=1; rw[0,1,0,0]=-2; rw[2,1,0,0]=1; rw[3,0,0,0]=-1
    inits.append(init('render_w',rw))
    nodes.append(helper.make_node('ConvInteger',['feat','render_w'],['output'],pads=[0,0,20,20]))
    graph=helper.make_graph(nodes,'task245_fresh_rewrite',[vi('input',TensorProto.FLOAT,[1,10,30,30])],[vi('output',TensorProto.INT32,[1,10,30,30])],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',14)],ir_version=10)
    onnx.checker.check_model(model,full_check=True); model=onnx.shape_inference.infer_shapes(model,strict_mode=True)
    onnx.save(model,path)

if __name__=='__main__':
    build(); print(OUT)
