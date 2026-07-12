from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name('int32_sparse.onnx')

def numpy_rule(grid):
    a=np.asarray(grid,dtype=np.int64)
    pts=np.argwhere(a!=0)
    assert len(pts)==2
    (r1,c1),(r2,c2)=pts
    col1=int(a[r1,c1]); col2=int(a[r2,c2])
    dr=np.sign(r2-r1); dc=np.sign(c2-c1)
    sep=abs(r2-r1)+abs(c2-c1)
    half=(sep+3)//2
    out=a.copy()
    for (r,c,color,sign) in [(r1,c1,col1,1),(r2,c2,col2,-1)]:
        for j in range(half-1): out[r+sign*j*dr,c+sign*j*dc]=color
        stem=half-2
        sr=r+sign*stem*dr; sc=c+sign*stem*dc
        for p in (-2,-1,1,2): out[sr+p*dc,sc+p*dr]=color
        ca=half-1
        cr=r+sign*ca*dr; cc=c+sign*ca*dc
        for p in (-2,2): out[cr+p*dc,cc+p*dr]=color
    return out.tolist()

def init(name, arr, dtype=None):
    a=np.asarray(arr,dtype=dtype)
    return numpy_helper.from_array(a,name)

def node(op, ins, outs, **attrs):
    return helper.make_node(op,ins,outs,**attrs)

def build():
    nodes=[]; inits=[]
    inits += [init('cmask',[0]+[1]*9,np.float32)]
    nodes += [
      node('Einsum',['input','cmask'],['rp'],equation='nchw,c->nh'),
      node('Einsum',['input','cmask'],['cp'],equation='nchw,c->nw'),
      node('ArgMax',['rp'],['rf'],axis=1,keepdims=0,select_last_index=0),
      node('ArgMax',['rp'],['rl'],axis=1,keepdims=0,select_last_index=1),
      node('ArgMax',['cp'],['cf'],axis=1,keepdims=0,select_last_index=0),
      node('ArgMax',['cp'],['cl'],axis=1,keepdims=0,select_last_index=1),
    ]
    for x in ['rf','rl','cf','cl']:
      nodes.append(node('Cast',[x],[x+'_i32'],to=TensorProto.INT32))
      nodes.append(node('Cast',[x],[x+'_i8'],to=TensorProto.INT8))
    inits += [init('one_i32',[1],np.int32),init('slice_hw',[2,3],np.int32)]
    for p,r,c in [('a','rf','cf'),('b','rl','cl')]:
      nodes += [
        node('Add',[r+'_i32','one_i32'],[p+'_re']),
        node('Add',[c+'_i32','one_i32'],[p+'_ce']),
        node('Concat',[r+'_i32',c+'_i32'],[p+'_st'],axis=0),
        node('Concat',[p+'_re',p+'_ce'],[p+'_en'],axis=0),
        node('Slice',['input',p+'_st',p+'_en','slice_hw'],[p+'_vals']),
        node('ArgMax',[p+'_vals'],[p+'_ci'],axis=1,keepdims=0),
        node('Cast',[p+'_ci'],[p+'_i8'],to=TensorProto.INT8),
        node('Squeeze',[p+'_i8'],[p+'_s']),
      ]
    inits += [
      init('three_i8',[3],np.int8),init('two_i8',[2],np.int8),init('one_i8',[1],np.int8),
      init('offsets8',np.arange(1,9),np.int8),init('side_perps',[-2,-1,1,2],np.int8),
      init('cap_perps',[-2,2],np.int8),init('true6',[1]*6,np.int8),
      init('axc',[1],np.int64)
    ]
    nodes += [
      node('Sub',['cl_i8','cf_i8'],['dx']), node('Sub',['rl_i8','rf_i8'],['dy']),
      node('Add',['dx','dy'],['sep']), node('Sub',['sep','three_i8'],['so']),
      node('Div',['so','two_i8'],['stem']), node('Add',['stem','one_i8'],['cap_along']),
      node('Div',['dy','sep'],['drow']), node('Div',['dx','sep'],['dcol']),
      node('Less',['offsets8','cap_along'],['lv']), node('Cast',['lv'],['lv8'],to=TensorProto.INT8),
      node('Concat',['lv8','true6'],['sv'],axis=0), node('Unsqueeze',['sv','axc'],['svc']),
    ]
    def axis(prefix, first, last, dmain, dperp):
      ns=[]
      ns += [node('Mul',['offsets8',dmain],[prefix+'la']), node('Add',[first,prefix+'la'],[prefix+'s1line'])]
      ns += [node('Mul',['stem',dmain],[prefix+'sa']), node('Add',[first,prefix+'sa'],[prefix+'s1sc'])]
      ns += [node('Mul',['side_perps',dperp],[prefix+'sp']), node('Add',[prefix+'s1sc',prefix+'sp'],[prefix+'s1stem'])]
      ns += [node('Mul',['cap_along',dmain],[prefix+'ca']), node('Add',[first,prefix+'ca'],[prefix+'s1cc'])]
      ns += [node('Mul',['cap_perps',dperp],[prefix+'cpp']), node('Add',[prefix+'s1cc',prefix+'cpp'],[prefix+'s1cap'])]
      ns += [node('Sub',[last,prefix+'la'],[prefix+'s2line']), node('Sub',[last,prefix+'sa'],[prefix+'s2sc'])]
      ns += [node('Add',[prefix+'s2sc',prefix+'sp'],[prefix+'s2stem']), node('Sub',[last,prefix+'ca'],[prefix+'s2cc'])]
      ns += [node('Add',[prefix+'s2cc',prefix+'cpp'],[prefix+'s2cap'])]
      ns += [node('Concat',[prefix+'s1line',prefix+'s1stem',prefix+'s1cap',prefix+'s2line',prefix+'s2stem',prefix+'s2cap'],[prefix+'shape'],axis=0)]
      ns += [node('Unsqueeze',[prefix+'shape','axc'],[prefix+'u28']), node('Concat',[prefix+'u28',prefix+'u28'],[prefix+'u56'],axis=0)]
      return ns
    nodes += axis('r_','rf_i8','rl_i8','drow','dcol')
    nodes += axis('c_','cf_i8','cl_i8','dcol','drow')
    inits += [
      init('zeros28',np.zeros((28,1),np.int8)),init('batch56',np.zeros((56,1),np.int8)),
      init('updates',np.r_[np.ones(28,np.float32),-np.ones(28,np.float32)])
    ]
    nodes += [
      node('Mul',['a_s','svc'],['sel1']), node('Mul',['b_s','svc'],['sel2']),
      node('Concat',['sel1','sel2','zeros28'],['chans'],axis=0),
      node('Concat',['batch56','chans','r_u56','c_u56'],['idx8'],axis=1),
      node('Cast',['idx8'],['indices'],to=TensorProto.INT32),
      node('ScatterND',['input','indices','updates'],['output'],reduction='add'),
    ]
    graph=helper.make_graph(nodes,'task284_int32_sparse',
      [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
      [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',16)],ir_version=10,producer_name='fresh-rule-builder')
    model=onnx.shape_inference.infer_shapes(model,strict_mode=True,data_prop=True)
    onnx.checker.check_model(model)
    onnx.save(model,OUT)
    print(OUT)

if __name__=='__main__': build()
