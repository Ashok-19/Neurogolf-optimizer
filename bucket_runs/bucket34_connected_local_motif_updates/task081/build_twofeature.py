#!/usr/bin/env python3
"""Task081 fresh rewrite: exact occupancy+missing-corner state, terminal signed QLinearConv renderer."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper,TensorProto,numpy_helper
OUT=Path(__file__).with_name('candidate_twofeature.onnx')
N=[];Z=[]
def I(n,a,d=None):Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw))
I('st',[0,8,0,0],np.int64);I('en',[1,9,7,7],np.int64);A('Slice',['input','st','en'],'cyan_f')
A('Cast',['cyan_f'],'cyan',to=TensorProto.UINT8)
W=np.zeros((2,1,3,3),np.int8);W[0,0,1,1]=1;W[1,0,:,:]=2;W[1,0,1,1]=0
B=np.array([1,-5],np.int32)
I('hw',W);I('hb',B);I('one_f',1.,np.float32);I('zero_u8',0,np.uint8);I('zero_i8',0,np.int8)
A('QLinearConv',['cyan','one_f','zero_u8','hw','one_f','zero_i8','one_f','zero_u8','hb'],'state',pads=[1,1,1,1])
RW=np.zeros((10,2,1,1),np.int8);RB=np.zeros(10,np.int32)
for ch,w,b in [(0,[-2,-2],3),(1,[-3,2],2),(8,[6,-4],-9)]:
 RW[ch,:,0,0]=w; RB[ch]=b+3*sum(w)
I('rw',RW);I('rb',RB);I('pad_zp',3,np.uint8)
A('QLinearConv',['state','one_f','pad_zp','rw','one_f','zero_i8','one_f','zero_u8','rb'],'output',pads=[0,0,23,23])
g=helper.make_graph(N,'task081_twofeature',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.UINT8,[1,10,30,30])],Z)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',14)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
