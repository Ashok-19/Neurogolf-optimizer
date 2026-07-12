#!/usr/bin/env python3
"""Task160 exact structural rewrite: occupancy + hollow-box center + terminal renderer."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate_exact_center2.onnx')
N=[];Z=[]
def I(n,a,d=None): Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw): N.append(helper.make_node(op,ins,[out],name=out,**kw))
I('ss',[0,1,0,0],np.int64);I('se',[1,2,10,10],np.int64)
A('Slice',['input','ss','se'],'fg')
A('Cast',['fg'],'o',to=TensorProto.UINT8)
W=np.zeros((2,1,3,3),np.int8)
W[0,0,1,1]=1
W[1,0]=1; W[1,0,1,1]=-8
B=np.array([0,-7],np.int32)
I('hw',W);I('hb',B);I('one',1.,np.float32);I('uz',0,np.uint8);I('iz',0,np.int8);I('ws',[1.,1.],np.float32)
A('QLinearConv',['o','one','uz','hw','ws','iz','one','uz','hb'],'state',pads=[1,1,1,1])
C=np.ones((3,3),np.int8)
R=np.array([[0,1,0],[1,1,1],[0,1,0]],np.int8)
FW=np.zeros((10,2,3,3),np.int8); FB=np.zeros(10,np.int32)
FW[0,0,1,1]=-1; FW[0,1]=C-R; FB[0]=1
FW[1,0,1,1]=1;  FW[1,1]=-C
FW[2,1]=R
I('fw',FW);I('fb',FB)
A('QLinearConv',['state','one','uz','fw','one','iz','one','uz','fb'],'output',pads=[1,1,21,21])
g=helper.make_graph(N,'task160_exact_center2',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.UINT8,[1,10,30,30])],Z)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
