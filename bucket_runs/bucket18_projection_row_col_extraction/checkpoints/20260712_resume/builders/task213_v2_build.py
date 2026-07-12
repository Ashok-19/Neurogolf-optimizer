#!/usr/bin/env python3
"""Task213: fixed 0/8/16 probes recover stripe colors; distinctness selects orientation."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper,TensorProto,numpy_helper
OUT=Path(__file__).with_name('candidate.onnx')
N=[];Z=[]
def I(n,a,d=None):Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,x,y,**kw):N.append(helper.make_node(op,x,[y],name=y,**kw))
# Base 55 keeps multiplicity (1..3 probes) below the next color digit.
weights=np.array([0,55,55**2,55**3,55**4,0,55**6,55**7,55**8,55**9],np.float32)
W=np.broadcast_to(weights.reshape(1,10,1,1),(1,10,3,3)).copy()
I('probe_w',W)
# Every 3-row block contains exactly one horizontal stripe. Columns 0,8,16
# guarantee at least one logical, non-box probe. The transpose does the same vertically.
A('Conv',['input','probe_w'],'h_raw',kernel_shape=[3,3],dilations=[1,8],strides=[3,1],pads=[0,0,-9,-13])
A('Conv',['input','probe_w'],'v_raw',kernel_shape=[3,3],dilations=[8,1],strides=[1,3],pads=[0,0,-13,-9])
I('shift2',[2],np.uint8);I('axes23',[2,3],np.int64);I('axes0123',[0,1,2,3],np.int64)
for p in ['h','v']:
 A('Log',[p+'_raw'],p+'_log');A('Cast',[p+'_log'],p+'_code',to=TensorProto.UINT8);A('BitShift',[p+'_code','shift2'],p+'_ids',direction='RIGHT')
# In the wrong orientation every nonzero probe code is the same color; in the
# correct orientation the generator supplies at least three distinct colors.
A('ReduceMax',['h_ids','axes23'],'h_max',keepdims=1)
A('Sub',['h_max','h_ids'],'h_gap');A('Mul',['h_ids','h_gap'],'h_mix')
A('ReduceMax',['h_mix','axes0123'],'h_diverse_u8',keepdims=0)
A('Cast',['h_diverse_u8'],'h_diverse',to=TensorProto.BOOL);A('Not',['h_diverse'],'vertical')
I('zero_u8',0,np.uint8)
# Select one compact line, orient it with a dynamic singleton axis, and
# broadcast against its support vector to form the square. Both possible
# oriented tensors have seven elements; explicit value_info keeps scoring static.
I('sq3',[3],np.int64);I('sq2',[2],np.int64);I('axis2',[2],np.int64);I('axis3',[3],np.int64)
A('Squeeze',['h_ids','sq3'],'h_line');A('Squeeze',['v_ids','sq2'],'v_line')
A('Where',['vertical','v_line','h_line'],'selected_ids');A('Cast',['selected_ids'],'selected_pos',to=TensorProto.BOOL)
A('Where',['vertical','axis2','axis3'],'ids_axis');A('Where',['vertical','axis3','axis2'],'pos_axis')
A('Unsqueeze',['selected_ids','ids_axis'],'ids_oriented');A('Unsqueeze',['selected_pos','pos_axis'],'pos_oriented')
A('Where',['pos_oriented','ids_oriented','zero_u8'],'labels')
A('Mul',['labels','labels'],'label_sq');A('Concat',['label_sq','labels'],'features',axis=1)
# Exact positive quadratic logits; zero labels outside the n x n square stay nonpositive.
I('x_scale',1.,np.float32);I('x_zero',0,np.uint8);I('w_scale',1.,np.float32);I('w_zero',0,np.int8);I('y_scale',1.,np.float32);I('y_zero',0,np.uint8)
QW=np.zeros((10,2,1,1),np.int8);B=np.zeros(10,np.int32)
for c in range(10):QW[c,:,0,0]=[-1,2*c];B[c]=1-c*c
# Black channel must be zero outside, not positive.
B[0]=0
I('q_weights',QW);I('q_bias',B)
A('QLinearConv',['features','x_scale','x_zero','q_weights','w_scale','w_zero','y_scale','y_zero','q_bias'],'output',kernel_shape=[1,1],pads=[0,0,23,23])
vi=[
 helper.make_tensor_value_info('ids_oriented',TensorProto.UINT8,[1,1,7,1]),
 helper.make_tensor_value_info('pos_oriented',TensorProto.BOOL,[1,1,1,7]),
 helper.make_tensor_value_info('labels',TensorProto.UINT8,[1,1,7,7]),
 helper.make_tensor_value_info('label_sq',TensorProto.UINT8,[1,1,7,7]),
 helper.make_tensor_value_info('features',TensorProto.UINT8,[1,2,7,7]),
]
g=helper.make_graph(N,'task213_fixed_probe_stripes',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.UINT8,[1,10,30,30])],Z,value_info=vi)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
