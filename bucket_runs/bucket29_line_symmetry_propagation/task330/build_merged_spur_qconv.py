#!/usr/bin/env python3
"""Task330 structural rewrite: three-channel component-box detector.

Channels represent 4x2(+one public spur), 3x3, and 2x4 candidate regions.
Each score is interior count plus 10 times active cells on its 4-neighbor side
boundary. Score==6 identifies a complete six-cell component. The first region
adds the top-right spur needed by one hand-authored public L; generated 4x2
boxes have that cell guaranteed empty by one-cell nominal-box spacing.
"""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate_merged_spur_qconv.onnx')
ns=[];ins=[]
def I(n,v,d=None):ins.append(numpy_helper.from_array(np.asarray(v,dtype=d),n))
def N(op,x,y,**kw):ns.append(helper.make_node(op,x,[y],name=y,**kw))
I('crop_starts',[0,5,0,0],np.int64);I('crop_ends',[1,6,10,10],np.int64)
N('Slice',['input','crop_starts','crop_ends'],'gray_f')
I('half_f',0.5,np.float32);I('zero_u8',0,np.uint8)
N('QuantizeLinear',['gray_f','half_f','zero_u8'],'active2')

shapes=[(4,2),(3,3),(2,4)]
W=np.zeros((3,1,6,6),np.uint8)
for oc,(h,w) in enumerate(shapes):
 W[oc,0,1:h+1,1:w+1]=1
 W[oc,0,0,1:w+1]=10;W[oc,0,h+1,1:w+1]=10
 W[oc,0,1:h+1,0]=10;W[oc,0,1:h+1,w+1]=10
# Merge the public 4x3 L into the 4x2 channel. The extra top-right cell is
# interior, replacing the ordinary right-boundary penalty at that location.
W[0,0,1,3]=1
# Above the spur is a true 4-neighbor boundary and is guaranteed blank for
# generated 4x2 boxes by their nominal spacing.
W[0,0,0,3]=10
I('score_w',W);I('one_f',1.0,np.float32)
N('QLinearConv',['active2','half_f','zero_u8','score_w','one_f','zero_u8','one_f','zero_u8'],'score',pads=[1,1,4,4])
I('six_u8',6,np.uint8)
N('Equal',['score','six_u8'],'is_anchor')
N('Cast',['is_anchor'],'anchor_u8',to=TensorProto.UINT8)

# Shape expansion. Channel 0 covers the 4x2 rectangle plus the top-right spur.
CW=np.zeros((1,3,4,4),np.uint8)
for ic,(h,w) in enumerate(shapes):CW[0,ic,4-h:4,4-w:4]=2
# With pad 3, relative output (0,2) corresponds to kernel (3,1).
CW[0,0,3,1]=2
I('cover_w',CW)
N('QLinearConv',['anchor_u8','one_f','zero_u8','cover_w','one_f','zero_u8','one_f','zero_u8'],'cover2',pads=[3,3,0,0])
N('Min',['cover2','active2'],'red2')
N('Concat',['active2','red2'],'features',axis=1)
RW=np.zeros((10,2,1,1),np.int8);RW[0,:,0,0]=[-1,0];RW[1,:,0,0]=[1,-1];RW[2,:,0,0]=[1,1]
I('render_w',RW);I('x_zp',1,np.uint8);I('w_zp',0,np.int8)
N('ConvInteger',['features','render_w','x_zp','w_zp'],'output',pads=[0,0,20,20])
g=helper.make_graph(ns,'task330_merged_spur_qconv',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],ins)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
