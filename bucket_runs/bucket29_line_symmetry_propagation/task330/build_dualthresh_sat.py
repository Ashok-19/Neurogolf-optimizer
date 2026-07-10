#!/usr/bin/env python3
"""Task330: dual saturated count thresholds; exact-six is their difference."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper,TensorProto,numpy_helper
OUT=Path(__file__).with_name('candidate_dualthresh_sat.onnx')
ns=[];ins=[]
def I(n,v,d=None):ins.append(numpy_helper.from_array(np.asarray(v,dtype=d),n))
def N(op,x,y,**kw):ns.append(helper.make_node(op,x,[y],name=y,**kw))
I('crop_starts',[0,5,0,0],np.int64);I('crop_ends',[1,6,10,10],np.int64)
N('Slice',['input','crop_starts','crop_ends'],'gray_f')
I('active_scale',1/255,np.float32);I('zero_u8',0,np.uint8)
N('QuantizeLinear',['gray_f','active_scale','zero_u8'],'active255')
# Two outputs per bbox: saturated indicators score>=6 and score>=7.
shapes=[(4,2),(3,3),(2,4)]
base=np.zeros((3,1,6,6),np.uint8)
for oc,(h,w) in enumerate(shapes):
 base[oc,0,1:h+1,1:w+1]=1
 base[oc,0,0,1:w+1]=10;base[oc,0,h+1,1:w+1]=10
 base[oc,0,1:h+1,0]=10;base[oc,0,1:h+1,w+1]=10
base[0,0,1,3]=1;base[0,0,0,3]=10
W=np.repeat(base,2,axis=0)
# q=clip(255*(s-5),0,255), q=clip(255*(s-6),0,255).
# bias is scaled by x_scale*w_scale, so -5 -> -1275 and -6 -> -1530.
B=np.tile(np.array([-1275,-1530],np.int32),3)
I('thr_w',W);I('one_f',1.,np.float32);I('thr_scale',1/255,np.float32);I('thr_b',B)
N('QLinearConv',['active255','active_scale','zero_u8','thr_w','one_f','zero_u8','thr_scale','zero_u8','thr_b'],'thresholds',pads=[1,1,4,4])
# Expand exact-six anchors. +A-B is 255 only at score==6. Saturate overlaps.
CW=np.zeros((1,6,4,4),np.int8)
for i,(h,w) in enumerate(shapes):
 for ch,sgn in [(2*i,1),(2*i+1,-1)]: CW[0,ch,4-h:4,4-w:4]=sgn
# Public 4x3 L spur for the first family.
CW[0,0,3,1]=1;CW[0,1,3,1]=-1
I('cover_w',CW);I('zero_i8',0,np.int8)
N('QLinearConv',['thresholds','thr_scale','zero_u8','cover_w','one_f','zero_i8','active_scale','zero_u8'],'cover255',pads=[3,3,0,0])
N('Concat',['active255','cover255'],'features',axis=1)
RW=np.zeros((10,2,1,1),np.int8);RW[0,:,0,0]=[-1,0];RW[1,:,0,0]=[1,-1];RW[2,:,0,0]=[1,1]
I('render_w',RW);I('x_zp',128,np.uint8)
N('ConvInteger',['features','render_w','x_zp','zero_i8'],'output',pads=[0,0,20,20])
g=helper.make_graph(ns,'task330_dualthresh_sat',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],ins)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
