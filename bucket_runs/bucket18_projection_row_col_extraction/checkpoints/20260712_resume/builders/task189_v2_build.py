#!/usr/bin/env python3
"""Task189 fresh structural rewrite: opposite 6x6 pattern crop plus one dilated palette merge."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate.onnx')
N=[]; I=[]
def init(n,a,d=None): I.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def node(op,ins,out,**kw): N.append(helper.make_node(op,ins,[out],name=out,**kw))
# Orientation flags from the cyan divider intersections.
init('h_idx',[[8,0,6]],np.int64); init('v_idx',[[8,6,0]],np.int64)
node('GatherND',['input','h_idx'],'h_f',batch_dims=1); node('Cast',['h_f'],'hflip',to=TensorProto.BOOL)
node('GatherND',['input','v_idx'],'v_f',batch_dims=1); node('Cast',['v_f'],'vflip',to=TensorProto.BOOL)
# Crop the opposite 6x6 pattern exactly as defined by the rule.
init('slice_axes',[1,2,3],np.int32); init('starts_flip',[3,0,0],np.int32); init('starts_normal',[3,3,3],np.int32); init('span',[1,6,6],np.int32)
node('Concat',['vflip','vflip','hflip'],'flip_axes',axis=0)
node('Where',['flip_axes','starts_flip','starts_normal'],'starts')
node('Add',['starts','span'],'ends')
node('Slice',['input','starts','ends','slice_axes'],'pattern_f')
node('Cast',['pattern_f'],'pattern',to=TensorProto.UINT8)
# A 2x2 kernel with dilation 7 samples the four possible palette corners.
# Black, green pattern, and cyan divider channels are excluded, leaving only palette colors.
W=np.zeros((1,10,2,2),np.float32)
for c in range(10):
    if c not in (0,3,8): W[0,c,:,:]=c
init('palette_w',W)
node('Conv',['input','palette_w'],'palette_f',kernel_shape=[2,2],dilations=[7,7],pads=[0,0,-21,-21])
node('Cast',['palette_f'],'palette',to=TensorProto.UINT8)
init('resize_scales',[3.,3.],np.float32)
N.append(helper.make_node('Resize',['palette','','resize_scales'],['palette_big'],name='palette_big',axes=[2,3],coordinate_transformation_mode='asymmetric',mode='nearest',nearest_mode='floor'))
node('Mul',['pattern','palette_big'],'labels')
# Compact exact signed-logit renderer. Outside the 6x6 logical output ConvInteger padding is zero/nonpositive.
node('Mul',['labels','labels'],'label_sq')
init('ones',np.ones((1,1,6,6),np.uint8))
node('Concat',['label_sq','labels','ones'],'features',axis=1)
RW=np.zeros((10,3,1,1),np.int8)
for c in range(10): RW[c,:,0,0]=[-1,2*c,1-c*c]
init('poly_w',RW)
node('ConvInteger',['features','poly_w'],'output',kernel_shape=[1,1],pads=[0,0,24,24])
g=helper.make_graph(N,'task189_dilated_palette',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],I,value_info=[helper.make_tensor_value_info('pattern_f',TensorProto.FLOAT,[1,1,6,6])])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True); onnx.save(m,OUT); print(OUT)
