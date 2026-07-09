#!/usr/bin/env python3
"""Build a fresh task204 solver from isolated-side scan events.

The input squares are non-overlapping. Horizontal border rows contain blue runs,
while every interior row contains isolated blue side pixels. A quantized signed
[ -1, +1, -1 ] filter extracts only those isolated side events. Prefix parity
then identifies exact interiors, and asymmetric uint8 MaxPool scans recover the
nearest endpoint parities to choose orange for odd side length and red for even.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT=Path(__file__).with_name('candidate.onnx')
nodes=[]; inits=[]; vis=[]

def init(name,arr): inits.append(numpy_helper.from_array(np.asarray(arr),name))
def vi(name,dtype,shape): vis.append(helper.make_tensor_value_info(name,dtype,shape))
def node(op,ins,out,**attrs): nodes.append(helper.make_node(op,ins,[out],name=out,**attrs))

# Slice channel 1 and the active 20x20 task canvas.
init('sl_st',np.array([0,1,0,0],np.int64)); init('sl_en',np.array([1,2,20,20],np.int64)); init('sl_ax',np.array([0,1,2,3],np.int64))
# Quantization constants.
init('q_scale',np.array([1.],np.float32)); init('q_zero_u8',np.array([0],np.uint8)); init('q_zero_i8',np.array([0],np.int8))
init('side_w',np.array([[[[-1,1,-1]]]],np.int8))
init('prefix_w',np.ones((1,1,1,20),np.uint8))
init('two_u8',np.array(2,np.uint8)); init('one_u8',np.array(1,np.uint8)); init('zero_u8',np.array(0,np.uint8)); init('ten_u8',np.array(10,np.uint8)); init('seven_u8',np.array(7,np.uint8)); init('red_u8',np.array(2,np.uint8))
# Broadcast coordinate codes. Left code parity equals column parity; right code
# parity also equals column parity, while their magnitude selects nearest event.
init('left_codes',np.arange(2,22,dtype=np.uint8).reshape(1,1,1,20))
init('right_codes',(22-np.arange(20,dtype=np.uint8)).reshape(1,1,1,20))
init('palette',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
init('range20_row',np.arange(20,dtype=np.uint8).reshape(1,1,20,1))
init('pad30',np.array([0,0,0,0,0,0,10,10],np.int64))

node('Slice',['input','sl_st','sl_en','sl_ax'],'blue_f'); vi('blue_f',TensorProto.FLOAT,[1,1,20,20])
node('Cast',['blue_f'],'blue_u8',to=TensorProto.UINT8); vi('blue_u8',TensorProto.UINT8,[1,1,20,20])
# Signed quantized filter clips every non-isolated pattern to zero and emits 1
# only for [0,1,0].
node('QLinearConv',['blue_u8','q_scale','q_zero_u8','side_w','q_scale','q_zero_i8','q_scale','q_zero_u8'],'side_u8',kernel_shape=[1,3],pads=[0,1,0,1]); vi('side_u8',TensorProto.UINT8,[1,1,20,20])
# Prefix count of isolated vertical-side events. Odd parity is exactly inside a
# paired interval; gaps between separate boxes have even parity.
node('QLinearConv',['side_u8','q_scale','q_zero_u8','prefix_w','q_scale','q_zero_u8','q_scale','q_zero_u8'],'prefix',kernel_shape=[1,20],pads=[0,19,0,0]); vi('prefix',TensorProto.UINT8,[1,1,20,20])
node('Mod',['prefix','two_u8'],'inside'); vi('inside',TensorProto.UINT8,[1,1,20,20])
# Coordinate-coded nearest left/right side events.
node('Mul',['side_u8','left_codes'],'left_src'); vi('left_src',TensorProto.UINT8,[1,1,20,20])
node('MaxPool',['left_src'],'left_code',kernel_shape=[1,20],pads=[0,19,0,0],strides=[1,1]); vi('left_code',TensorProto.UINT8,[1,1,20,20])
node('Mul',['side_u8','right_codes'],'right_src'); vi('right_src',TensorProto.UINT8,[1,1,20,20])
node('MaxPool',['right_src'],'right_code',kernel_shape=[1,20],pads=[0,0,0,19],strides=[1,1]); vi('right_code',TensorProto.UINT8,[1,1,20,20])
node('Mod',['left_code','two_u8'],'left_parity'); vi('left_parity',TensorProto.UINT8,[1,1,20,20])
node('Mod',['right_code','two_u8'],'right_parity'); vi('right_parity',TensorProto.UINT8,[1,1,20,20])
node('Equal',['left_parity','right_parity'],'odd_length'); vi('odd_length',TensorProto.BOOL,[1,1,20,20])
node('Where',['odd_length','seven_u8','red_u8'],'fill_color'); vi('fill_color',TensorProto.UINT8,[1,1,20,20])
node('Mul',['inside','fill_color'],'fill_map'); vi('fill_map',TensorProto.UINT8,[1,1,20,20])
node('Equal',['blue_u8','one_u8'],'blue_b'); vi('blue_b',TensorProto.BOOL,[1,1,20,20])
node('Where',['blue_b','blue_u8','fill_map'],'label_core'); vi('label_core',TensorProto.UINT8,[1,1,20,20])
# Recover the dynamic square size from compact row occupancy. Invalid cells are
# encoded as label 10 so the final equality renderer emits no active channel.
node('ReduceMax',['input'],'row_present',axes=[1,3],keepdims=1); vi('row_present',TensorProto.FLOAT,[1,1,30,1])
node('ArgMax',['row_present'],'last_row64',axis=2,keepdims=1,select_last_index=1); vi('last_row64',TensorProto.INT64,[1,1,1,1])
node('Cast',['last_row64'],'last_row_u8',to=TensorProto.UINT8); vi('last_row_u8',TensorProto.UINT8,[1,1,1,1])
node('Add',['last_row_u8','one_u8'],'size_u8'); vi('size_u8',TensorProto.UINT8,[1,1,1,1])
node('Greater',['size_u8','range20_row'],'valid_row'); vi('valid_row',TensorProto.BOOL,[1,1,20,1])
node('Where',['valid_row','zero_u8','ten_u8'],'invalid_row'); vi('invalid_row',TensorProto.UINT8,[1,1,20,1])
node('Transpose',['invalid_row'],'invalid_col',perm=[0,1,3,2]); vi('invalid_col',TensorProto.UINT8,[1,1,1,20])
node('Max',['label_core','invalid_row','invalid_col'],'label20'); vi('label20',TensorProto.UINT8,[1,1,20,20])
node('Pad',['label20','pad30','ten_u8'],'label30',mode='constant'); vi('label30',TensorProto.UINT8,[1,1,30,30])
node('Equal',['label30','palette'],'output')

inp=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])
out=helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])
g=helper.make_graph(nodes,'task204_isolated_side_scan',[inp],[out],initializer=inits,value_info=vis)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)],ir_version=10)
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
