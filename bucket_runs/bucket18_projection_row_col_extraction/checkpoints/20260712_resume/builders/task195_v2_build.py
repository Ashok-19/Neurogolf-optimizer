from pathlib import Path
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
OUT=Path(__file__).with_name('candidate.onnx')
N=[];Z=[]
def I(n,a,d=None): Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw): N.append(helper.make_node(op,ins,[out],name=out,**kw))
# Sample one representative gray pixel from every 3x3 macro-cell.
I('sample_w',np.eye(10,dtype=np.float32)[5].reshape(1,10,1,1))
A('Conv',['input','sample_w'],'sample_f',kernel_shape=[1,1],strides=[3,3],pads=[-3,-3,-12,-12])
A('Cast',['sample_f'],'sample',to=TensorProto.UINT8)
A('ReduceMax',['sample'],'row_has',axes=[0,1,3],keepdims=0)
A('ReduceMax',['sample'],'col_has',axes=[0,1,2],keepdims=0)
A('ArgMax',['row_has'],'top',axis=0,keepdims=1)
A('ArgMax',['col_has'],'left',axis=0,keepdims=1)
I('three',[3],np.int64);I('axis2',[2],np.int64);I('axis3',[3],np.int64)
A('Add',['top','three'],'bottom');A('Add',['left','three'],'right')
A('Slice',['sample','top','bottom','axis2'],'crop_rows')
A('Slice',['crop_rows','left','right','axis3'],'sprite')
# Encode the 3x3 sprite as {0,2}, then form the Kronecker product in
# [source-row, repeat-row, source-col, repeat-col] order.
I('shift1',1,np.uint8);A('BitShift',['sprite','shift1'],'sprite02',direction='LEFT')
A('Unsqueeze',['sprite02'],'sprite_a',axes=[3,5])
A('Unsqueeze',['sprite02'],'sprite_b',axes=[2,4])
A('Min',['sprite_a','sprite_b'],'outer02')
I('shape99',[1,1,9,9],np.int64);A('Reshape',['outer02','shape99'],'labels02')
# Interior black=0, gray=2, while ConvInteger padding uses zero-point 1 and is neutral.
I('x_zero',1,np.uint8)
W=np.zeros((10,1,1,1),np.int8);W[0,0,0,0]=-1;W[5,0,0,0]=1
I('render_w',W)
A('ConvInteger',['labels02','render_w','x_zero'],'output',kernel_shape=[1,1],pads=[0,0,21,21])
vi=[
 helper.make_tensor_value_info('crop_rows',TensorProto.UINT8,[1,1,3,5]),
 helper.make_tensor_value_info('sprite',TensorProto.UINT8,[1,1,3,3]),
 helper.make_tensor_value_info('sprite02',TensorProto.UINT8,[1,1,3,3]),
 helper.make_tensor_value_info('sprite_a',TensorProto.UINT8,[1,1,3,1,3,1]),
 helper.make_tensor_value_info('sprite_b',TensorProto.UINT8,[1,1,1,3,1,3]),
 helper.make_tensor_value_info('outer02',TensorProto.UINT8,[1,1,3,3,3,3]),
 helper.make_tensor_value_info('labels02',TensorProto.UINT8,[1,1,9,9]),
]
g=helper.make_graph(N,'task195_direct_encoded_outer',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],Z,value_info=vi)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',12)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
