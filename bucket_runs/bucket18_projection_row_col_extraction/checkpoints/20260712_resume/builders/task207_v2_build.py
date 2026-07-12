from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate.onnx')
N=[];Z=[]
def I(n,a,d=None): Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw): N.append(helper.make_node(op,ins,[out],name=out,**kw))
# For each local 2x2 coordinate, count colored occupancy across four sprites.
# If the shared bit is s and the different bit is d, count=3*s+d, so count mod 3=d.
w=np.zeros((1,10,2,2),np.float32); w[0,1:,:,:]=1
I('count_w',w); I('three',3,np.uint8)
A('Conv',['input','count_w'],'colored_count_f',kernel_shape=[2,2],dilations=[3,3],pads=[0,0,-25,-25])
A('Cast',['colored_count_f'],'colored_count',to=TensorProto.UINT8)
A('Mod',['colored_count','three'],'different_bit')
# The sole nonblack color is recovered globally.
A('GlobalMaxPool',['input'],'present')
A('ArgMax',['present'],'color_i64',axis=1,keepdims=1,select_last_index=1)
A('Cast',['color_i64'],'color',to=TensorProto.UINT8)
A('Mul',['different_bit','color'],'labels')
# Exact quadratic signed logits on the compact 2x2 label image.
A('Mul',['labels','labels'],'label_sq')
I('ones',np.ones((1,1,2,2),np.uint8))
A('Concat',['label_sq','labels','ones'],'features',axis=1)
poly=np.array([[-1,2*c,1-c*c] for c in range(10)],np.int8).reshape(10,3,1,1)
I('poly_w',poly)
A('ConvInteger',['features','poly_w'],'output',kernel_shape=[1,1],pads=[0,0,28,28])
vi=[
 helper.make_tensor_value_info('colored_count_f',TensorProto.FLOAT,[1,1,2,2]),
 helper.make_tensor_value_info('colored_count',TensorProto.UINT8,[1,1,2,2]),
 helper.make_tensor_value_info('different_bit',TensorProto.UINT8,[1,1,2,2]),
 helper.make_tensor_value_info('present',TensorProto.FLOAT,[1,10,1,1]),
 helper.make_tensor_value_info('color_i64',TensorProto.INT64,[1,1,1,1]),
 helper.make_tensor_value_info('color',TensorProto.UINT8,[1,1,1,1]),
 helper.make_tensor_value_info('labels',TensorProto.UINT8,[1,1,2,2]),
 helper.make_tensor_value_info('label_sq',TensorProto.UINT8,[1,1,2,2]),
 helper.make_tensor_value_info('features',TensorProto.UINT8,[1,3,2,2]),
]
g=helper.make_graph(N,'task207_colored_count_mod3',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],Z,value_info=vi)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
