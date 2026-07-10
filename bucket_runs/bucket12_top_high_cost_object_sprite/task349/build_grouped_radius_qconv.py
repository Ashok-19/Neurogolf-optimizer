import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT=os.path.join(os.path.dirname(__file__),'candidate_grouped_radius_qconv.onnx')
nodes=[];inits=[]
def init(name,a,dtype=None):
 a=np.asarray(a,dtype=dtype);inits.append(numpy_helper.from_array(a,name=name));return name
def node(op,ins,outs,**attrs):
 nodes.append(helper.make_node(op,ins,outs,**attrs));return outs[0] if len(outs)==1 else outs

scale=init('scale',np.array(1,np.float32));xz=init('xz',np.array(0,np.uint8));wz=init('wz',np.array(0,np.int8))
zero=init('zero',np.array(0,np.uint8));three=init('three',np.array([3],np.uint8));nine=init('nine',np.array(9,np.uint8));sentinel=init('sentinel',np.array(255,np.uint8))
ch9_idx=init('ch9_idx',np.array([9],np.int64));colors=init('colors',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))

# Exact horizontal-run anchors. Radii 1 and 2 share one amplitude-coded plane:
# width 2 -> code 1 and width 4 -> code 16. Radii 3,4,5 retain exact planes.
W=np.zeros((4,1,1,12),np.int8)
W[0,0,0,0]=-128;W[0,0,0,1]=101;W[0,0,0,3]=15;W[0,0,0,5]=-127
for out,r in enumerate((3,4,5),start=1):
 L=2*r;W[out,0,0,0]=-100;W[out,0,0,1:L+1]=1
 if L+1<12:W[out,0,0,L+1]=-100
hw=init('anchor_w',W);hb=init('anchor_b',np.array([-100,-5,-7,-9],np.int32))

# Nested per-row halo footprints. The global -100 bias separates the shared
# r1/r2 codes: code1 activates the inner footprint via weight103, while code16
# also activates the r2 extension via weight7. With at most six objects, all
# subthreshold r1-extension evidence is at most 6*2*7=84, below the bias.
foot=[]
for r in range(1,6):
 a=np.zeros((11,20),bool)
 a[5-r:5+r+1,15-3*r:15+r]=True
 foot.append(a)
assert [int(x.sum()) for x in foot]==[12,40,84,144,220]
R=np.zeros((1,4,11,20),np.int8)
R[0,0,foot[0]]=103;R[0,0,foot[1]&~foot[0]]=7
for ch,r in enumerate((3,4,5),start=1):R[0,ch,foot[r-1]]=103
rw=init('halo_w',R);rb=init('halo_b',np.array([-100],np.int32))

node('Gather',['input','ch9_idx'],['ch9'],axis=1)
node('Cast',['ch9'],['ch9_u8'],to=TensorProto.UINT8)
node('ReduceMax',['input'],['valid_row_f'],axes=[1,3],keepdims=1)
node('Cast',['valid_row_f'],['valid_row_b'],to=TensorProto.BOOL)
node('Where',['valid_row_b','zero','sentinel'],['row_sent'])
node('Transpose',['row_sent'],['col_sent'],perm=[0,1,3,2])
node('QLinearConv',['ch9_u8','scale','xz','anchor_w','scale','wz','scale','xz','anchor_b'],['anchor_code'],kernel_shape=[1,12],pads=[0,1,0,10])
node('QLinearConv',['anchor_code','scale','xz','halo_w','scale','wz','scale','xz','halo_b'],['halo_raw'],kernel_shape=[11,20],pads=[5,14,5,5])
node('Min',['halo_raw','three'],['halo'])
node('MaxPool',['ch9_u8'],['beam'],kernel_shape=[30,1],pads=[29,0,0,0],strides=[1,1])
node('Mul',['ch9_u8','nine'],['maroon'])
node('Max',['maroon','halo','beam','row_sent','col_sent'],['labels'])
node('Equal',['labels','colors'],['output'])

g=helper.make_graph(nodes,'task349_grouped_radius_qconv',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],initializer=inits)
m=helper.make_model(g,opset_imports=[helper.make_operatorsetid('',17)],producer_name='neurogolf-structural-rewrite');m.ir_version=10
onnx.checker.check_model(m);onnx.save(m,OUT)
print(OUT,'nodes',len(nodes),'inits',len(inits),'bytes',os.path.getsize(OUT))
