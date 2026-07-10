#!/usr/bin/env python3
"""Task397 structural rewrite: qanchor, exact distinct count, direct label scatter, qpoly final."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper,TensorProto,numpy_helper
OUT=Path(__file__).with_name('candidate_shiftcode_directscatter_qpoly.onnx')
N=[];Z=[]
def I(n,a,d=None):Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw))
# Compact 8x10 color IDs.
W=np.zeros((1,10,2,2),np.float32)
for c in range(10): W[0,c,0,0]=c+1
I('code_w',W);A('Conv',['input','code_w'],'code_f',kernel_shape=[2,2],dilations=[22,20]);A('Cast',['code_f'],'ids',to=TensorProto.UINT8)
# Complete 2x2 nonblack anchors in 7x9.
I('black_code',1,np.uint8);A('Greater',['ids','black_code'],'occ_b');A('Cast',['occ_b'],'occ',to=TensorProto.UINT8);I('sum_w',np.ones((1,1,2,2),np.uint8));I('one_f',1.,np.float32);I('zero_u8',0,np.uint8)
A('QLinearConv',['occ','one_f','zero_u8','sum_w','one_f','zero_u8','one_f','zero_u8'],'qsum')
# Top three anchors; shape [3,1] is retained to avoid later unsqueezes.
I('s631',[63,1],np.int64);A('Reshape',['qsum','s631'],'qflat');A('Cast',['qflat'],'qhalf',to=TensorProto.FLOAT16)
I('k3',[3],np.int64);N.append(helper.make_node('TopK',['qhalf','k3'],['topv','topi'],name='topk',axis=0,largest=1,sorted=0));I('four_h',4,np.float16);A('Equal',['topv','four_h'],'valid')
# 7x9 top-left index -> 8x10 flat color buffer index.
A('Cast',['topi'],'ti32',to=TensorProto.INT32);I('nine',9,np.int32);A('Div',['ti32','nine'],'row');A('Add',['ti32','row'],'base')
I('s80',[80],np.int64);A('Reshape',['ids','s80'],'flat')
# Gather four box colors. Shapes stay [3,1].
I('one',1,np.int32);I('ten',10,np.int32);I('eleven',11,np.int32)
A('Gather',['flat','base'],'ga',axis=0);A('Add',['base','one'],'ib');A('Gather',['flat','ib'],'gb',axis=0);A('Add',['base','ten'],'ic');A('Gather',['flat','ic'],'gc',axis=0);A('Add',['base','eleven'],'id');A('Gather',['flat','id'],'gd',axis=0)
# Distinct count = 1 + new(b) + new(c) + new(d).
A('Equal',['ga','gb'],'eab');A('Not',['eab'],'nx')
A('Equal',['gc','ga'],'eca');A('Equal',['gc','gb'],'ecb');A('Or',['eca','ecb'],'ecseen');A('Not',['ecseen'],'ny')
A('Equal',['gd','ga'],'eda');A('Equal',['gd','gb'],'edb');A('Or',['eda','edb'],'edab');A('Equal',['gd','gc'],'edc');A('Or',['edab','edc'],'edseen');A('Not',['edseen'],'nz')
A('Cast',['nx'],'x32',to=TensorProto.INT32);A('Cast',['ny'],'y32',to=TensorProto.INT32);A('Cast',['nz'],'z32',to=TensorProto.INT32);A('Add',['x32','y32'],'xy');A('Add',['xy','z32'],'newcnt')
# Four possible shadow rows; valid slots only.
I('levels',np.array([[0,1,2,3]],np.int32));A('GreaterOrEqual',['newcnt','levels'],'up4');A('And',['up4','valid'],'upv');A('Concat',['upv','upv'],'up38',axis=1);I('s24',[24],np.int64);A('Reshape',['up38','s24'],'up24');I('green',4,np.uint8)
# Directly scatter green into a flat padded label vector.
I('offs',np.array([[20,30,-60,-50,21,31,-59,-49]],np.int32));A('Add',['base','offs'],'si38');A('Reshape',['si38','s24'],'si')
# Pad compact labels with valid black code 1; preserve inactive TopK slots.
I('pad1d',[0,20],np.int64);N.append(helper.make_node('Pad',['flat','pad1d','black_code'],['label0'],name='label0',mode='constant'))
A('Gather',['label0','si'],'old_updates',axis=0);A('Where',['up24','green','old_updates'],'updates')
N.append(helper.make_node('ScatterElements',['label0','si','updates'],['label1'],name='label1',axis=0))
I('s1110',[1,1,10,10],np.int64);A('Reshape',['label1','s1110'],'label')
# Quadratic signed logits with QLinearConv bias; no all-ones feature plane.
A('Mul',['label','label'],'label2');A('Concat',['label2','label'],'feat',axis=1)
RW=np.zeros((10,2,1,1),np.int8);B=np.zeros(10,np.int32)
for c in range(10):
 k=c+1;RW[c,:,0,0]=[-2,4*k];B[c]=1-2*k*k
I('rw',RW);I('rb',B);I('zero_i8',0,np.int8)
A('QLinearConv',['feat','one_f','zero_u8','rw','one_f','zero_i8','one_f','zero_u8','rb'],'output',pads=[0,0,20,20])
g=helper.make_graph(N,'task397_shiftcode_directscatter_qpoly',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.UINT8,[1,10,30,30])],Z)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
