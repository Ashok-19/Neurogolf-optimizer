#!/usr/bin/env python3
"""Fresh task383 rewrite from the boundary-marker rule.

The rectangle is recovered from row/column occupancy. Each hidden stripe has its
one surviving inner-color marker on one of the two outer border bands, so the
stripe rows/columns are read directly from those four border lines. The graph
uses ArgMax(select_last_index=1) for the far rectangle edges and never builds
reversed row/column vectors.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT=Path(__file__).with_name('candidate_direct_boundary_markers.onnx')
N=[]; I=[]
def init(name,a,dtype=None):
    I.append(numpy_helper.from_array(np.asarray(a,dtype=dtype),name)); return name
def node(op,ins,outs,**kw): N.append(helper.make_node(op,ins,outs,**kw))

# Convert one-hot cells to compact labels on the only 24x24 region reachable by
# the generator. Missing padded cells become sentinel 11 through the bias.
w=np.zeros((1,10,2,2),np.float32)
w[0,:,0,0]=np.array([-1,-10,-9,-8,-7,-6,-5,-4,-3,-2],np.float32)
init('cw',w); init('cb',[11.],np.float32)
init('u10',[10],np.uint8); init('u11',[11],np.uint8)
init('one',[1],np.int32); init('two',[2],np.int32)
init('sr',[1,1,24,1],np.int64); init('sc',[1,1,1,24],np.int64)
init('pad',[0,0,0,0,0,0,6,6],np.int64)
init('chv',np.array([10,1,2,3,4,5,6,7,8,9],dtype=np.uint8).reshape(1,10,1,1))

node('Conv',['input','cw','cb'],['color_f'],kernel_shape=[2,2],dilations=[6,6])
node('Cast',['color_f'],['color'],to=TensorProto.UINT8)
# Min label along each line distinguishes rectangle occupancy (<10), actual
# grid extent (<11), and scorer padding (=11).
node('ReduceMin',['color'],['rmin'],axes=[1,3],keepdims=0)
node('ReduceMin',['color'],['cmin'],axes=[1,2],keepdims=0)
node('Less',['rmin','u11'],['ingr_b']); node('Less',['cmin','u11'],['ingc_b'])
node('Less',['rmin','u10'],['rany_b']); node('Less',['cmin','u10'],['cany_b'])
node('Cast',['rany_b'],['rany'],to=TensorProto.UINT8)
node('Cast',['cany_b'],['cany'],to=TensorProto.UINT8)
# Near and far rectangle edges. select_last_index avoids reversed vectors.
node('ArgMax',['rany'],['top64'],axis=1,keepdims=0)
node('ArgMax',['rany'],['bot64'],axis=1,keepdims=0,select_last_index=1)
node('ArgMax',['cany'],['left64'],axis=1,keepdims=0)
node('ArgMax',['cany'],['rig64'],axis=1,keepdims=0,select_last_index=1)
for a in ('top','bot','left','rig'):
    node('Cast',[a+'64'],[a],to=TensorProto.INT32)
node('Add',['top','one'],['top1']); node('Sub',['bot','one'],['bot1'])
node('Add',['left','one'],['left1']); node('Sub',['rig','one'],['rig1'])
node('Add',['top','two'],['top2']); node('Add',['left','two'],['left2'])
# Dynamic outer and inner colors.
node('Gather',['color','top'],['orow'],axis=2); node('Gather',['orow','left'],['outc'],axis=3)
node('Gather',['color','top2'],['irow'],axis=2); node('Gather',['irow','left2'],['innc'],axis=3)
# A vertical stripe has its inner-color marker on the top or bottom border band.
node('Gather',['color','top1'],['gt'],axis=2); node('Gather',['color','bot1'],['gb'],axis=2)
node('Equal',['gt','innc'],['et']); node('Equal',['gb','innc'],['eb']); node('Or',['et','eb'],['colmark'])
# A horizontal stripe has its marker on the left or right border band.
node('Gather',['color','left1'],['gl'],axis=3); node('Gather',['color','rig1'],['gr'],axis=3)
node('Equal',['gl','innc'],['el']); node('Equal',['gr','innc'],['er']); node('Or',['el','er'],['rowmark'])
# Stripe extension is inner color outside the rectangle, outer color through the
# rectangle, and sentinel outside the actual variable-sized grid.
node('Reshape',['rany_b','sr'],['rany_col']); node('Reshape',['ingr_b','sr'],['ingr_col'])
node('Where',['ingr_col','innc','u11'],['cv1']); node('Where',['rany_col','outc','cv1'],['colval'])
node('Reshape',['cany_b','sc'],['cany_row']); node('Reshape',['ingc_b','sc'],['ingc_row'])
node('Where',['ingc_row','innc','u11'],['rv1']); node('Where',['cany_row','outc','rv1'],['rowval'])
node('Where',['colmark','colval','color'],['final0'])
node('Where',['rowmark','rowval','final0'],['final'])
node('Pad',['final','pad','u11'],['padded'])
node('Equal',['padded','chv'],['output'])

g=helper.make_graph(N,'task383_direct_boundary_markers',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],I)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',12)],producer_name='fresh-task383-direct-boundary-markers')
m.ir_version=9
onnx.checker.check_model(m); m=onnx.shape_inference.infer_shapes(m); onnx.save(m,OUT)
print(OUT,OUT.stat().st_size,'nodes',len(N),'params',sum(numpy_helper.to_array(x).size for x in I))
