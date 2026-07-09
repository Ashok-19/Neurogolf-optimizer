#!/usr/bin/env python3
"""Fresh task328 rewrite from the corner Voronoi + Chebyshev parity rule.

The winner comparison is expressed as a signed separable inequality:
  top_distance <= bottom_distance
  2*r-(S-1) <= bottom_horizontal_distance-top_horizontal_distance
This avoids materializing the two full Manhattan-distance fields.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT=Path(__file__).with_name('candidate_affine_voronoi.onnx')
N=[]; I=[]
def init(name,a,dtype=None):
    I.append(numpy_helper.from_array(np.asarray(a,dtype=dtype),name)); return name
def node(op,ins,outs,**kw): N.append(helper.make_node(op,ins,outs,**kw))

# Static coordinates for the maximum 18x18 generator extent.
init('aH',np.arange(18,dtype=np.uint8).reshape(1,1,18,1))
init('aW',np.arange(18,dtype=np.uint8).reshape(1,1,1,18))
init('twoH_i8',(2*np.arange(18,dtype=np.int8)).reshape(1,1,18,1))
init('one_u8',np.array(1,np.uint8)); init('zero_u8',np.array(0,np.uint8))
init('ten_u8',np.array(10,np.uint8)); init('p100',np.array(100,np.uint8))
init('colorIdx',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
init('pads',np.array([0,0,0,0,0,0,12,12],np.int64))
init('rshp4',np.array([1,1,1,1],np.int64))
init('ar30',np.arange(30,dtype=np.int64))
init('row0',np.r_[1.,np.zeros(29,dtype=np.float32)].astype(np.float32))
init('one_i64',np.array(1,np.int64))

# Grid size from total one-hot occupancy, and a dynamic selector for the last row/column.
node('ReduceSum',['input'],['total'],keepdims=0)
node('Sqrt',['total'],['S_fs'])
node('Cast',['S_fs'],['S_u8s'],to=TensorProto.UINT8)
node('Reshape',['S_u8s','rshp4'],['S4'])
node('Sub',['S4','one_u8'],['last_u8'])
node('Cast',['S_fs'],['S_i64s'],to=TensorProto.INT64)
node('Sub',['S_i64s','one_i64'],['last_s'])
node('Equal',['ar30','last_s'],['lastB'])
node('Cast',['lastB'],['lastM'],to=TensorProto.FLOAT)

# Read the four dynamic corner colors without materializing a spatial crop.
for name,rsel,csel in [('TL','row0','row0'),('TR','row0','lastM'),('BL','lastM','row0'),('BR','lastM','lastM')]:
    node('Einsum',['input',rsel,csel],[name+'_v'],equation='nchw,h,w->nc')
    node('ArgMax',[name+'_v'],[name+'_am'],axis=1,keepdims=1)
    node('Cast',[name+'_am'],[name+'_ca'],to=TensorProto.UINT8)
    node('Reshape',[name+'_ca','rshp4'],[name+'_c'])

# Inactive corners receive a 100-cell horizontal penalty.
for name in ('TL','TR','BL','BR'):
    node('Greater',[name+'_c','zero_u8'],[name+'_act'])
    node('Where',[name+'_act','zero_u8','p100'],[name+'_pen'])

# Nearest active top corner as a 1-D function of column.
node('Sub',['last_u8','aW'],['cpR'])
node('Add',['aW','TL_pen'],['top_vL'])
node('Add',['cpR','TR_pen'],['top_vR'])
node('Min',['top_vL','top_vR'],['top_mrow'])
node('Equal',['top_vL','top_vR'],['top_rtie'])
node('LessOrEqual',['top_vL','top_vR'],['top_cond'])
node('Where',['top_cond','TL_c','TR_c'],['top_col0'])
node('Where',['top_rtie','zero_u8','top_col0'],['top_col'])

# Nearest active bottom corner as a 1-D function of column.
node('Add',['aW','BL_pen'],['bot_vL'])
node('Add',['cpR','BR_pen'],['bot_vR'])
node('Min',['bot_vL','bot_vR'],['bot_mrow'])
node('Equal',['bot_vL','bot_vR'],['bot_rtie'])
node('LessOrEqual',['bot_vL','bot_vR'],['bot_cond'])
node('Where',['bot_cond','BL_c','BR_c'],['bot_col0'])
node('Where',['bot_rtie','zero_u8','bot_col0'],['bot_col'])

# Compare top and bottom Manhattan candidates without constructing either 18x18 distance field.
# top = r + top_h, bottom = (L-r) + bottom_h
# top <= bottom iff 2r-L <= bottom_h-top_h.
node('Cast',['last_u8'],['last_i8'],to=TensorProto.INT8)
node('Sub',['twoH_i8','last_i8'],['row_balance'])
node('Cast',['top_mrow'],['top_mrow_i8'],to=TensorProto.INT8)
node('Cast',['bot_mrow'],['bot_mrow_i8'],to=TensorProto.INT8)
node('Sub',['bot_mrow_i8','top_mrow_i8'],['col_balance'])
node('LessOrEqual',['row_balance','col_balance'],['Tle'])
node('Equal',['row_balance','col_balance'],['vTie'])

# Winner color. Horizontal and vertical ties are explicitly background.
node('Where',['Tle','top_col','bot_col'],['colorField'])
node('Where',['vTie','zero_u8','colorField'],['colorF2'])

# Chebyshev parity around the winning corner. The selected horizontal minimum is the
# actual horizontal distance because a penalized side can never win against an active side.
node('Sub',['last_u8','aH'],['aB'])
node('Where',['Tle','aH','aB'],['winRp'])
node('Where',['Tle','top_mrow','bot_mrow'],['winCp'])
node('Max',['winRp','winCp'],['cheb'])
node('BitwiseAnd',['cheb','one_u8'],['parOdd'])
node('Sub',['one_u8','parOdd'],['evenU8'])
node('Mul',['colorF2','evenU8'],['coreLabel'])

# Mask the 18x18 work canvas to the actual SxS grid, then render one-hot directly.
node('Less',['aH','S4'],['rIn'])
node('Less',['aW','S4'],['cIn'])
node('And',['rIn','cIn'],['inside'])
node('Where',['inside','coreLabel','ten_u8'],['label'])
node('Pad',['label','pads','ten_u8'],['labelPad'],mode='constant')
node('Equal',['labelPad','colorIdx'],['output'])

inp=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])
out=helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])
g=helper.make_graph(N,'task328_affine_voronoi',[inp],[out],I)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],producer_name='fresh-task328-affine-voronoi')
m.ir_version=9
onnx.checker.check_model(m)
m=onnx.shape_inference.infer_shapes(m)
onnx.save(m,OUT)
print(OUT,OUT.stat().st_size,'nodes',len(N),'params',sum(numpy_helper.to_array(x).size for x in I))
