import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT=os.path.join(os.path.dirname(__file__),'candidate_nearest_scans.onnx')
N=[]; I=[]
def init(name,a,dtype=None):
    a=np.asarray(a,dtype=dtype); I.append(numpy_helper.from_array(a,name)); return name
def node(op,ins,outs,**kw): N.append(helper.make_node(op,ins,outs,**kw))

# Compact cell labels: valid background=10, colors=1..9, padded/outside=0.
init('label_w',np.array([10,1,2,3,4,5,6,7,8,9],np.float32).reshape(1,10,1,1))
init('zero_u8',np.array(0,np.uint8)); init('ten_u8',np.array(10,np.uint8))
init('fifteen_u8',np.array(15,np.uint8)); init('u255',np.array(255,np.uint8))
# Position dominates color in the high nibble; low nibble carries color.
init('xbase',(16*np.arange(10,dtype=np.uint8)).reshape(1,1,1,10))
init('xrbase',(16*np.arange(9,-1,-1,dtype=np.uint8)).reshape(1,1,1,10))
init('ybase',(16*np.arange(10,dtype=np.uint8)).reshape(1,1,10,1))
init('yrbase',(16*np.arange(9,-1,-1,dtype=np.uint8)).reshape(1,1,10,1))
# Line-color read and 3x rendering constants.
init('sl_st',[2,2],np.int64); init('sl_en',[3,3],np.int64); init('sl_ax',[2,3],np.int64)
init('v_st',[1],np.int64); init('v_en',[10],np.int64); init('v_ax',[3],np.int64)
init('h_st',[1],np.int64); init('h_en',[10],np.int64); init('h_ax',[2],np.int64)
init('v_pad',[0,0,0,0,0,0,0,1],np.int64); init('h_pad',[0,0,0,0,0,0,1,0],np.int64)
init('color_ids',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))

node('Conv',['input','label_w'],['lab_f'],kernel_shape=[1,1],strides=[3,3])
node('Cast',['lab_f'],['lab'],to=TensorProto.UINT8)
node('Greater',['lab','zero_u8'],['valid'])
node('Less',['lab','ten_u8'],['lt10'])
node('And',['valid','lt10'],['fg'])
node('Where',['fg','lab','zero_u8'],['orig'])

# Horizontal nearest colored endpoints.
node('Add',['lab','xbase'],['hx0'])
node('Where',['fg','hx0','zero_u8'],['hx'])
node('MaxPool',['hx'],['hl'],kernel_shape=[1,10],pads=[0,9,0,0],strides=[1,1])
node('Add',['lab','xrbase'],['hr0'])
node('Where',['fg','hr0','zero_u8'],['hr'])
node('MaxPool',['hr'],['hrr'],kernel_shape=[1,10],pads=[0,0,0,9],strides=[1,1])
node('BitwiseAnd',['hl','fifteen_u8'],['hlc'])
node('BitwiseAnd',['hrr','fifteen_u8'],['hrc'])
node('Equal',['hlc','hrc'],['heq'])
node('Greater',['hl','zero_u8'],['hany'])
node('And',['heq','hany'],['hmatch'])
node('Where',['hmatch','hlc','zero_u8'],['hfill'])

# Vertical nearest colored endpoints.
node('Add',['lab','ybase'],['vy0'])
node('Where',['fg','vy0','zero_u8'],['vy'])
node('MaxPool',['vy'],['vt'],kernel_shape=[10,1],pads=[9,0,0,0],strides=[1,1])
node('Add',['lab','yrbase'],['vb0'])
node('Where',['fg','vb0','zero_u8'],['vb'])
node('MaxPool',['vb'],['vbb'],kernel_shape=[10,1],pads=[0,0,9,0],strides=[1,1])
node('BitwiseAnd',['vt','fifteen_u8'],['vtc'])
node('BitwiseAnd',['vbb','fifteen_u8'],['vbc'])
node('Equal',['vtc','vbc'],['veq'])
node('Greater',['vt','zero_u8'],['vany'])
node('And',['veq','vany'],['vmatch'])
node('Where',['vmatch','vtc','zero_u8'],['vfill'])

node('Max',['orig','hfill','vfill'],['filled'])
node('Where',['valid','filled','u255'],['content_grid'])

# Read line color from a guaranteed separator location and render the compact grid.
node('Slice',['input','sl_st','sl_en','sl_ax'],['line_oh'])
node('ArgMax',['line_oh'],['line_i64'],axis=1,keepdims=1)
node('Cast',['line_i64'],['line_u8'],to=TensorProto.UINT8)
node('Where',['valid','line_u8','u255'],['line_grid'])
node('Slice',['line_grid','v_st','v_en','v_ax'],['vg_tail'])
node('Pad',['vg_tail','v_pad','u255'],['v_sep'])
node('Slice',['line_grid','h_st','h_en','h_ax'],['hg_tail'])
node('Pad',['hg_tail','h_pad','u255'],['h_sep'])
node('Max',['v_sep','h_sep'],['x_sep'])
node('Concat',['content_grid','content_grid','v_sep',
               'content_grid','content_grid','v_sep',
               'h_sep','h_sep','x_sep'],['blocks'],axis=1)
node('DepthToSpace',['blocks'],['colormap'],blocksize=3,mode='DCR')
node('Equal',['colormap','color_ids'],['output'])

g=helper.make_graph(N,'task009_nearest_scans',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],initializer=I)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],producer_name='fresh-task009-nearest-scans')
m.ir_version=9
onnx.checker.check_model(m)
m=onnx.shape_inference.infer_shapes(m)
onnx.save(m,OUT)
print(OUT,os.path.getsize(OUT),'nodes',len(N),'params',sum(np.asarray(numpy_helper.to_array(x)).size for x in I))
