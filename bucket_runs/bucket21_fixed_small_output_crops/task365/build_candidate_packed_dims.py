"""Fresh deterministic ONNX rewrite for task365.

Semantic state:
- collapse the 10x10 logical grid to uint8 labels;
- detect up to three filled-rectangle top-left corners;
- decode each rectangle's width/height from three occupancy probes packed as
  a 3-bit token and a tiny exact lookup table;
- count red cells, select the winner, gather its at-most-6x6 labels;
- render with one terminal positive-logit ConvTranspose.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
OUT=Path(__file__).with_name('candidate_packed_dims.onnx')

def init(name,a): return numpy_helper.from_array(np.asarray(a),name=name)
def vi(name,dt,shape): return helper.make_tensor_value_info(name,dt,shape)
nodes=[]
def N(op,ins,outs,**kw): nodes.append(helper.make_node(op,ins,outs,name=outs[0],**kw))

# Compact labels and candidate rectangle corners.
N('Conv',['input','collapse_weights'],['labels_f32'],dilations=[20,20],kernel_shape=[2,2])
N('Cast',['labels_f32'],['labels'],to=TensorProto.UINT8)
N('Greater',['labels','zero_u8'],['occupied'])
N('Equal',['labels','red_u8'],['red'])
N('Cast',['occupied'],['occupied_i8'],to=TensorProto.INT8)
N('QLinearConv',['occupied_i8','q_scale','q_zero','corner_kernel','q_scale','q_zero','q_scale','q_zero'],['corner_scores_i8'],pads=[1,1,0,0])
N('Flatten',['corner_scores_i8'],['corner_scores_flat_i8'],axis=1)
N('Cast',['corner_scores_flat_i8'],['corner_scores'],to=TensorProto.FLOAT16)
N('TopK',['corner_scores','topk_three'],['corner_values','corner_indices64'],axis=1,largest=1,sorted=1)
N('Equal',['corner_values','four_f16'],['present_2d'])
N('Squeeze',['present_2d','axis0'],['present'])
N('Cast',['corner_indices64'],['corner_indices32'],to=TensorProto.INT32)
N('Squeeze',['corner_indices32','axis0'],['corner_indices'])
N('Div',['corner_indices','eight_i32'],['rows'])
N('Mod',['corner_indices','eight_i32'],['columns'])

# Width: sample occupancy at offsets 3,4,5, pack bits, exact prefix lookup.
N('Unsqueeze',['columns','axis1'],['columns_2d'])
N('Gather',['occupied','rows'],['corner_rows'],axis=2)
N('Unsqueeze',['columns_2d','axes01'],['columns_4d'])
N('Add',['columns_4d','offsets'],['width_indices'])
N('Less',['width_indices','ten_i32'],['width_valid'])
N('Min',['width_indices','nine_i32'],['width_indices_safe'])
N('GatherElements',['corner_rows','width_indices_safe'],['width_cells_raw'],axis=3)
N('And',['width_cells_raw','width_valid'],['width_cells'])
N('Cast',['width_cells'],['width_bits'],to=TensorProto.UINT8)
N('MatMulInteger',['width_bits','bit_weights'],['width_token'])
N('Gather',['prefix_extra','width_token'],['width_extra'],axis=0)
N('Add',['width_extra','three_i32'],['width_4d'])
N('Squeeze',['width_4d','axes01'],['width_2d'])
N('Squeeze',['width_2d','axis1'],['width'])

# Height: identical token decoder on transposed candidate columns.
N('Gather',['occupied','columns'],['corner_columns'],axis=3)
N('Transpose',['corner_columns'],['corner_columns_t'],perm=[0,1,3,2])
N('Unsqueeze',['rows','axis1'],['rows_2d'])
N('Unsqueeze',['rows_2d','axes01'],['rows_4d'])
N('Add',['rows_4d','offsets'],['height_indices'])
N('Less',['height_indices','ten_i32'],['height_valid'])
N('Min',['height_indices','nine_i32'],['height_indices_safe'])
N('GatherElements',['corner_columns_t','height_indices_safe'],['height_cells_raw'],axis=3)
N('And',['height_cells_raw','height_valid'],['height_cells'])
N('Cast',['height_cells'],['height_bits'],to=TensorProto.UINT8)
N('MatMulInteger',['height_bits','bit_weights'],['height_token'])
N('Gather',['prefix_extra','height_token'],['height_extra'],axis=0)
N('Add',['height_extra','three_i32'],['height_4d'])
N('Squeeze',['height_4d','axes01'],['height_2d'])
N('Squeeze',['height_2d','axis1'],['height'])

# Candidate rectangle gates and exact red counts.
N('GreaterOrEqual',['coordinate10','rows_2d'],['row_ge'])
N('Add',['rows_2d','height_2d'],['row_end'])
N('Less',['coordinate10','row_end'],['row_lt'])
N('And',['row_ge','row_lt'],['row_gate'])
N('Cast',['row_gate'],['row_gate_f16'],to=TensorProto.FLOAT16)
N('GreaterOrEqual',['coordinate10','columns_2d'],['column_ge'])
N('Add',['columns_2d','width_2d'],['column_end'])
N('Less',['coordinate10','column_end'],['column_lt'])
N('And',['column_ge','column_lt'],['column_gate'])
N('Cast',['column_gate'],['column_gate_f16'],to=TensorProto.FLOAT16)
N('Cast',['red'],['red_f16'],to=TensorProto.FLOAT16)
N('Einsum',['row_gate_f16','red_f16','column_gate_f16'],['red_counts'],equation='ir,abrc,ic->i')
N('Where',['present','red_counts','zero_f16'],['candidate_scores'])
N('ArgMax',['candidate_scores'],['winner'],axis=0,keepdims=1)
N('Gather',['rows','winner'],['selected_row'])
N('Gather',['columns','winner'],['selected_column'])
N('Gather',['height','winner'],['selected_height'])
N('Gather',['width','winner'],['selected_width'])

# Gather the selected at-most-6x6 rectangle and mark cells outside its true extent.
N('Add',['selected_row','coordinate6'],['row_indices_raw'])
N('Min',['row_indices_raw','nine_i32'],['row_indices'])
N('Add',['selected_column','coordinate6'],['column_indices_raw'])
N('Min',['column_indices_raw','nine_i32'],['column_indices'])
N('Gather',['labels','column_indices'],['selected_columns'],axis=3)
N('Gather',['selected_columns','row_indices'],['selected_patch'],axis=2)
N('Less',['coordinate6','selected_height'],['valid_rows'])
N('Less',['coordinate6','selected_width'],['valid_columns'])
N('Unsqueeze',['valid_rows','axes013'],['valid_rows_4d'])
N('Unsqueeze',['valid_columns','axes012'],['valid_columns_4d'])
N('And',['valid_rows_4d','valid_columns_4d'],['valid_patch'])
N('Where',['valid_patch','selected_patch','sentinel'],['output_labels'])

# Terminal signed quadratic renderer. Sentinel label 10 stays nonpositive.
N('Mul',['output_labels','output_labels'],['labels_squared'])
N('Concat',['output_labels','labels_squared'],['quadratic_features_u8'],axis=1)
N('Cast',['quadratic_features_u8'],['quadratic_features'],to=TensorProto.FLOAT16)
N('ConvTranspose',['quadratic_features','quadratic_weights','quadratic_bias'],['output'],pads=[0,0,-24,-24],strides=[1,1])

collapse=np.zeros((1,10,2,2),np.float32)
for c in range(10): collapse[0,c,0,0]=c
corner=np.zeros((1,1,4,4),np.int8); corner[0,0,0,1]=-1;corner[0,0,1,0]=-1;corner[0,0,1,1]=4
qW=np.zeros((2,10,1,1),np.float16);qB=np.empty(10,np.float16)
for k in range(10):
    if k==0: qW[:,k,0,0]=0;qB[k]=-1
    else: qW[0,k,0,0]=2*k;qW[1,k,0,0]=-1;qB[k]=np.float16(.25-k*k)
inits=[
 init('collapse_weights',collapse),init('corner_kernel',corner),
 init('q_scale',np.array([1],np.float32)),init('q_zero',np.array([0],np.int8)),
 init('zero_u8',np.array(0,np.uint8)),init('red_u8',np.array(2,np.uint8)),
 init('four_f16',np.array([4],np.float16)),init('topk_three',np.array([3],np.int64)),
 init('eight_i32',np.array([8],np.int32)),init('ten_i32',np.array(10,np.int32)),init('nine_i32',np.array(9,np.int32)),
 init('three_i32',np.array([[[[3]]]],np.int32)),init('offsets',np.array([[[[3,4,5]]]],np.int32)),
 init('bit_weights',np.array([[1],[2],[4]],np.uint8)),init('prefix_extra',np.array([0,1,0,2,0,1,0,3],np.int32)),
 init('coordinate10',np.arange(10,dtype=np.int32)[None,:]),init('coordinate6',np.arange(6,dtype=np.int32)),
 init('axis0',np.array([0],np.int64)),init('axis1',np.array([1],np.int64)),init('axes01',np.array([0,1],np.int64)),
 init('axes013',np.array([0,1,3],np.int64)),init('axes012',np.array([0,1,2],np.int64)),
 init('zero_f16',np.array(0,np.float16)),init('sentinel',np.array([[[[10]]]],np.uint8)),
 init('quadratic_weights',qW),init('quadratic_bias',qB),
]
g=helper.make_graph(nodes,'task365_packed_dims',[vi('input',TensorProto.FLOAT,[1,10,30,30])],[vi('output',TensorProto.FLOAT16,[1,10,30,30])],inits)
m=helper.make_model(g,ir_version=10,opset_imports=[helper.make_opsetid('',18)])
m=onnx.shape_inference.infer_shapes(m,strict_mode=True)
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
