"""Fresh deterministic ONNX builder for task377.

Rule state:
  1. One scalar signature per input row identifies horizontal bands.
  2. The middle band crosses every nested rectangle.
  3. Run starts in that row give the outer-to-inner color sequence.
  4. A 5x9 edge-depth lookup renders the at-most-9-square output.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name('candidate_terminal_einsum_topk10.onnx')

def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

nodes=[]
def N(op, inputs, outputs, **attrs):
    nodes.append(helper.make_node(op, inputs, outputs, name=f'n{len(nodes):02d}_{op}', **attrs))

# Row-band discovery.
N('Einsum',['input','digit_weights','column_ones'],['row_signature'],equation='nchw,c,w->nh')
N('Cast',['row_signature'],['row_signature_f16'],to=TensorProto.FLOAT16)
N('Slice',['row_signature_f16','s0','e29','a1'],['row_top'])
N('Slice',['row_signature_f16','s1','e30','a1'],['row_bottom'])
N('Equal',['row_top','row_bottom'],['same_rows'])
N('Not',['same_rows'],['row_changed'])
N('Concat',['first_true','row_changed'],['row_starts_bool'],axis=1)
N('Where',['row_starts_bool','descending_scores','zero_f16'],['scored_rows'])
N('TopK',['scored_rows','k10'],['row_topk_values','row_start_indices'],axis=1,largest=1,sorted=1)
N('Greater',['row_topk_values','zero_f16'],['row_topk_positive'])
N('Cast',['row_topk_positive'],['row_topk_positive_f16'],to=TensorProto.FLOAT16)
N('ReduceSum',['row_topk_positive_f16','a1'],['boundary_count_plus'],keepdims=0)
N('Slice',['row_signature_f16','e29','e30','a1'],['last_signature'])
N('Equal',['last_signature','zero_f16'],['last_is_padding'])
N('Cast',['last_is_padding'],['padding_f16'],to=TensorProto.FLOAT16)
N('Squeeze',['padding_f16','a1'],['padding_adjustment'])
N('Sub',['boundary_count_plus','padding_adjustment'],['boundary_count'])
N('Sub',['boundary_count','one_f16'],['boundary_minus_one'])
N('Div',['boundary_minus_one','two_f16'],['middle_f16'])
N('Cast',['middle_f16'],['middle_index'],to=TensorProto.INT64)
N('Unsqueeze',['middle_index','a1'],['middle_index_2d'])
N('GatherElements',['row_start_indices','middle_index_2d'],['selected_row_2d'],axis=1)
N('Squeeze',['selected_row_2d','a1'],['selected_row'])

# Extract compact run-color sequence from the selected middle band.
N('OneHot',['selected_row','depth30','onehot_values'],['selected_row_gate'],axis=-1)
N('Einsum',['input','digit_weights','selected_row_gate'],['selected_labels_f32'],equation='nchw,c,nh->nw')
N('Cast',['selected_labels_f32'],['selected_labels'],to=TensorProto.UINT8)
N('Slice',['selected_labels','s0','e29','a1'],['selected_left'])
N('Slice',['selected_labels','s1','e30','a1'],['selected_right'])
N('Equal',['selected_left','selected_right'],['selected_same'])
N('Not',['selected_same'],['selected_changed'])
N('Concat',['first_true','selected_changed'],['selected_starts_bool'],axis=1)
N('Where',['selected_starts_bool','descending_scores','zero_f16'],['selected_start_scores'])
N('TopK',['selected_start_scores','k9'],['selected_topk_values','selected_start_indices'],axis=1,largest=1,sorted=1)
N('GatherElements',['selected_labels','selected_start_indices'],['run_colors'],axis=1)

# Keep only the outer-to-inner half (maximum five layers).
N('Slice',['run_colors','s0','e5','a1'],['colors5'])
N('Cast',['colors5'],['colors_f'],to=TensorProto.FLOAT)
N('Mul',['colors_f','colors_f'],['colors_sq'])
N('Slice',['colors_f','s0','e4','a1'],['colors_prev4'])
N('Slice',['colors_sq','s0','e4','a1'],['sq_prev4'])
N('Concat',['zero_f32_2d','colors_prev4'],['colors_prev'],axis=1)
N('Concat',['zero_f32_2d','sq_prev4'],['sq_prev'],axis=1)
N('Sub',['colors_f','colors_prev'],['delta_color'])
N('Sub',['colors_sq','sq_prev'],['delta_sq'])
N('Concat',['inside_delta','delta_color','delta_sq'],['features'],axis=0)
N('OneHot',['middle_index','depth5','onehot_values'],['layer_gate'],axis=-1)
# Final n-ary renderer: nested square gates telescope color and color^2.
N('Einsum',['layer_gate','row_gates','row_gates','features','channel_coeff'],['output'],equation='nl,ldx,ldy,fd,fk->nkxy')

row_gates=np.zeros((5,5,30),dtype=np.float32)
for m in range(5):
    side=2*m+1
    for d in range(m+1):
        row_gates[m,d,d:side-d]=1
channel_coeff=np.empty((3,10),dtype=np.float32)
for k in range(10):
    channel_coeff[0,k]=0.25-k*k
    channel_coeff[1,k]=2*k
    channel_coeff[2,k]=-1

inits=[
 init('digit_weights',np.arange(10,dtype=np.float32)),
 init('column_ones',np.ones(30,dtype=np.float32)),
 init('s0',np.array([0],np.int64)), init('s1',np.array([1],np.int64)),
 init('e4',np.array([4],np.int64)), init('e5',np.array([5],np.int64)),
 init('e29',np.array([29],np.int64)), init('e30',np.array([30],np.int64)),
 init('a1',np.array([1],np.int64)),
 init('first_true',np.array([[True]],np.bool_)),
 init('descending_scores',np.arange(30,0,-1,dtype=np.float16)[None,:]),
 init('k9',np.array([9],np.int64)), init('k10',np.array([10],np.int64)),
 init('zero_f16',np.array([0],np.float16)), init('one_f16',np.array([1],np.float16)), init('two_f16',np.array([2],np.float16)),
 init('depth30',np.array(30,np.int64)), init('depth5',np.array(5,np.int64)), init('onehot_values',np.array([0,1],np.float32)),
 init('zero_f32_2d',np.zeros((1,1),np.float32)),
 init('inside_delta',np.array([[1,0,0,0,0]],np.float32)),
 init('row_gates',row_gates), init('channel_coeff',channel_coeff),
]

graph=helper.make_graph(nodes,'task377_terminal_einsum',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
model=helper.make_model(graph,ir_version=10,opset_imports=[helper.make_opsetid('',18)])
model=onnx.shape_inference.infer_shapes(model,strict_mode=True)
onnx.checker.check_model(model,full_check=True)
onnx.save(model,OUT)
print(OUT,OUT.stat().st_size)
