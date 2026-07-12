"""Fresh deterministic ONNX rewrite for task346.

The selected color is compressed to twelve exact 12-bit row masks.  Their
scalar string sequence is matched by one free regex attribute for an aligned
111/101/111 3x3 ring.  The other present color is the center/output color.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
OUT=Path(__file__).with_name('candidate_rowmask_regex_crop10.onnx')

def init(name,a): return numpy_helper.from_array(np.asarray(a),name=name)
def vi(name,dt,shape): return helper.make_tensor_value_info(name,dt,shape)
nodes=[]
def N(op,ins,outs,domain='',**kw): nodes.append(helper.make_node(op,ins,outs,name=outs[0],domain=domain,**kw))

# Find the two present nonzero colors and choose the lower ID for ring testing.
N('ReduceMax',['input','present_axes'],['present10'],keepdims=0)
N('Slice',['present10','one64','ten64'],['present9'])
N('ArgMax',['present9'],['low_relative'],axis=0,keepdims=1,select_last_index=0)
N('ArgMax',['present9'],['high_relative'],axis=0,keepdims=1,select_last_index=1)
N('Add',['low_relative','one64'],['low_color'])
N('Add',['high_relative','one64'],['high_color'])
N('OneHot',['low_color','depth10','onehot_values'],['low_gate'],axis=-1)

# Exact 12-bit mask for each physical row, then keep the first 12 rows.
N('Einsum',['input','low_gate','column_bits'],['row_masks30'],equation='nchw,nc,w->nh')
N('Slice',['row_masks30','row1','row11','axis1'],['row_masks10'])
N('Cast',['row_masks10'],['row_tokens'],to=TensorProto.INT32)
N('Add',['row_tokens','token_offset'],['fixed_tokens'])
N('Cast',['fixed_tokens'],['row_strings'],to=TensorProto.STRING)
parts=[f'row_s{i}' for i in range(10)]
N('Split',['row_strings'],parts,axis=1,num_outputs=10)
# Fixed-width tokens concatenate without separators.
level=parts
round_id=0
while len(level)>1:
    nxt=[]
    for i in range(0,len(level),2):
        if i+1==len(level): nxt.append(level[i])
        else:
            q=f'cat{round_id}_{i//2}';N('StringConcat',[level[i],level[i+1]],[q]);nxt.append(q)
    level=nxt;round_id+=1
N('RegexFullMatch',[level[0]],['low_is_ring_2d'],pattern='__PATTERN__')
N('Squeeze',['low_is_ring_2d','axis1'],['low_is_ring'])
N('Where',['low_is_ring','high_color','low_color'],['winner_color'])
N('Mul',['winner_color','plane_size'],['flat_index'])
N('Unsqueeze',['flat_index','axes123'],['unpool_index'])
N('MaxUnpool',['one_value','unpool_index','output_shape'],['output'],kernel_shape=[1,1],strides=[1,1])

# Construct the exact decimal-token regex.  For each horizontal offset c,
# A_c contains row masks with bits c:c+3 == 111; B_c contains 101.
def alts(kind,c):
    target=7 if kind=='A' else 5
    return '|'.join(str(10000+m) for m in range(1024) if ((m>>c)&7)==target)
branches=[]
for c in range(8):
    A=alts('A',c);B=alts('B',c)
    branches.append(f'(?:[0-9]{{5}})*(?:{A})(?:{B})(?:{A})(?:[0-9]{{5}})*')
pattern='(?:'+'|'.join(branches)+')'
# Replace placeholder after node construction to keep source readable.
for node in nodes:
    if node.op_type=='RegexFullMatch':
        for a in node.attribute:
            if a.name=='pattern': a.s=pattern.encode()

inits=[
 init('present_axes',np.array([0,2,3],np.int64)),init('one64',np.array([1],np.int64)),init('ten64',np.array([10],np.int64)),
 init('depth10',np.array(10,np.int64)),init('onehot_values',np.array([0,1],np.float32)),
 init('column_bits',np.r_[0.,2.0**np.arange(10,dtype=np.float32),np.zeros(19,dtype=np.float32)]),
 init('row1',np.array([1],np.int64)),init('row11',np.array([11],np.int64)),init('axis1',np.array([1],np.int64)),
 init('token_offset',np.array(10000,np.int32)),
 init('plane_size',np.array([900],np.int64)),init('axes123',np.array([1,2,3],np.int64)),
 init('one_value',np.ones((1,1,1,1),np.float16)),init('output_shape',np.array([1,10,30,30],np.int64)),
]
g=helper.make_graph(nodes,'task346_rowmask_regex',[vi('input',TensorProto.FLOAT,[1,10,30,30])],[vi('output',TensorProto.FLOAT16,[1,10,30,30])],inits)
m=helper.make_model(g,ir_version=10,opset_imports=[helper.make_opsetid('',20)])
m=onnx.shape_inference.infer_shapes(m,strict_mode=True)
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT,'pattern_bytes',len(pattern.encode()),'model_bytes',OUT.stat().st_size)
