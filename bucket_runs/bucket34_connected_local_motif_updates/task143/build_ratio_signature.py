#!/usr/bin/env python3
"""Task143 exact structural rewrite using a translation-invariant exponential moment ratio.

For a binary sprite S and exponent e=r+4c:
  p1 = sum 2^e, p2 = sum 4^e, sig = p2 / p1^2.
A translation by (dr,dc) multiplies p1 by 2^(dr+4dc) and p2 by
4^(dr+4dc), so sig is invariant. Within a 3x3 box this signature is
injective over every connected 3-, 4-, and 5-cell shape used by the task.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = Path(__file__).with_name('candidate_ratio_signature.onnx')
N=[]; I=[]
def init(name,a,dtype=None):
    I.append(numpy_helper.from_array(np.asarray(a,dtype=dtype),name))
def node(op,ins,out,**kw):
    N.append(helper.make_node(op,ins,[out],name=out,**kw))

# Exact powers of two. The largest p2 term is 2^90, safely finite in float32.
R = np.asarray([2.0**r for r in range(30)],np.float32)
C = np.asarray([2.0**(4*c) for c in range(30)],np.float32)
R3=R[:3].copy(); C3=C[:3].copy()
init('R',R); init('C',C); init('R3',R3); init('C3',C3)
init('one',1.0,np.float32)
init('ref_starts',[0,0,0,0],np.int64); init('ref_ends',[1,1,3,3],np.int64)

# Per-color global signatures.
node('Einsum',['input','R','C'],'p1',equation='nchw,h,w->nc')
node('Einsum',['input','R','R','C','C'],'p2',equation='nchw,h,h,w,w->nc')
node('Mul',['p1','p1'],'p1sq')
node('Div',['p2','p1sq'],'sig')

# The reference shape is exactly the nonblack support in the top-left 3x3.
node('Slice',['input','ref_starts','ref_ends'],'ref_black')
node('Sub',['one','ref_black'],'ref_occ')
node('Einsum',['ref_occ','R3','C3'],'ref_p1',equation='nchw,h,w->nc')
node('Einsum',['ref_occ','R3','R3','C3','C3'],'ref_p2',equation='nchw,h,h,w,w->nc')
node('Mul',['ref_p1','ref_p1'],'ref_p1sq')
node('Div',['ref_p2','ref_p1sq'],'ref_sig')

node('Equal',['sig','ref_sig'],'same_shape')
# The copied target is translated away from the top-left reference; powers are
# monotone, so this also excludes the reference color itself.
node('Greater',['p1','ref_p1'],'translated')
node('And',['same_shape','translated'],'target_ch')
init('neg_one',[-1.0],np.float32); init('pos_one',[1.0],np.float32)
node('Where',['target_ch','neg_one','pos_one'],'sign')

# Signed terminal renderer: the selected color is suppressed and contributes
# positively to gray; all other input channels remain positive at their pixels.
M=np.eye(10,dtype=np.float32)
M[:,5]-=1
M[5,5]=1
init('M',M)
node('Einsum',['input','sign','M'],'output',equation='nchw,nc,co->nohw')

g=helper.make_graph(N,'task143_ratio_signature',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],I)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,
                    producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True)
onnx.save(m,OUT)
print(OUT)
