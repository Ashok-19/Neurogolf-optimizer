#!/usr/bin/env python3
"""Task215 exact rewrite: rank-3 modulo-row propagation in one terminal Einsum."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate_periodic_bilinear.onnx')
# A[r,q] marks row residue q. A A^T is the same-mod-3 relation.
A=np.zeros((30,3),np.float32)
for r in range(30): A[r,r%3]=1
# Channel renderer. Nonblack channels preserve their periodic signal.
# For black, P_black - 7*P_nonblack is positive iff no motif color occurs
# in that column/residue. At most 7 valid rows share a residue for height<=20.
M=np.eye(10,dtype=np.float32)
M[0,1:]=-7
inits=[numpy_helper.from_array(A,'A'),numpy_helper.from_array(M,'M')]
node=helper.make_node('Einsum',['input','input','A','A','M'],['output'],name='periodic_render',
 equation='nkrw,nbhw,rq,hq,ob->norw')
g=helper.make_graph([node],'task215_periodic_bilinear',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
