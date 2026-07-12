#!/usr/bin/env python3
"""Task215 exact rewrite: rank-3 modulo-row propagation in one terminal Einsum.

The output color is the periodic row-residue sum for each input channel. Black may
remain positive at motif positions, but nonblack channels are scaled by 16, which
is greater than the maximum possible number (10) of valid black rows in one
residue class. Therefore the correct nonblack channel wins wherever a motif cell
exists; otherwise black wins. Outside the valid grid all logits are zero and
argmax remains black.
"""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
OUT=Path(__file__).with_name('candidate_periodic_scaled.onnx')
A=np.zeros((30,3),np.float32)
for r in range(30): A[r,r%3]=1
S=np.full((10,),16,np.float32); S[0]=1
inits=[numpy_helper.from_array(A,'A'),numpy_helper.from_array(S,'S')]
node=helper.make_node('Einsum',['input','input','A','A','S'],['output'],name='periodic_scaled',
 equation='nkrw,nchw,rq,hq,c->ncrw')
g=helper.make_graph([node],'task215_periodic_scaled',
 [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
 [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)],ir_version=10,
 producer_name='neurogolf-structural-rewrite')
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
