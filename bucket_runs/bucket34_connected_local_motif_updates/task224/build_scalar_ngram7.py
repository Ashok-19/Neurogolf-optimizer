#!/usr/bin/env python3
"""Task224: decode four rectangle bounds as scalars, then n-gram interval masks and final renderer."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper,TensorProto,numpy_helper
OUT=Path(__file__).with_name('candidate_scalar_ngram7.onnx')
N=[];Z=[]
def I(n,a,d=None):Z.append(numpy_helper.from_array(np.asarray(a,dtype=d),n))
def A(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw))
I('e5',[0,0,0,0,0,1,0,0,0,0],np.float32)
I('rank_min',4.0**(-np.arange(30,dtype=np.float32)))
I('rank_max',4.0**(np.arange(30,dtype=np.float32)-29))
I('ln4',np.log(4).astype(np.float32));I('c29',29.,np.float32)
# Gray clues have unique outer extrema. Base-4 domination decodes each extremum exactly.
def bounds(axis,eq):
 A('Einsum',['input','e5','rank_min'],axis+'_min_code',equation=eq)
 A('Log',[axis+'_min_code'],axis+'_min_log');A('Div',[axis+'_min_log','ln4'],axis+'_min_l4');A('Floor',[axis+'_min_l4'],axis+'_min_floor');A('Neg',[axis+'_min_floor'],axis+'_min_f');A('Cast',[axis+'_min_f'],axis+'_mn',to=TensorProto.INT32)
 A('Einsum',['input','e5','rank_max'],axis+'_max_code',equation=eq)
 A('Log',[axis+'_max_code'],axis+'_max_log');A('Div',[axis+'_max_log','ln4'],axis+'_max_l4');A('Add',[axis+'_max_l4','c29'],axis+'_max_pos');A('Floor',[axis+'_max_pos'],axis+'_max_f');A('Cast',[axis+'_max_f'],axis+'_mx',to=TensorProto.INT32)
bounds('r','bchw,c,h->b');bounds('c','bchw,c,w->b')
I('c30',30,np.int32);I('type_off',np.array([[0],[900],[1800]],np.int32))
I('left_tok',np.tile(np.array([[-6,-5,-4]],np.int32),(3,1)));I('right_tok',np.tile(np.array([[-3,-2,-1]],np.int32),(3,1)))
def toks(p):
 A('Mul',[p+'_mn','c30'],p+'_mn30');A('Add',[p+'_mn30',p+'_mx'],p+'_pair');A('Add',[p+'_pair','type_off'],p+'_codes');A('Concat',['left_tok',p+'_codes','right_tok'],p+'_tok',axis=1)
toks('r');toks('c')
intervals=[]
for l in range(1,8):
 for s in range(8-l):
  if s<=3<s+l:intervals.append((l,s))
assert len(intervals)==16
by={l:[] for l in range(1,8)}
for mn in range(30):
 for mx in range(mn,30):
  pair=30*mn+mx
  masks=[range(16),range(mn+1,mx),range(mn+2,mx-1)]
  for t,poss in enumerate(masks):
   seq=(-6,-5,-4,pair+900*t,-3,-2,-1)
   for pos,(l,s) in zip(poss,intervals):by[l].append((seq[s:s+l],pos))
by[1].append(((999999,),29))
pool=[];idx=[];counts=[]
for l in range(1,8):
 counts.append(len(pool))
 for gram,pos in by[l]:pool.extend(gram);idx.append(pos)
attrs=dict(mode='TF',min_gram_length=1,max_gram_length=7,max_skip_count=0,ngram_counts=counts,ngram_indexes=idx,pool_int64s=pool)
A('TfIdfVectorizer',['r_tok'],'R',**attrs);A('TfIdfVectorizer',['c_tok'],'C',**attrs)
S=np.zeros((3,10),np.float32);S[0,0]=1;S[1,5]=1;S[2,[1,2,3,4,6,7,8,9]]=1
E=np.array([[[1,0,0],[0,1,0],[0,0,1]],[[-1,-1,1],[0,0,0],[0,0,0]],[[1,1,-1],[0,0,0],[0,0,0]]],np.float32)
I('S',S);I('E',E)
A('Einsum',['input','input','S','S','E','R','C'],'output',equation='ndhw,ncij,kd,qc,tkq,th,tw->nchw')
g=helper.make_graph(N,'task224_scalar_ngram7',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],Z)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',16)],ir_version=10,producer_name='neurogolf-structural-rewrite');onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
