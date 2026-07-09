#!/usr/bin/env python3
"""Fresh task004 rewrite: shift rule on a compact scalar uint8 label plane."""
from __future__ import annotations
import importlib.util,json,random,sys
from pathlib import Path
import numpy as np,onnx
from onnx import TensorProto,checker,helper,numpy_helper
ROOT=Path(__file__).resolve().parents[3]; TASK=ROOT/'tasks/task004'; OUT=TASK/'work/candidate_compact.onnx'
sys.path.insert(0,str(ROOT/'ARC-GEN'))
spec=importlib.util.spec_from_file_location('arc004',TASK/'arcgen_task.py');arc=importlib.util.module_from_spec(spec);spec.loader.exec_module(arc)

def solve(g):
 a=np.asarray(g,dtype=np.uint8); h,w=a.shape; o=a.copy()
 for r in range(h):
  nxt=a[r+1] if r+1<h else np.zeros(w,dtype=np.uint8)
  nz=np.flatnonzero(nxt); edge=int(nz[-1]) if len(nz) else 0
  sh=(a[r]!=0)&(np.arange(w)<edge)
  base=a[r].copy();base[sh]=0
  shifted=np.zeros(w,dtype=np.uint8);shifted[1:]=sh[:-1].astype(np.uint8)*int(a[r].max())
  o[r]=np.where(shifted!=0,shifted,base)
 return o.tolist()

def test_numpy():
 d=json.loads((TASK/'task.json').read_text())
 for s in ('train','test','arc-gen'):
  for i,p in enumerate(d[s]): assert solve(p['input'])==p['output'],(s,i)
 seen={str(p['input']) for s in d for p in d[s]};n=0
 for seed in range(10000,30000):
  random.seed(seed)
  try:p=arc.generate()
  except Exception:continue
  if str(p['input']) in seen:continue
  assert solve(p['input'])==p['output'],seed;n+=1
  if n==1000:break
 assert n==1000;print('numpy exact',n)

def it(n,a):return numpy_helper.from_array(np.asarray(a),name=n)
def build():
 I=[it('w',(16+np.arange(10,dtype=np.float32)).reshape(1,10,1,1)),it('zero_u8',np.array(0,np.uint8)),it('bg_u8',np.array(16,np.uint8)),it('colors',(16+np.arange(10,dtype=np.uint8)).reshape(1,10,1,1)),it('cols',np.arange(16,dtype=np.uint8).reshape(1,1,1,16)),it('pads_next',np.array([0,0,-1,0,0,0,1,0],np.int64)),it('pads_shift',np.array([0,0,0,1,0,0,0,-1],np.int64)),it('pads_out',np.array([0,0,0,0,0,0,14,14],np.int64))]
 N=[]
 def n(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw));return out
 lf=n('Conv',['input','w'],'label16_f',kernel_shape=[1,1],pads=[0,0,-14,-14]);lu=n('Cast',[lf],'label16_u8',to=TensorProto.UINT8);occ=n('Greater',[lu,'bg_u8'],'occupied_bool')
 rowlab=n('ReduceMax',[lu],'row_label_u8',axes=[3],keepdims=1);rai=n('ArgMax',[lu],'row_argmax_i64',axis=3,keepdims=1,select_last_index=1);rau=n('Cast',[rai],'row_argmax_u8',to=TensorProto.UINT8);rh=n('Greater',[rowlab,'bg_u8'],'row_has_fg');rmax=n('Where',[rh,rau,'zero_u8'],'row_max_u8')
 nxt=n('Pad',[rmax,'pads_next','zero_u8'],'next_row_max',mode='constant');left=n('Less',['cols',nxt],'left_of_next_edge');sh=n('And',[left,occ],'shift_pixel');base=n('Where',[sh,'bg_u8',lu],'base_shifted_u8');shp=n('Pad',[sh,'pads_shift'],'shifted_pixel',mode='constant');out16=n('Where',[shp,rowlab,base],'label16_out_u8');out30=n('Pad',[out16,'pads_out','zero_u8'],'label30_out_u8',mode='constant');n('Equal',[out30,'colors'],'output')
 g=helper.make_graph(N,'task004_compact',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],I)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)],producer_name='fresh-rule-rewrite');m.ir_version=8;checker.check_model(m,full_check=True);onnx.save(m,OUT);print('saved',OUT)
if __name__=='__main__':test_numpy();build()
