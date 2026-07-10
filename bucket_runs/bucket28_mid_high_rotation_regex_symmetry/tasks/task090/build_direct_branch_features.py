from __future__ import annotations
from pathlib import Path
import sys,json,random
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper

HERE=Path(__file__).resolve().parent; TASK=HERE.parent; ROOT=TASK.parents[1]
sys.path.insert(0,str(ROOT/'ARC-GEN'));sys.path.insert(0,str(TASK))
import arcgen_task

INTERVALS=[(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(2,2),(2,3),(3,2)]

def solve_np(grid):
 a=np.array(grid,np.int64); h,w=a.shape; best=(-1,None)
 for r2 in range(h):
  for r1 in range(r2):
   ok=np.all(a[r1:r2+1]==0,axis=0); start=0
   while start<w:
    while start<w and not ok[start]: start+=1
    end=start
    while end<w and ok[end]: end+=1
    if end>start:
     width=end-start
     if width>=2:
      area=(r2-r1+1)*width
      if area>best[0]: best=(area,(r1,width,end-1,r2-r1+1))
    start=end+1
 out=a.copy(); r,width,end,height=best[1]; out[r:r+height,end-width+1:end+1]=6
 return out.tolist()

def check_np(n=5000):
 d=json.loads((TASK/'task.json').read_text())
 for s in ('train','test','arc-gen'):
  for i,p in enumerate(d.get(s,[])): assert solve_np(p['input'])==p['output'],(s,i)
 seen={str(p['input']) for s in ('train','test','arc-gen') for p in d.get(s,[])}; good=0
 for i in range(30000):
  if good>=n: break
  random.seed(10000+i)
  try:p=arcgen_task.generate()
  except Exception: continue
  k=str(p['input'])
  if k in seen: continue
  seen.add(k); assert solve_np(p['input'])==p['output'],i; good+=1
 assert good==n,good
 print('numpy exact public/arc-gen +',good,'fresh')

def init(name,a): return numpy_helper.from_array(np.asarray(a),name)
def vi(name,tp,shape): return helper.make_tensor_value_info(name,tp,shape)

def build(path):
 ns=[];ins=[]
 def I(n,a): ins.append(init(n,a)); return n
 def N(op,x,y,**kw): ns.append(helper.make_node(op,x,[y],name=f'n{len(ns):03d}_{op}',**kw)); return y
 I('starts',np.array([0,0,0],np.int64)); I('ends',np.array([1,5,30],np.int64)); I('axes',np.array([1,2,3],np.int64))
 # One-based coordinates are used both by the causal run transform and the final renderer.
 I('coord1',np.arange(1,31,dtype=np.uint8).reshape(1,1,1,30))
 I('row1',np.arange(1,31,dtype=np.uint8).reshape(1,1,30,1))
 I('zero_u8',np.array(0,np.uint8)); I('one_u8',np.array(1,np.uint8)); I('eight',np.array(8,np.uint8))
 I('heights',np.array([2,3,4,5,2,3,4,2,3,2],np.uint8).reshape(1,1,1,10))
 # Pack (top+1)*8 + height so row membership can use the one-based row vector directly.
 I('topheight1',np.array([10,11,12,13,18,19,20,26,27,34],np.uint8).reshape(1,1,1,10))
 # Direct branch features: Where(cond,[1,1],[1,0]) emits [constant-one, membership]
 # without a float Cast and without a feature Concat.
 I('feat_true',np.array([1.0,1.0],np.float32).reshape(2,1,1,1))
 I('feat_false',np.array([1.0,0.0],np.float32).reshape(2,1,1,1))
 O=np.zeros((10,4),np.float32); S=np.zeros((10,4),np.float32); T=np.zeros((4,2,2),np.float32)
 O[0,0]=1; S[0,0]=1; T[0,0,0]=1
 O[1,1]=1; S[1,1]=1; T[1,0,0]=1
 O[5,2]=1; S[5,2]=1; T[2,0,0]=1
 O[0,3]=-1; O[6,3]=1; S[0,3]=1; T[3,1,1]=1
 I('O',O); I('S',S); I('T',T)

 # Exact causal black-run transform for the first five possible rows.
 N('Slice',['input','starts','ends','axes'],'black_f')
 N('Cast',['black_f'],'black',to=TensorProto.BOOL)
 N('Where',['black','zero_u8','coord1'],'nonzero_pos')
 N('MaxPool',['nonzero_pos'],'last_nonzero',kernel_shape=[1,30],pads=[0,29,0,0],strides=[1,1])
 N('Sub',['coord1','last_nonzero'],'runs')
 ns.append(helper.make_node('Split',['runs'],['R0','R1','R2','R3','R4'],name=f'n{len(ns):03d}_Split',axis=2))
 N('Min',['R0','R1'],'m01'); N('Min',['m01','R2'],'m02'); N('Min',['m02','R3'],'m03'); N('Min',['m03','R4'],'m04')
 N('Min',['R1','R2'],'m12'); N('Min',['m12','R3'],'m13'); N('Min',['m13','R4'],'m14')
 N('Min',['R2','R3'],'m23'); N('Min',['m23','R4'],'m24'); N('Min',['R3','R4'],'m34')
 mats=['m01','m02','m03','m04','m12','m13','m14','m23','m24','m34']
 widths=[]; ends=[]
 for i,m in enumerate(mats):
  N('ReduceMax',[m],f'W{i}',axes=[3],keepdims=1); widths.append(f'W{i}')
  N('ArgMax',[m],f'pi{i}',axis=3,keepdims=1)
  N('Cast',[f'pi{i}'],f'end{i}',to=TensorProto.UINT8); ends.append(f'end{i}')
 ns.append(helper.make_node('Concat',widths,['width_vec'],name=f'n{len(ns):03d}_Concat',axis=3))
 N('Mul',['width_vec','heights'],'raw_area_vec')
 N('Greater',['width_vec','one_u8'],'wide_vec')
 N('Where',['wide_vec','raw_area_vec','zero_u8'],'area_vec')
 N('ArgMax',['area_vec'],'best_i',axis=3,keepdims=1)
 ns.append(helper.make_node('Concat',ends,['end_vec'],name=f'n{len(ns):03d}_Concat',axis=3))
 N('GatherElements',['width_vec','best_i'],'best_w',axis=3)
 N('GatherElements',['end_vec','best_i'],'best_end0',axis=3)
 N('GatherElements',['topheight1','best_i'],'best_th1',axis=3)
 N('Div',['best_th1','eight'],'best_top1')
 N('Mod',['best_th1','eight'],'best_h',fmod=0)
 N('Add',['best_end0','one_u8'],'best_end1')

 # Unsigned subtraction gives a compact range test: values before the start wrap
 # high and fail Less; values from start through start+length-1 remain below length.
 N('Sub',['row1','best_top1'],'rdelta')
 N('Less',['rdelta','best_h'],'row_in')
 N('Sub',['best_end1','coord1'],'cdelta')
 N('Less',['cdelta','best_w'],'col_in')
 N('Where',['row_in','feat_true','feat_false'],'Rfeat')
 N('Where',['col_in','feat_true','feat_false'],'Cfeat')

 # Four CP terms copy black/blue/gray and replace only selected-rectangle black by pink.
 # The 30x30x10 result is the graph output, so it is not charged by the scorer.
 N('Einsum',['input','O','S','Rfeat','Cfeat','T'],'output',equation='nirc,ot,it,axry,bxyc,tab->norc')
 g=helper.make_graph(ns,'task090_direct_branch_features',[vi('input',TensorProto.FLOAT,[1,10,30,30])],[vi('output',TensorProto.FLOAT,[1,10,30,30])],ins)
 m=helper.make_model(g,opset_imports=[helper.make_opsetid('',14)],ir_version=10,producer_name='neurogolf_structural_rewrite')
 onnx.checker.check_model(m,full_check=True); onnx.save(m,path)
 print('saved',path,'nodes',len(ns),'params',sum(int(np.prod(x.dims or [1])) for x in ins),'bytes',path.stat().st_size)

if __name__=='__main__':
 check_np(); build(HERE/'candidate_direct_branch_features.onnx')
