#!/usr/bin/env python3
"""Bounded fresh validator for task330.

The official random generator can spend unbounded time rejecting layouts. This
script samples the same sprite/count/placement rule directly with bounded
backtracking, preserving one-cell bounding-box separation.
"""
from __future__ import annotations
import importlib.util, json, random, sys
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ARC-GEN'))
import common
import validate

def load_task():
 p=ROOT/'tasks/task330/arcgen_task.py'
 spec=importlib.util.spec_from_file_location('task330_arcgen_direct',p)
 m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def place_boxes(rng, wides, talls, size=10):
 order=list(range(len(wides)))
 choices=[]
 def rec(i):
  if i==len(order): return True
  j=order[i]
  pos=[(r,c) for r in range(size-talls[j]+1) for c in range(size-wides[j]+1)]
  rng.shuffle(pos)
  for r,c in pos:
   ok=True
   for k,(rr,cc) in enumerate(choices):
    jj=order[k]
    if not (rr+talls[jj]+1<=r or r+talls[j]+1<=rr or cc+wides[jj]+1<=c or c+wides[j]+1<=cc):
     ok=False; break
   if ok:
    choices.append((r,c))
    if rec(i+1): return True
    choices.pop()
  return False
 return choices.copy() if rec(0) else None

def fresh_cases(n=1000, seed=330_20260710):
 rng=random.Random(seed); task=load_task(); out=[]
 random_state=random.getstate(); random.seed(seed ^ 0x330)
 try:
  while len(out)<n:
   ns=rng.randint(3,5)
   wides=[rng.randint(2,4) for _ in range(ns)]; talls=[6-w for w in wides]
   pos=place_boxes(rng,wides,talls)
   if pos is None: continue
   counts=[rng.randint(4,8) if rng.randint(0,2) else 6 for _ in range(ns)]
   rows=[]; cols=[]; idxs=[]
   for i,(count,w,h) in enumerate(zip(counts,wides,talls)):
    pix=common.continuous_creature(count,w,h)
    rows.extend(r for r,c in pix); cols.extend(c for r,c in pix); idxs.extend([i]*len(pix))
   brows=[p[0] for p in pos]; bcols=[p[1] for p in pos]
   out.append(task.generate(rows=rows,cols=cols,idxs=idxs,brows=brows,bcols=bcols,size=10))
 finally:
  random.setstate(random_state)
 return out

def main():
 model=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'tasks/task330/model.onnx'
 off=validate.load_official_utils(); san=off.sanitize_model(onnx.load(model)); assert san is not None
 so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL; so.intra_op_num_threads=1;so.inter_op_num_threads=1
 sess=ort.InferenceSession(san.SerializeToString(),so,providers=['CPUExecutionProvider'])
 cases=fresh_cases(1000)
 right,wrong,_=off.verify_subset(sess,cases)
 print(json.dumps({'fresh_right':right,'fresh_total':len(cases),'fresh_wrong':wrong,'seed':330_20260710,'method':'bounded direct sampling of authoritative task rule'},indent=2))
 raise SystemExit(0 if wrong==0 else 1)
if __name__=='__main__': main()
