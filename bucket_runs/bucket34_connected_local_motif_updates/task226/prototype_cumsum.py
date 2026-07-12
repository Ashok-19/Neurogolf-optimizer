#!/usr/bin/env python
import numpy as np, json, random, sys
sys.path.insert(0,'ARC-GEN')
import importlib.util
sp=importlib.util.spec_from_file_location('t226','tasks/task226/arcgen_task.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)

def cats(sep):
    sep=np.asarray(sep,dtype=np.int32)
    pre=np.cumsum(sep)
    total=pre[-1]
    half=total//2
    non=1-sep
    return np.stack([np.ones_like(sep),sep,np.ones_like(sep),non*(pre==0),non*(pre==half),non*(pre==total)])

def solve(grid):
    a=np.asarray(grid)
    rs=a[:,0]==5; cs=a[0,:]==5
    rf=cats(rs);cf=cats(cs)
    terms=np.stack([np.ones((len(rs),len(cs)),dtype=np.int32), rs[:,None].astype(np.int32)*np.ones((1,len(cs)),dtype=np.int32), np.ones((len(rs),1),dtype=np.int32)*cs[None,:].astype(np.int32), rf[3,:,None]*cf[3,None,:], rf[4,:,None]*cf[4,None,:], rf[5,:,None]*cf[5,None,:]])
    W=np.zeros((10,6),dtype=np.int32);W[:,0]=-1
    W[0]=[1,-2,-2,-2,-2,-2]
    W[1]=[-1,-2,-2,2,0,0]
    W[2]=[-1,-2,-2,0,2,0]
    W[3]=[-1,-2,-2,0,0,2]
    W[5]=[-1,2,2,0,0,0]
    z=np.einsum('kf,fhw->khw',W,terms)
    out=np.zeros_like(a)
    for r in range(a.shape[0]):
      for c in range(a.shape[1]):
       pos=np.where(z[:,r,c]>0)[0]
       if len(pos)!=1:
        raise AssertionError((r,c,pos,z[:,r,c],terms[:,r,c]))
       out[r,c]=pos[0]
    return out

data=json.load(open('tasks/task226/task.json'))
for split in data:
 for i,p in enumerate(data[split]):
  got=solve(p['input']);exp=np.array(p['output'])
  if not np.array_equal(got,exp):
   print('FAIL',split,i,np.argwhere(got!=exp)[:20]);raise SystemExit(1)
print('bundled exact')
seen={str(x['input']) for s in data.values() for x in s}
n=0
for seed in range(10000,13000):
 random.seed(seed);p=g.generate()
 if str(p['input']) in seen:continue
 seen.add(str(p['input']));n+=1
 if not np.array_equal(solve(p['input']),np.array(p['output'])):
  print('fresh fail',seed);raise SystemExit(1)
 if n>=1000:break
print('fresh',n,'exact')
