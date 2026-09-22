from __future__ import annotations

from pathlib import Path
import json, random, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE=Path(__file__).resolve().parent
TASK=HERE.parent
ROOT=TASK.parents[1]
sys.path.insert(0,str(ROOT/'ARC-GEN'))
sys.path.insert(0,str(TASK))
import arcgen_task


def solve_np(grid):
    a=np.asarray(grid,dtype=np.uint8)
    pink=a==6
    left=np.zeros_like(pink); right=np.zeros_like(pink)
    up=np.zeros_like(pink); down=np.zeros_like(pink)
    for k in range(6):
        if k==0:
            left|=pink; right|=pink; up|=pink; down|=pink
        else:
            left[:,k:]|=pink[:,:-k]
            right[:,:-k]|=pink[:,k:]
            up[k:,:]|=pink[:-k,:]
            down[:-k,:]|=pink[k:,:]
    bounded=left&right&up&down
    q=np.pad(bounded,1)
    dilated=np.zeros_like(bounded)
    for dr in range(3):
        for dc in range(3):
            dilated|=q[dr:dr+a.shape[0],dc:dc+a.shape[1]]
    out=np.full_like(a,8)
    out[dilated]=3
    out[bounded]=4
    out[pink]=6
    return out.tolist()


def check_np(fresh_n=5000):
    ex=json.loads((TASK/'task.json').read_text())
    for split in ('train','test','arc-gen'):
        for i,p in enumerate(ex.get(split,[])):
            assert solve_np(p['input'])==p['output'],(split,i)
    seen={str(p['input']) for split in ('train','test','arc-gen') for p in ex.get(split,[])}
    fresh=0
    for i in range(30000):
        if fresh>=fresh_n: break
        random.seed(10000+i)
        try:p=arcgen_task.generate()
        except Exception:continue
        k=str(p['input'])
        if k in seen:continue
        seen.add(k)
        assert solve_np(p['input'])==p['output'],i
        fresh+=1
    assert fresh==fresh_n,fresh
    print(f'numpy exact public/arc-gen + {fresh} fresh')


def init(name,value): return numpy_helper.from_array(np.asarray(value),name)
def vi(name,dtype,shape): return helper.make_tensor_value_info(name,dtype,shape)


def build(path:Path):
    nodes=[]; inits=[]
    def I(name,value): inits.append(init(name,value)); return name
    def N(op,inputs,output,**attrs):
        nodes.append(helper.make_node(op,inputs,[output],name=f'n{len(nodes):03d}_{op}',**attrs)); return output

    # Only the 13x13 interior can contain pink rectangles. Work in uint8 so
    # every morphology tensor costs one byte per cell.
    I('crop_starts',np.array([0,6,1,1],np.int64))
    I('crop_ends',np.array([1,7,14,14],np.int64))
    N('Slice',['input','crop_starts','crop_ends'],'pink_f')
    N('Cast',['pink_f'],'pink',to=TensorProto.UINT8)
    N('MaxPool',['pink'],'left',kernel_shape=[1,6],pads=[0,5,0,0])
    N('MaxPool',['pink'],'right',kernel_shape=[1,6],pads=[0,0,0,5])
    N('MaxPool',['pink'],'up',kernel_shape=[6,1],pads=[5,0,0,0])
    N('MaxPool',['pink'],'down',kernel_shape=[6,1],pads=[0,0,5,0])
    N('Min',['left','right','up','down'],'bounded')

    # Compact center code: 0=cyan/outside the enclosure, 1=yellow cavity,
    # 2=pink rectangle. First QLinearConv maps it to hinge coordinates:
    #   cyan=(2,0), yellow=(1,0), pink=(0,1).
    N('Add',['bounded','pink'],'code')
    I('s1',np.array(1.0,np.float32))
    I('z0u',np.array(0,np.uint8))
    I('z0i',np.array(0,np.int8))
    W1=np.array([[[[-1]]],[[[1]]]],np.int8)
    B1=np.array([2,-1],np.int32)
    I('W1',W1); I('B1',B1)
    N('QLinearConv',['code','s1','z0u','W1','s1','z0i','s1','z0u','B1'],
      'hinges',kernel_shape=[1,1])

    # The final 3x3 quantized convolution is both the halo operator and the
    # color renderer. x_zero=2 makes implicit padding (2,2), whereas real
    # hinge states are cyan=(2,0), yellow=(1,0), pink=(0,1).
    # In dequantized coordinates A,B:
    #   cyan=(0,-2), yellow=(-1,-2), pink=(-2,-1), padding=(0,0).
    # Exact scores:
    #   green = -sum(A) + 18*A_center
    #   yellow = -A_center - 2*B_center - 4
    #   pink = -A_center + B_center
    #   cyan = 18*sum(A) - sum(B)
    # Green is positive iff the center is unbounded and a bounded neighbor
    # exists. Cyan is positive iff the patch contains real cells and no
    # bounded cell. All tensors after this node are graph-output-free.
    I('z2u',np.array(2,np.uint8))
    W=np.zeros((10,2,3,3),np.int8)
    B=np.zeros((10,),np.int32)
    W[3,0,:,:]=-1; W[3,0,1,1]=17
    W[4,0,1,1]=-1; W[4,1,1,1]=-2; B[4]=-4
    W[6,0,1,1]=-1; W[6,1,1,1]=1
    W[8,0,:,:]=18; W[8,1,:,:]=-1
    I('W',W); I('B',B)
    N('QLinearConv',['hinges','s1','z2u','W','s1','z0i','s1','z0u','B'],
      'output',kernel_shape=[3,3],pads=[2,2,17,17])

    graph=helper.make_graph(nodes,'task125_spatial_hinge_qconv',
        [vi('input',TensorProto.FLOAT,[1,10,30,30])],
        [vi('output',TensorProto.UINT8,[1,10,30,30])],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',14)],ir_version=10,
                            producer_name='neurogolf_structural_rewrite')
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,path)
    print('saved',path,'nodes',len(nodes),'params',sum(int(np.prod(x.dims or [1])) for x in inits),'bytes',path.stat().st_size)


if __name__=='__main__':
    check_np()
    build(HERE/'candidate_spatial_hinge_qconv.onnx')
