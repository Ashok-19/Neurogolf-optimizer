#!/usr/bin/env python3
"""Build a fresh sparse reflection solver for task285.

Rule:
- Encode each active cell as its color and padded cells as -1.
- Extract all at-most-33 colored cells.
- A shown sprite center is the only colored cell having both a same-color
  8-neighbour and a different-color cardinal neighbour.
- Classify its quadrant from the distinct-color 2x2 center pattern.
- Flood the shown color exactly inside an oriented 5x5 canonical window.
- Reflect each recovered component cell into all four quadrants and color it
  from the corresponding 2x2 center cell.
- Render with max reduction so a black quadrant can never erase a valid color
  from another sprite.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name("candidate_sparse_reflection.onnx")

def init(name, arr): return numpy_helper.from_array(np.asarray(arr), name=name)
def vi(name, dtype, shape): return helper.make_tensor_value_info(name, dtype, shape)

def main():
    N=[]; I=[]; V=[]
    def add(op, ins, outs, **kw): N.append(helper.make_node(op, ins, outs, **kw))
    def val(name,dtype,shape): V.append(vi(name,dtype,shape))

    # Color label map: inside color c -> c; outside fixed canvas -> -1.
    Wc=np.arange(1,11,dtype=np.float32).reshape(1,10,1,1)
    I += [init('Wc',Wc), init('Bc',np.array([-1],np.float32)),
          init('sh900',np.array([900],np.int64)),
          init('padamt',np.array([0,124],np.int64)),
          init('padval',np.array(-1,np.int8)),
          init('k33',np.array([33],np.int64)),
          init('k3',np.array([3],np.int64)),
          init('k9',np.array([9],np.int64)),
          init('z8',np.array(0,np.int8)), init('o8',np.array(1,np.int8)),
          init('f0',np.array(0,np.float16))]
    add('Conv',['input','Wc','Bc'],['cf']); val('cf',TensorProto.FLOAT,[1,1,30,30])
    add('Cast',['cf'],['c2d'],to=TensorProto.INT8); val('c2d',TensorProto.INT8,[1,1,30,30])
    add('Reshape',['c2d','sh900'],['g']); val('g',TensorProto.INT8,[900])
    add('Pad',['g','padamt','padval'],['gp'],mode='constant'); val('gp',TensorProto.INT8,[1024])
    add('Cast',['g'],['gf'],to=TensorProto.FLOAT16); val('gf',TensorProto.FLOAT16,[900])
    add('TopK',['gf','k33'],['tv','ti'],axis=0,largest=1,sorted=1)
    val('tv',TensorProto.FLOAT16,[33]); val('ti',TensorProto.INT64,[33])
    add('Cast',['ti'],['t'],to=TensorProto.INT32); val('t',TensorProto.INT32,[33])
    add('Cast',['tv'],['c'],to=TensorProto.INT8); val('c',TensorProto.INT8,[33])

    # Sparse center detection on the 33 colored-cell slots.
    off8=np.array([-1,1,-30,30,-31,-29,29,31],np.int32).reshape(8,1)
    I += [init('OFF8',off8), init('s0',np.array([0],np.int64)),
          init('s4',np.array([4],np.int64)), init('ax0',np.array([0],np.int64))]
    add('Add',['t','OFF8'],['nidx']); val('nidx',TensorProto.INT32,[8,33])
    add('Gather',['gp','nidx'],['NN'],axis=0); val('NN',TensorProto.INT8,[8,33])
    add('Equal',['NN','c'],['eq']); val('eq',TensorProto.BOOL,[8,33])
    add('Cast',['eq'],['eqi'],to=TensorProto.INT8); val('eqi',TensorProto.INT8,[8,33])
    add('Clip',['NN','z8','o8'],['nzu']); val('nzu',TensorProto.INT8,[8,33])
    add('Sub',['nzu','eqi'],['diffu']); val('diffu',TensorProto.INT8,[8,33])
    add('ReduceMax',['eqi','ax0'],['noniso'],keepdims=0); val('noniso',TensorProto.INT8,[33])
    add('Slice',['diffu','s0','s4','ax0'],['diff4s']); val('diff4s',TensorProto.INT8,[4,33])
    add('ReduceMax',['diff4s','ax0'],['diff4'],keepdims=0); val('diff4',TensorProto.INT8,[33])
    add('Mul',['noniso','diff4'],['valid']); val('valid',TensorProto.INT8,[33])
    add('Mul',['valid','c'],['score']); val('score',TensorProto.INT8,[33])
    add('Cast',['score'],['scoref'],to=TensorProto.FLOAT16); val('scoref',TensorProto.FLOAT16,[33])
    add('TopK',['scoref','k3'],['sv','si'],axis=0,largest=1,sorted=1)
    val('sv',TensorProto.FLOAT16,[3]); val('si',TensorProto.INT64,[3])
    add('Greater',['sv','f0'],['avalid']); val('avalid',TensorProto.BOOL,[3])
    add('Gather',['t','si'],['a'],axis=0); val('a',TensorProto.INT32,[3])
    add('Where',['avalid','a','safeA'],['asafe']); val('asafe',TensorProto.INT32,[3])
    add('Gather',['c','si'],['acol'],axis=0); val('acol',TensorProto.INT8,[3])

    # Quadrant from the 2x2 center's different-color neighbour pattern.
    # OFF order: L,R,U,D,NW,NE,SW,SE.
    QT=np.zeros((8,4),np.float16)
    for q,ks in enumerate(((1,3,7),(0,3,6),(2,1,5),(2,0,4))):
        QT[list(ks),q]=1
    I.append(init('QT',QT))
    add('Gather',['diffu','si'],['dsel'],axis=1); val('dsel',TensorProto.INT8,[8,3])
    add('Transpose',['dsel'],['dT'],perm=[1,0]); val('dT',TensorProto.INT8,[3,8])
    add('Cast',['dT'],['df'],to=TensorProto.FLOAT16); val('df',TensorProto.FLOAT16,[3,8])
    add('MatMul',['df','QT'],['Q']); val('Q',TensorProto.FLOAT16,[3,4])
    add('ArgMax',['Q'],['q'],axis=1,keepdims=0); val('q',TensorProto.INT64,[3])

    # Oriented canonical 5x5 window around each shown center.
    WL=np.empty((4,25),np.int32)
    for q0 in range(4):
        sr=-1 if q0<2 else 1
        sc=-1 if q0%2==0 else 1
        WL[q0]=[30*sr*i+sc*j for i in range(5) for j in range(5)]
    I += [init('WLUT',WL), init('axs1',np.array([1],np.int64)),
          init('sh3155',np.array([3,1,5,5],np.int64)),
          init('sh325',np.array([3,25],np.int64))]
    add('Gather',['WLUT','q'],['woff'],axis=0); val('woff',TensorProto.INT32,[3,25])
    add('Unsqueeze',['asafe','axs1'],['a2']); val('a2',TensorProto.INT32,[3,1])
    add('Add',['woff','a2'],['widx']); val('widx',TensorProto.INT32,[3,25])
    add('Gather',['gp','widx'],['wcol'],axis=0); val('wcol',TensorProto.INT8,[3,25])
    add('Unsqueeze',['acol','axs1'],['acol2']); val('acol2',TensorProto.INT8,[3,1])
    add('Equal',['wcol','acol2'],['memb']); val('memb',TensorProto.BOOL,[3,25])
    add('Cast',['memb'],['membu'],to=TensorProto.UINT8); val('membu',TensorProto.UINT8,[3,25])
    add('Reshape',['membu','sh3155'],['mem2d']); val('mem2d',TensorProto.UINT8,[3,1,5,5])
    seed=np.zeros((1,1,5,5),np.uint8); seed[0,0,0,0]=1
    I.append(init('seed',seed))
    add('Mul',['mem2d','seed'],['m0']); val('m0',TensorProto.UINT8,[3,1,5,5])
    prev='m0'
    # Exact for any <=9-cell 8-connected component: maximum graph distance is 7.
    for k in range(1,8):
        p=f'p{k}'; m=f'm{k}'
        add('MaxPool',[prev],[p],kernel_shape=[3,3],pads=[1,1,1,1]); val(p,TensorProto.UINT8,[3,1,5,5])
        add('Mul',[p,'mem2d'],[m]); val(m,TensorProto.UINT8,[3,1,5,5])
        prev=m
    add('Reshape',[prev,'sh325'],['mflat']); val('mflat',TensorProto.UINT8,[3,25])
    add('Cast',['mflat'],['mf'],to=TensorProto.FLOAT16); val('mf',TensorProto.FLOAT16,[3,25])
    add('TopK',['mf','k9'],['mv','mi'],axis=1,largest=1,sorted=1)
    val('mv',TensorProto.FLOAT16,[3,9]); val('mi',TensorProto.INT64,[3,9])
    add('Greater',['mv','f0'],['memv']); val('memv',TensorProto.BOOL,[3,9])

    # Reflection offsets indexed only by canonical (i,j) position.
    TO=np.empty((25,4),np.int32)
    for i in range(5):
      for j in range(5):
        TO[i*5+j]=[-30*i-j,-30*i+j+1,30*i+30-j,30*i+30+j+1]
    I += [init('TOFF',TO), init('centerOff',np.array([0,1,30,31],np.int32)),
          init('center4',np.array([0,1,30,31],np.int32)),
          init('axs12',np.array([1,2],np.int64)), init('axs2',np.array([2],np.int64)),
          init('sh108',np.array([108],np.int64)), init('shout',np.array([1,1,30,30],np.int64)),
          init('safeA',np.array(155,np.int32)), init('zero64',np.array(0,np.int64)),
          init('ar10',np.arange(10,dtype=np.int8).reshape(1,10,1,1))]
    add('Gather',['centerOff','q'],['qoff'],axis=0); val('qoff',TensorProto.INT32,[3])
    add('Sub',['asafe','qoff'],['base']); val('base',TensorProto.INT32,[3])
    add('Unsqueeze',['avalid','axs1'],['av2']); val('av2',TensorProto.BOOL,[3,1])
    add('And',['memv','av2'],['live']); val('live',TensorProto.BOOL,[3,9])
    add('Where',['live','mi','zero64'],['misafe']); val('misafe',TensorProto.INT64,[3,9])
    add('Gather',['TOFF','misafe'],['toff'],axis=0); val('toff',TensorProto.INT32,[3,9,4])
    add('Unsqueeze',['base','axs12'],['base3']); val('base3',TensorProto.INT32,[3,1,1])
    add('Add',['toff','base3'],['tidx']); val('tidx',TensorProto.INT32,[3,9,4])
    add('Reshape',['tidx','sh108'],['tflat']); val('tflat',TensorProto.INT32,[108])
    add('Unsqueeze',['base','axs1'],['base2']); val('base2',TensorProto.INT32,[3,1])
    add('Add',['base2','center4'],['cidx']); val('cidx',TensorProto.INT32,[3,4])
    add('Gather',['gp','cidx'],['cols'],axis=0); val('cols',TensorProto.INT8,[3,4])
    add('Unsqueeze',['cols','axs1'],['cols3']); val('cols3',TensorProto.INT8,[3,1,4])
    add('Unsqueeze',['live','axs2'],['cond']); val('cond',TensorProto.BOOL,[3,9,1])
    add('Cast',['cond'],['condi'],to=TensorProto.INT8); val('condi',TensorProto.INT8,[3,9,1])
    add('Mul',['cols3','condi'],['vals']); val('vals',TensorProto.INT8,[3,9,4])
    add('Reshape',['vals','sh108'],['vflat']); val('vflat',TensorProto.INT8,[108])
    add('ScatterElements',['g','tflat','vflat'],['newg'],axis=0,reduction='max'); val('newg',TensorProto.INT8,[900])
    add('Reshape',['newg','shout'],['out2d']); val('out2d',TensorProto.INT8,[1,1,30,30])
    add('Equal',['out2d','ar10'],['output'])

    graph=helper.make_graph(N,'task285_sparse_reflection',
      [vi('input',TensorProto.FLOAT,[1,10,30,30])],
      [vi('output',TensorProto.BOOL,[1,10,30,30])],initializer=I,value_info=V)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],producer_name='neurogolf-task285-sparse-reflection')
    model.ir_version=10
    onnx.checker.check_model(model,full_check=True)
    model=onnx.shape_inference.infer_shapes(model,strict_mode=True)
    onnx.save(model,OUT)
    print(OUT)

if __name__=='__main__': main()
