#!/usr/bin/env python3
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper, checker
TD=Path('/mnt/data/neurogolf_bucket27_work/tasks/task131'); OUT=TD/'work/task131_single_canvas_v7_rebuilt.onnx'

def it(n,a):return numpy_helper.from_array(np.asarray(a),name=n)
I=[
 it('red_sel',np.eye(10,dtype=np.float32)[2]),it('green_sel',np.eye(10,dtype=np.float32)[3]),it('lin30',np.arange(30,dtype=np.float32)),it('half',np.array([.5],np.float32)),
 it('zero_f',np.array(0.,np.float32)),it('zero_i64',np.array([0],np.int64)),it('one_i64',np.array([1],np.int64)),it('three_i64',np.array([3],np.int64)),it('four_i64',np.array([4],np.int64)),it('eighteen_i64',np.array([18],np.int64)),it('thirty_i64',np.array([30],np.int64)),
 it('slice_starts',np.array([3,0,0],np.int64)),it('slice_axes',np.array([1,2,3],np.int64)),it('slice_steps',np.array([1,1,1],np.int64)),it('v_starts',np.array([0,0],np.int64)),it('v_ends',np.array([1,18],np.int64)),it('v_axes',np.array([0,1],np.int64)),
 it('bg_u8',np.array(16,np.uint8)),it('red_u8',np.array(18,np.uint8)),it('green_u8',np.array(19,np.uint8)),it('cyan_u8',np.array(24,np.uint8)),
 it('label_codes',(16+np.arange(10,dtype=np.uint8)).reshape(1,10,1,1))]
N=[]
def n(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw));return out
area=n('Einsum',['input'],'area',equation='bkhw->b')
rc=n('Einsum',['input','red_sel'],'redcount_f',equation='bkhw,k->b')
cs=n('Einsum',['input','red_sel','lin30'],'red_colsum',equation='bkhw,k,w->b')
rs=n('Einsum',['input','red_sel','lin30'],'red_rowsum',equation='bkhw,k,h->b')
c2=n('Einsum',['input','red_sel','lin30','lin30'],'red_col2',equation='bkhw,k,w,w->b')
cm=n('Div',[cs,rc],'red_colmean');rm=n('Div',[rs,rc],'red_rowmean');ec2=n('Div',[c2,rc],'red_ec2');cm2=n('Mul',[cm,cm],'red_colmean2');var=n('Sub',[ec2,cm2],'red_colvar');isv=n('Less',[var,'half'],'isvert')
R_f=n('Where',[isv,cm,rm],'red_axis_f');R=n('Cast',[R_f],'red_axis',to=TensorProto.INT64)
short=n('Cast',[rc],'short_dim',to=TensorProto.INT64);long_f=n('Div',[area,rc],'long_dim_f');long=n('Cast',[long_f],'long_dim',to=TensorProto.INT64)
H=n('Where',[isv,short,long],'height');W=n('Where',[isv,long,short],'width')
ends=n('Concat',['four_i64',H,W],'green_ends',axis=0)
gf=n('Slice',['input','slice_starts',ends,'slice_axes','slice_steps'],'green_f')
g=n('Greater',[gf,'zero_f'],'green')
co30=n('Einsum',['input','green_sel'],'col_occ30',equation='bkhw,k->bw')
ro30=n('Einsum',['input','green_sel'],'row_occ30',equation='bkhw,k->bh')
co18f=n('Slice',[co30,'v_starts','v_ends','v_axes'],'col_occ18_f')
ro18f=n('Slice',[ro30,'v_starts','v_ends','v_axes'],'row_occ18_f')
co18b=n('Greater',[co18f,'zero_f'],'col_occ18_b');ro18b=n('Greater',[ro18f,'zero_f'],'row_occ18_b')
co18=n('Cast',[co18b],'col_occ18',to=TensorProto.UINT8);ro18=n('Cast',[ro18b],'row_occ18',to=TensorProto.UINT8)
lo=n('Where',[isv,co18,ro18],'long_occ')
gmin=n('ArgMax',[lo],'green_min',axis=1,keepdims=0,select_last_index=0);gmax=n('ArgMax',[lo],'green_max',axis=1,keepdims=0,select_last_index=1)
before=n('Less',[gmax,R],'before_red');Rm1=n('Sub',[R,'one_i64'],'red_minus1');Rp1=n('Add',[R,'one_i64'],'red_plus1');shA=n('Sub',[Rm1,gmax],'shift_before');shB=n('Sub',[Rp1,gmin],'shift_after');sh=n('Where',[before,shA,shB],'shift')
sr=n('Where',[isv,'zero_i64',sh],'shift_row');sc=n('Where',[isv,sh,'zero_i64'],'shift_col');nsr=n('Neg',[sr],'neg_shift_row');nsc=n('Neg',[sc],'neg_shift_col')
sp=n('Concat',['zero_i64','zero_i64',sr,sc,'zero_i64','zero_i64',nsr,nsc],'shift_pads',axis=0)
gs=n('Pad',[g,sp],'green_shifted',mode='constant')
nmin=n('Add',[gmin,sh],'new_min');nmax=n('Add',[gmax,sh],'new_max');ca=n('Sub',[nmin,'one_i64'],'cyan_before');cb=n('Add',[nmax,'one_i64'],'cyan_after');cy=n('Where',[before,ca,cb],'cyan_axis')
coord=n('Range',['zero_i64',long,'one_i64'],'long_coord')
rvec=n('Equal',[coord,R],'red_vec');cvec=n('Equal',[coord,cy],'cyan_vec');lv0=n('Where',[rvec,'red_u8','bg_u8'],'line0');lv=n('Where',[cvec,'cyan_u8',lv0],'line_labels')
d2=n('Where',[isv,'one_i64',long],'line_d2');d3=n('Where',[isv,long,'one_i64'],'line_d3');lshape=n('Concat',['one_i64','one_i64',d2,d3],'line_shape',axis=0);lr=n('Reshape',[lv,lshape],'line_reshaped')
gshape=n('Shape',[gf],'green_shape');canvas=n('Expand',[lr,gshape],'line_canvas')
small=n('Where',[gs,'green_u8',canvas],'small_labels')
hp=n('Sub',['thirty_i64',H],'height_pad');wp=n('Sub',['thirty_i64',W],'width_pad');op=n('Concat',['zero_i64','zero_i64','zero_i64','zero_i64','zero_i64','zero_i64',hp,wp],'out_pads',axis=0);outl=n('Pad',[small,op],'out_labels',mode='constant');n('Equal',[outl,'label_codes'],'output')
V=[]
def vi(name,typ,shape):V.append(helper.make_tensor_value_info(name,typ,shape))
for name,typ,shape in [('green_f',TensorProto.FLOAT,[1,1,1,1]),('green',TensorProto.BOOL,[1,1,1,1]),('green_shifted',TensorProto.BOOL,[1,1,1,1]),('green_shape',TensorProto.INT64,[4]),('long_coord',TensorProto.INT64,[18]),('red_vec',TensorProto.BOOL,[18]),('cyan_vec',TensorProto.BOOL,[18]),('line0',TensorProto.UINT8,[18]),('line_labels',TensorProto.UINT8,[18]),('line_reshaped',TensorProto.UINT8,[1,1,1,1]),('line_canvas',TensorProto.UINT8,[1,1,1,1]),('small_labels',TensorProto.UINT8,[1,1,1,1]),('out_labels',TensorProto.UINT8,[1,1,30,30])]:vi(name,typ,shape)
graph=helper.make_graph(N,'task131_single_canvas_v2',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],I,value_info=V)
m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],producer_name='fresh-rule-rewrite');m.ir_version=10
checker.check_model(m,full_check=False);onnx.save(m,OUT);print(OUT)
