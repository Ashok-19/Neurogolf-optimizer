#!/usr/bin/env python3
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper, checker
T=Path(__file__).resolve().parent; OUT=T/'work_task117_reflect_v2.onnx'
def I(n,a):return numpy_helper.from_array(np.asarray(a),name=n)
inits=[
 I('zero_u8',np.array(0,np.uint8)), I('one_i64',np.array([1],np.int64)),
 I('four_i64',np.array([4],np.int64)),I('five_i64',np.array([5],np.int64)),I('six_i64',np.array([6],np.int64)),
 I('ten_i64',np.array([10],np.int64)),I('twelve_i64',np.array(12,np.int64)),I('twenty_two_i64',np.array([22],np.int64)),
 I('zero_color',np.zeros((1,1,1,1),np.float32)),I('axes_c13',np.array([1,2,3],np.int64)),I('axes_hw',np.array([2,3],np.int64)),
 I('priority',np.array([[[[40,99,118,97,36],[95,154,173,152,91],[110,169,188,167,106],[85,144,163,142,81],[20,79,98,77,16]]]],np.uint8)),
 I('zero_i64',np.array([0],np.int64)),
 I('fg_kernel',np.array([0,1,1,1,1,1,1,1,1,1],np.float32).reshape(1,10,1,1)),
 I('detector_kernel_f',np.array([[1,-5,1],[-5,1,-5],[1,-5,1]],np.float32).reshape(1,1,3,3)),I('five_f',np.array(5,np.float32)),
 I('starts0',np.array([0],np.int64)),I('ends7',np.array([7],np.int64)),I('starts12',np.array([12],np.int64)),I('ends5',np.array([5],np.int64)),I('step1',np.array([1],np.int64)),I('stepm1',np.array([-1],np.int64)),
 I('axis2',np.array([2],np.int64)),I('axis3',np.array([3],np.int64)),
 I('clear_idx',np.array([[0,0,5,5],[0,0,6,6]],np.int64)),I('clear_upd',np.array([False,False],bool)),
 I('reflect_pads',np.array([0,0,0,0,0,0,6,6],np.int64)),
]
N=[]
def n(op,ins,out,**kw):N.append(helper.make_node(op,ins,[out],name=out,**kw));return out
fgf=n('Conv',['input','fg_kernel'],'fg_f',kernel_shape=[1,1],pads=[-1,-1,-16,-16])
fg=n('Cast',[fgf],'foreground',to=TensorProto.BOOL)
bs=n('Conv',[fgf,'detector_kernel_f'],'body_score_f',kernel_shape=[3,3],pads=[-3,-3,-3,-3])
bh=n('Equal',[bs,'five_f'],'body_hit')
bw=n('Where',[bh,'priority','zero_u8'],'body_weighted')
rs=n('ReduceMax',[bw],'row_scores',axes=[3],keepdims=0);cs=n('ReduceMax',[bw],'col_scores',axes=[2],keepdims=0)
ra=n('ArgMax',[rs],'row_arg_2d',axis=2,keepdims=0);ca=n('ArgMax',[cs],'col_arg_2d',axis=2,keepdims=0)
rl=n('Squeeze',[ra,'one_i64'],'row_local_i64');cl=n('Squeeze',[ca,'one_i64'],'col_local_i64')
cy=n('Add',[rl,'four_i64'],'cy_crop_i64');cx=n('Add',[cl,'four_i64'],'cx_crop_i64')
bye=n('Add',[cy,'one_i64'],'body_end_y');bxe=n('Add',[cx,'one_i64'],'body_end_x')
bst=n('Concat',['one_i64',cy,cx],'body_starts',axis=0);ben=n('Concat',['ten_i64',bye,bxe],'body_ends',axis=0)
bcell=n('Slice',['input',bst,ben,'axes_c13'],'body_cell')
bpres=n('ReduceMax',[bcell],'body_present',axes=[2,3],keepdims=0)
pres=n('ReduceMax',['input'],'present',axes=[2,3],keepdims=0)
pcol=n('Slice',[pres,'one_i64','ten_i64','one_i64'],'present_colors')
legp=n('Sub',[pcol,bpres],'leg_present');legu=n('Unsqueeze',[legp,'axes_hw'],'leg_present_u');ocol=n('Concat',['zero_color',legu],'object_color',axis=1)
st=n('Sub',['six_i64',cy],'shift_top');sl=n('Sub',['six_i64',cx],'shift_left');sb=n('Sub',[cy,'six_i64'],'shift_bottom');sr=n('Sub',[cx,'six_i64'],'shift_right')
lpads=n('Concat',['zero_i64','zero_i64',st,sl,'zero_i64','zero_i64',sb,sr],'local_pads',axis=0)
loc=n('Pad',[fg,lpads],'fg_local',mode='constant')
locu=n('Cast',[loc],'fg_local_u8',to=TensorProto.UINT8)
rhas=n('ReduceMax',[locu],'row_has',axes=[0,1,3],keepdims=0);chas=n('ReduceMax',[locu],'col_has',axes=[0,1,2],keepdims=0)
ru=n('Cast',[rhas],'row_has_u8',to=TensorProto.UINT8);cu=n('Cast',[chas],'col_has_u8',to=TensorProto.UINT8)
rf=n('ArgMax',[ru],'row_first',axis=0,keepdims=0,select_last_index=0);rr=n('ArgMax',[ru],'row_last',axis=0,keepdims=0,select_last_index=1)
cf=n('ArgMax',[cu],'col_first',axis=0,keepdims=0,select_last_index=0);cr=n('ArgMax',[cu],'col_last',axis=0,keepdims=0,select_last_index=1)
rsm=n('Add',[rf,rr],'row_extent_sum');csm=n('Add',[cf,cr],'col_extent_sum')
top=n('Less',[rsm,'twelve_i64'],'source_top');left=n('Less',[csm,'twelve_i64'],'source_left')
rst=n('Where',[top,'starts0','starts12'],'row_start');ren=n('Where',[top,'ends7','ends5'],'row_end');rsp=n('Where',[top,'step1','stepm1'],'row_step')
cst=n('Where',[left,'starts0','starts12'],'col_start');cen=n('Where',[left,'ends7','ends5'],'col_end');csp=n('Where',[left,'step1','stepm1'],'col_step')
qrows=n('Slice',[loc,rst,ren,'axis2',rsp],'quadrant_rows');quad=n('Slice',[qrows,cst,cen,'axis3',csp],'quadrant')
qclear=n('ScatterND',[quad,'clear_idx','clear_upd'],'quadrant_clear')
full=n('Pad',[qclear,'reflect_pads'],'condition_local',mode='reflect')
ct=n('Sub',[cy,'five_i64'],'condition_top');cleft=n('Sub',[cx,'five_i64'],'condition_left');cb=n('Sub',['twenty_two_i64',cy],'condition_bottom');cright=n('Sub',['twenty_two_i64',cx],'condition_right')
cpads=n('Concat',['zero_i64','zero_i64',ct,cleft,'zero_i64','zero_i64',cb,cright],'condition_pads',axis=0)
cond=n('Pad',[full,cpads],'condition',mode='constant')
n('Where',[cond,ocol,'input'],'output')
g=helper.make_graph(N,'task117_reflect_quadrant',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])],inits)
for name,tp,sh in [('fg_local',TensorProto.BOOL,[1,1,13,13]),('quadrant_rows',TensorProto.BOOL,[1,1,7,13]),('quadrant',TensorProto.BOOL,[1,1,7,7]),('quadrant_clear',TensorProto.BOOL,[1,1,7,7]),('condition_local',TensorProto.BOOL,[1,1,13,13]),('condition',TensorProto.BOOL,[1,1,30,30]),('body_cell',TensorProto.FLOAT,[1,9,1,1])]:g.value_info.append(helper.make_tensor_value_info(name,tp,sh))
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)],producer_name='fresh-rule-reflect-quadrant');m.ir_version=8
checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
