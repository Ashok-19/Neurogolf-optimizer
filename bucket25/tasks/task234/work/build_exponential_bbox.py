#!/usr/bin/env python3
"""Build task234 from scratch using exponential bbox coordinates and rank-1 rendering."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Full validated builder is mirrored from the bucket artifact. It is intentionally
# stored with the model so the structural rewrite remains reproducible.

def init(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)

def build(path: Path) -> None:
    nodes=[]; inits=[]
    def I(name,a,dtype=None): inits.append(init(name,np.asarray(a,dtype=dtype))); return name
    def N(op,inputs,outputs,**attrs): nodes.append(helper.make_node(op,inputs,outputs,**attrs))
    fwd=np.zeros(30,np.float32); rev=np.zeros(30,np.float32)
    for i in range(20): fwd[i]=np.float32(32.0**i); rev[i]=np.float32(32.0**(19-i))
    I('coord_fwd',fwd);I('coord_rev',rev);I('inv_ln32',np.array(1/np.log(32.),np.float32));I('nineteen_i32',np.array(19,np.int32));I('zero_f',np.array(0.,np.float32));I('one_i32',np.array(1,np.int32));I('nonbg10',np.array([False]+[True]*9,np.bool_));I('idx10_u8',np.arange(10,dtype=np.uint8));I('idx30_i32',np.arange(30,dtype=np.int32));I('base_w',np.array([[[1,0,0,0,0,0,0,0,0,0]]],np.float16));I('zero_i64',np.array(0,np.int64))
    N('ReduceSum',['input'],['pixel_all'],axes=[0,2,3],keepdims=0);N('Greater',['pixel_all','zero_f'],['present_raw']);N('And',['present_raw','nonbg10'],['present']);N('Cast',['present'],['present_u8'],to=TensorProto.UINT8);N('ArgMax',['present_u8'],['color0'],axis=0,keepdims=0);N('ArgMax',['present_u8'],['color1'],axis=0,keepdims=0,select_last_index=1);N('Cast',['color0'],['color0_u8'],to=TensorProto.UINT8);N('Cast',['color1'],['color1_u8'],to=TensorProto.UINT8)
    for out,w,eq,reverse in [('rmax','coord_fwd','ncrw,r->c',False),('rmin','coord_rev','ncrw,r->c',True),('cmax','coord_fwd','ncrw,w->c',False),('cmin','coord_rev','ncrw,w->c',True)]:
        N('Einsum',['input',w],[out+'_enc'],equation=eq);N('Log',[out+'_enc'],[out+'_log']);N('Mul',[out+'_log','inv_ln32'],[out+'_q']);N('Cast',[out+'_q'],[out+'_raw'],to=TensorProto.INT32);N('Sub',['nineteen_i32',out+'_raw'],[out]) if reverse else N('Identity',[out+'_raw'],[out])
    N('Gather',['rmax','zero_i64'],['bg_r1'],axis=0);N('Gather',['cmax','zero_i64'],['bg_c1'],axis=0)
    for j in (0,1):
        c=f'color{j}'
        for field in ('rmin','rmax','cmin','cmax'): N('Gather',[field,c],[f'o{j}_{field}'],axis=0)
        N('Gather',['pixel_all',c],[f'o{j}_pix_f'],axis=0);N('Cast',[f'o{j}_pix_f'],[f'o{j}_pix'],to=TensorProto.INT32);N('Sub',[f'o{j}_rmax',f'o{j}_rmin'],[f'o{j}_h0']);N('Add',[f'o{j}_h0','one_i32'],[f'o{j}_h']);N('Sub',[f'o{j}_cmax',f'o{j}_cmin'],[f'o{j}_w0']);N('Add',[f'o{j}_w0','one_i32'],[f'o{j}_w']);N('Mul',[f'o{j}_h',f'o{j}_w'],[f'o{j}_area']);N('Less',[f'o{j}_pix',f'o{j}_area'],[f'o{j}_armed'])
    N('Where',['o0_armed','color0_u8','color1_u8'],['arm_color']);N('Where',['o0_armed','color1_u8','color0_u8'],['pure_color']);N('Where',['o0_armed','o0_pix','o1_pix'],['arm_pix'])
    for field in ('rmin','rmax','cmin','cmax'): N('Where',['o0_armed',f'o0_{field}',f'o1_{field}'],[f'arm_{field}']);N('Where',['o0_armed',f'o1_{field}',f'o0_{field}'],[f'pure_{field}'])
    N('Sub',['arm_rmax','arm_rmin'],['bbox_h0']);N('Add',['bbox_h0','one_i32'],['bbox_h']);N('Sub',['arm_cmax','arm_cmin'],['bbox_w0']);N('Add',['bbox_w0','one_i32'],['bbox_w']);N('Less',['pure_rmax','arm_rmin'],['top_arm']);N('Greater',['pure_rmin','arm_rmax'],['bottom_arm']);N('Less',['pure_cmax','arm_cmin'],['left_arm']);N('Greater',['pure_cmin','arm_cmax'],['right_arm']);N('Or',['top_arm','bottom_arm'],['vertical']);N('Sub',['arm_pix','bbox_h'],['vertical_num']);N('Sub',['arm_pix','bbox_w'],['horizontal_num']);N('Where',['vertical','vertical_num','horizontal_num'],['rect_num']);N('Where',['vertical','bbox_w0','bbox_h0'],['rect_den']);N('Div',['rect_num','rect_den'],['rect_extent']);N('Where',['vertical','rect_extent','bbox_h'],['rect_h']);N('Where',['vertical','bbox_w','rect_extent'],['rect_w']);N('Sub',['rect_h','one_i32'],['rect_hm1']);N('Sub',['rect_w','one_i32'],['rect_wm1']);N('Sub',['arm_rmax','rect_hm1'],['bottom_start']);N('Add',['arm_rmin','rect_hm1'],['top_end']);N('Sub',['arm_cmax','rect_wm1'],['right_start']);N('Add',['arm_cmin','rect_wm1'],['left_end']);N('Where',['bottom_arm','bottom_start','arm_rmin'],['arm_out_r0']);N('Where',['top_arm','top_end','arm_rmax'],['arm_out_r1']);N('Where',['right_arm','right_start','arm_cmin'],['arm_out_c0']);N('Where',['left_arm','left_end','arm_cmax'],['arm_out_c1'])
    N('LessOrEqual',['idx30_i32','bg_r1'],['base_r']);N('LessOrEqual',['idx30_i32','bg_c1'],['base_c'])
    def interval(p,lo,hi): N('GreaterOrEqual',['idx30_i32',lo],[p+'_ge']);N('LessOrEqual',['idx30_i32',hi],[p+'_le']);N('And',[p+'_ge',p+'_le'],[p])
    interval('pure_r','pure_rmin','pure_rmax');interval('pure_c','pure_cmin','pure_cmax');interval('arm_r','arm_out_r0','arm_out_r1');interval('arm_c','arm_out_c0','arm_out_c1')
    for name in ('base_r','pure_r','arm_r','base_c','pure_c','arm_c'): N('Unsqueeze',[name],[name+'_u'],axes=[0,1])
    N('Concat',['base_r_u','pure_r_u','arm_r_u'],['rows_b'],axis=1);N('Concat',['base_c_u','pure_c_u','arm_c_u'],['cols_b'],axis=1);N('Cast',['rows_b'],['rows'],to=TensorProto.FLOAT16);N('Cast',['cols_b'],['cols'],to=TensorProto.FLOAT16);N('Equal',['idx10_u8','pure_color'],['pure_ch_b']);N('Equal',['idx10_u8','arm_color'],['arm_ch_b']);N('Cast',['pure_ch_b'],['pure_ch0'],to=TensorProto.FLOAT16);N('Cast',['arm_ch_b'],['arm_ch0'],to=TensorProto.FLOAT16);N('Unsqueeze',['pure_ch0'],['pure_ch'],axes=[0,1]);N('Unsqueeze',['arm_ch0'],['arm_ch'],axes=[0,1]);N('Sub',['pure_ch','base_w'],['pure_term']);N('Sub',['arm_ch','base_w'],['arm_term']);N('Concat',['base_w','pure_term','arm_term'],['channels'],axis=1);N('Einsum',['rows','cols','channels'],['output'],equation='bkh,bkw,bkc->bchw')
    model=helper.make_model(helper.make_graph(nodes,'task234_exponential_bbox',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.FLOAT16,[1,10,30,30])],inits),opset_imports=[helper.make_opsetid('',12)],producer_name='neurogolf-task234-expbbox');model.ir_version=10;onnx.checker.check_model(model);path.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,path)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).with_name('candidate.onnx'));a=ap.parse_args();build(a.output)
