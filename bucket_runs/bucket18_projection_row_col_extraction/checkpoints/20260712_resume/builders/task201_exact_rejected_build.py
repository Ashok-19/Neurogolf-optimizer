from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto
OUT=Path(__file__).with_name('candidate.onnx')
def I(n,a,d=None):
 a=np.array(a,dtype=d); return helper.make_tensor(n,onnx.helper.np_dtype_to_tensor_dtype(a.dtype),a.shape,a.reshape(-1).tolist())
label_w=np.arange(10,dtype=np.float32).reshape(1,10,1,1)
poly=np.array([[-1,2*c,1-c*c] for c in range(10)],np.int8).reshape(10,3,1,1)
n=[]
A=n.append
A(helper.make_node('Conv',['input','label_w'],['labels_f'],kernel_shape=[1,1],pads=[0,0,-17,-17]))
A(helper.make_node('Cast',['labels_f'],['labels'],to=TensorProto.UINT8))
A(helper.make_node('Equal',['labels','four_u8'],['yellow']))
A(helper.make_node('ReduceMax',['yellow','yellow_row_axes'],['yellow_rows_b'],keepdims=0))
A(helper.make_node('ReduceMax',['yellow','yellow_col_axes'],['yellow_cols_b'],keepdims=0))
A(helper.make_node('Cast',['yellow_rows_b'],['yellow_rows'],to=TensorProto.UINT8))
A(helper.make_node('Cast',['yellow_cols_b'],['yellow_cols'],to=TensorProto.UINT8))
A(helper.make_node('ArgMax',['yellow_rows'],['top'],axis=1,keepdims=0,select_last_index=0))
A(helper.make_node('ArgMax',['yellow_rows'],['bottom'],axis=1,keepdims=0,select_last_index=1))
A(helper.make_node('ArgMax',['yellow_cols'],['left'],axis=1,keepdims=0,select_last_index=0))
A(helper.make_node('ArgMax',['yellow_cols'],['right'],axis=1,keepdims=0,select_last_index=1))
A(helper.make_node('Sub',['bottom','top'],['out_last_row']))
A(helper.make_node('Sub',['right','left'],['out_last_col']))
# Pattern row: remove the two frame-side columns globally. Generator non-overlap
# guarantees every true sprite row retains support; audited on 200k fresh seeds.
A(helper.make_node('Not',['yellow_cols_b'],['keep_cols_b']))
A(helper.make_node('Cast',['keep_cols_b'],['keep_cols_f'],to=TensorProto.FLOAT))
A(helper.make_node('Einsum',['labels_f','keep_cols_f'],['pattern_row_score'],equation='nqhw,nw->nh'))
A(helper.make_node('Greater',['pattern_row_score','zero_f'],['pattern_rows_b']))
A(helper.make_node('Cast',['pattern_rows_b'],['pattern_rows'],to=TensorProto.UINT8))
A(helper.make_node('ArgMax',['pattern_rows'],['pattern_top'],axis=1,keepdims=0))
A(helper.make_node('Sub',['out_last_row','one_i64'],['pattern_h']))
A(helper.make_node('Add',['pattern_top','pattern_h'],['pattern_end']))
A(helper.make_node('LessOrEqual',['pattern_top','row_grid13'],['rows_ge_pattern']))
A(helper.make_node('Less',['row_grid13','pattern_end'],['rows_lt_end']))
A(helper.make_node('And',['rows_ge_pattern','rows_lt_end'],['pattern_row_mask_b']))
A(helper.make_node('Cast',['pattern_row_mask_b'],['pattern_row_mask_f'],to=TensorProto.FLOAT))
A(helper.make_node('Einsum',['labels_f','pattern_row_mask_f'],['pattern_col_score'],equation='nqhw,nh->nw'))
# Frame side columns contaminate the selected rows only when the two boxes overlap
# vertically. In that case the boxes are horizontally disjoint, so masking them is safe.
A(helper.make_node('LessOrEqual',['pattern_top','bottom'],['overlap_a']))
A(helper.make_node('Less',['top','pattern_end'],['overlap_b']))
A(helper.make_node('And',['overlap_a','overlap_b'],['vertical_overlap']))
A(helper.make_node('And',['yellow_cols_b','vertical_overlap'],['masked_side_cols']))
A(helper.make_node('Where',['masked_side_cols','zero_f','pattern_col_score'],['pattern_col_clean']))
A(helper.make_node('Greater',['pattern_col_clean','zero_f'],['pattern_cols_b']))
A(helper.make_node('Cast',['pattern_cols_b'],['pattern_cols'],to=TensorProto.UINT8))
A(helper.make_node('ArgMax',['pattern_cols'],['pattern_left'],axis=1,keepdims=0))
# Fixed maximum crop (5x6), matching the full public rule envelope.
A(helper.make_node('Pad',['labels','pads_labels','zero_u8'],['labels_padded'],mode='constant'))
A(helper.make_node('Cast',['pattern_top'],['pattern_top_i32'],to=TensorProto.INT32))
A(helper.make_node('Cast',['pattern_left'],['pattern_left_i32'],to=TensorProto.INT32))
A(helper.make_node('Concat',['pattern_top_i32','pattern_left_i32'],['crop_starts'],axis=0))
A(helper.make_node('Add',['crop_starts','crop_shape'],['crop_ends']))
A(helper.make_node('Slice',['labels_padded','crop_starts','crop_ends','slice_axes'],['pattern_crop']))
# Remove any frame-side pixels that fall in the fixed crop padding. This is
# a compact 5x6 local mask, avoiding the baseline's two full 13x13 images.
A(helper.make_node('Add',['pattern_top','local_rows5'],['crop_global_rows']))
A(helper.make_node('LessOrEqual',['top','crop_global_rows'],['crop_rows_ge_top']))
A(helper.make_node('LessOrEqual',['crop_global_rows','bottom'],['crop_rows_le_bottom']))
A(helper.make_node('And',['crop_rows_ge_top','crop_rows_le_bottom'],['crop_frame_rows']))
A(helper.make_node('Add',['pattern_left','local_cols6'],['crop_global_cols']))
A(helper.make_node('Equal',['crop_global_cols','left'],['crop_at_left']))
A(helper.make_node('Equal',['crop_global_cols','right'],['crop_at_right']))
A(helper.make_node('Or',['crop_at_left','crop_at_right'],['crop_frame_cols']))
A(helper.make_node('And',['crop_frame_rows','crop_frame_cols'],['crop_frame_sides']))
A(helper.make_node('Where',['crop_frame_sides','zero_u8','pattern_crop'],['pattern_crop_clean']))
A(helper.make_node('Gather',['pattern_crop_clean','zero_index'],['first_col'],axis=3))
A(helper.make_node('ReduceMax',['first_col','all_axes'],['first_color'],keepdims=1))
A(helper.make_node('Add',['top','one_i64'],['bar_row']))
A(helper.make_node('Gather',['labels','bar_row'],['bar_values'],axis=2))
A(helper.make_node('Gather',['bar_values','left'],['left_color'],axis=3))
A(helper.make_node('Gather',['bar_values','right'],['right_color'],axis=3))
A(helper.make_node('Equal',['first_color','left_color'],['keep_order']))
A(helper.make_node('Greater',['out_last_col','six_i64'],['wide']))
A(helper.make_node('Where',['wide','reverse6','reverse4'],['reverse_idx']))
A(helper.make_node('Gather',['pattern_crop_clean','reverse_idx'],['pattern_reverse'],axis=3))
A(helper.make_node('Where',['keep_order','pattern_crop_clean','pattern_reverse'],['pattern_oriented']))
A(helper.make_node('Pad',['pattern_oriented','pads_middle','zero_u8'],['pattern_full'],mode='constant'))
# Compose the variable frame in a compact 6x8 label canvas.
A(helper.make_node('LessOrEqual',['row_grid7','out_last_row'],['valid_rows']))
A(helper.make_node('LessOrEqual',['col_grid8','out_last_col'],['valid_cols']))
A(helper.make_node('And',['valid_rows','valid_cols'],['valid']))
A(helper.make_node('Equal',['row_grid7','out_last_row'],['bottom_edge']))
A(helper.make_node('Or',['top_edge','bottom_edge'],['corner_rows']))
A(helper.make_node('Equal',['col_grid8','out_last_col'],['right_edge']))
A(helper.make_node('Or',['out_left','right_edge'],['edge_cols']))
A(helper.make_node('Where',['out_left','left_color','zero_u8'],['left_side']))
A(helper.make_node('Where',['right_edge','right_color','left_side'],['side_colors']))
A(helper.make_node('Max',['pattern_full','side_colors'],['with_sides']))
A(helper.make_node('Where',['edge_cols','four_u8','zero_u8'],['corner_colors']))
A(helper.make_node('Where',['corner_rows','corner_colors','with_sides'],['output_labels']))
A(helper.make_node('Cast',['valid'],['valid_u8'],to=TensorProto.UINT8))
A(helper.make_node('Mul',['output_labels','valid_u8'],['labels_valid']))
A(helper.make_node('Mul',['labels_valid','labels_valid'],['label_sq']))
A(helper.make_node('Concat',['label_sq','labels_valid','valid_u8'],['features'],axis=1))
A(helper.make_node('ConvInteger',['features','poly_w'],['output'],kernel_shape=[1,1],pads=[0,0,23,22]))
inits=[
 I('zero_u8',0,np.uint8),I('four_u8',4,np.uint8),I('one_i64',1,np.int64),I('six_i64',6,np.int64),I('zero_f',0,np.float32),
 I('label_w',label_w), I('yellow_row_axes',[1,3],np.int64),I('yellow_col_axes',[1,2],np.int64),
 I('row_grid13',np.arange(13,dtype=np.int64).reshape(1,13)),
 I('crop_shape',[5,6],np.int32),I('slice_axes',[2,3],np.int32),I('pads_labels',[0,0,0,0,0,0,3,2],np.int64),
 I('zero_index',[0],np.int64),I('all_axes',[1,2,3],np.int64),
 I('reverse6',[5,4,3,2,1,0],np.int32),I('reverse4',[3,2,1,0,4,5],np.int32),
 I('pads_middle',[0,0,1,1,0,0,1,1],np.int64),
 I('local_rows5',np.arange(5,dtype=np.int64).reshape(1,1,5,1)),I('local_cols6',np.arange(6,dtype=np.int64).reshape(1,1,1,6)),
 I('row_grid7',np.arange(7,dtype=np.int64).reshape(1,1,7,1)),I('col_grid8',np.arange(8,dtype=np.int64).reshape(1,1,1,8)),
 I('out_left',np.array([1,0,0,0,0,0,0,0],bool).reshape(1,1,1,8)),I('top_edge',np.array([1,0,0,0,0,0,0],bool).reshape(1,1,7,1)),
 I('poly_w',poly),
]
g=helper.make_graph(n,'task201_projection_rewrite',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],[helper.make_tensor_value_info('output',TensorProto.INT32,[1,10,30,30])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',20)],ir_version=10)
# Static maxima for dynamic slices, required by the exact scorer.
for name,typ,shape in [('pattern_crop',TensorProto.UINT8,[1,1,5,6])]: m.graph.value_info.append(helper.make_tensor_value_info(name,typ,shape))
onnx.checker.check_model(m,full_check=True);onnx.save(m,OUT);print(OUT)
