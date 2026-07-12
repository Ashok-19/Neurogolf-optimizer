from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto

OUT = Path(__file__).with_name('candidate.onnx')

def I(name, arr, dtype=None):
    a = np.asarray(arr, dtype=dtype)
    return helper.make_tensor(name, helper.np_dtype_to_tensor_dtype(a.dtype), a.shape, a.reshape(-1).tolist())

# Fresh deterministic rewrite from the rule.
# A line-color one-hot vector is converted to a channel mask that accepts every
# nonblack color except the line color. Row/column sums under that mask are
# nonzero only on the four hidden lattice rows/columns. RoiAlign then samples
# exactly the compact 4x4 corner state, from which the 3x3 answer is decoded.
poly = np.zeros((10, 3, 1, 1), np.int8)
for c in range(10):
    poly[c, 0, 0, 0] = -1
    poly[c, 1, 0, 0] = 2 * c
    poly[c, 2, 0, 0] = 1 - c * c

nodes = [
    helper.make_node('Conv', ['input', 'label_w'], ['line2'], kernel_shape=[1, 1], pads=[-2, 0, -27, -29]),
    helper.make_node('Conv', ['input', 'label_w'], ['line3'], kernel_shape=[1, 1], pads=[-3, 0, -26, -29]),
    helper.make_node('Conv', ['input', 'label_w'], ['line4'], kernel_shape=[1, 1], pads=[-4, 0, -25, -29]),
    helper.make_node('Max', ['line2', 'line3', 'line4'], ['line_id_f']),
    helper.make_node('Cast', ['line2'], ['p3b'], to=TensorProto.BOOL),
    helper.make_node('Cast', ['line3'], ['p4b'], to=TensorProto.BOOL),
    helper.make_node('Or', ['p3b', 'p4b'], ['p34b']),
    helper.make_node('Not', ['p34b'], ['p5b']),
    helper.make_node('Concat', ['p3b', 'p4b', 'p5b'], ['flags_b'], axis=1),
    helper.make_node('Cast', ['flags_b'], ['flags'], to=TensorProto.FLOAT),
    helper.make_node('Einsum', ['flags', 'period_coef'], ['period'], equation='nzab,zq->nq'),
    helper.make_node('Cast', ['line_id_f'], ['line_id_i32'], to=TensorProto.INT32),
    helper.make_node('OneHot', ['line_id_i32', 'depth10', 'onehot_values'], ['line_onehot'], axis=-1),
    helper.make_node('Sub', ['nonblack', 'line_onehot'], ['other_color_u8']),
    helper.make_node('Cast', ['other_color_u8'], ['other_color'], to=TensorProto.FLOAT),
    helper.make_node('Einsum', ['input', 'other_color'], ['row_hits'], equation='nchw,nabdc->nh'),
    helper.make_node('Einsum', ['input', 'other_color'], ['col_hits'], equation='nchw,nabdc->nw'),
    helper.make_node('Cast', ['row_hits'], ['row_count'], to=TensorProto.UINT8),
    helper.make_node('Cast', ['col_hits'], ['col_count'], to=TensorProto.UINT8),
    helper.make_node('ArgMax', ['row_count'], ['top'], axis=1, keepdims=1),
    helper.make_node('ArgMax', ['col_count'], ['left'], axis=1, keepdims=1),
    helper.make_node('Cast', ['top'], ['top_f'], to=TensorProto.FLOAT),
    helper.make_node('Cast', ['left'], ['left_f'], to=TensorProto.FLOAT),
    helper.make_node('Mul', ['period', 'neg_half'], ['roi_lo']),
    helper.make_node('Mul', ['period', 'three_half'], ['roi_hi']),
    helper.make_node('Add', ['left_f', 'roi_lo'], ['x1']),
    helper.make_node('Add', ['top_f', 'roi_lo'], ['y1']),
    helper.make_node('Add', ['left_f', 'roi_hi'], ['x2']),
    helper.make_node('Add', ['top_f', 'roi_hi'], ['y2']),
    helper.make_node('Concat', ['x1', 'y1', 'x2', 'y2'], ['rois'], axis=1),
    helper.make_node('RoiAlign', ['input', 'rois', 'batch0'], ['sampled'], output_height=4, output_width=4, sampling_ratio=1, spatial_scale=1.0, mode='avg', coordinate_transformation_mode='output_half_pixel'),
    helper.make_node('Einsum', ['sampled', 'label_col'], ['sample_labels_f'], equation='nchw,cq->nqhw'),
    helper.make_node('Cast', ['sample_labels_f'], ['sample_labels'], to=TensorProto.UINT8),
    helper.make_node('Cast', ['line_id_i32'], ['line_id'], to=TensorProto.UINT8),
    helper.make_node('Equal', ['sample_labels', 'line_id'], ['is_line']),
    helper.make_node('Where', ['is_line', 'zero_u8', 'sample_labels'], ['marker']),
    helper.make_node('Slice', ['marker', 'tl_s', 'tl_e', 'axes_hw'], ['tl']),
    helper.make_node('Slice', ['marker', 'tr_s', 'tr_e', 'axes_hw'], ['tr']),
    helper.make_node('Slice', ['marker', 'bl_s', 'bl_e', 'axes_hw'], ['bl']),
    helper.make_node('Slice', ['marker', 'br_s', 'br_e', 'axes_hw'], ['br']),
    helper.make_node('Min', ['tl', 'tr', 'bl', 'br'], ['corner_min']),
    helper.make_node('Max', ['tl', 'tr', 'bl', 'br'], ['corner_max']),
    helper.make_node('Equal', ['corner_min', 'corner_max'], ['corner_equal']),
    helper.make_node('Where', ['corner_equal', 'corner_min', 'zero_u8'], ['labels']),
    helper.make_node('Mul', ['labels', 'labels'], ['label_sq']),
    helper.make_node('Concat', ['label_sq', 'labels', 'ones'], ['features'], axis=1),
    helper.make_node('ConvInteger', ['features', 'poly_w'], ['output'], kernel_shape=[1, 1], pads=[0, 0, 27, 27]),
]

inits = [
    I('label_w', np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
    I('period_coef', np.array([3, 4, 5], np.float32).reshape(3, 1)),
    I('depth10', 10, np.int64),
    I('onehot_values', [0, 1], np.uint8),
    I('nonblack', np.array([0] + [1] * 9, np.uint8)),
    I('neg_half', -0.5, np.float32), I('three_half', 3.5, np.float32),
    I('batch0', [0], np.int64),
    I('label_col', np.arange(10, dtype=np.float32).reshape(10, 1)),
    I('zero_u8', 0, np.uint8),
    I('axes_hw', [2, 3], np.int64),
    I('tl_s', [0, 0], np.int64), I('tl_e', [3, 3], np.int64),
    I('tr_s', [0, 1], np.int64), I('tr_e', [3, 4], np.int64),
    I('bl_s', [1, 0], np.int64), I('bl_e', [4, 3], np.int64),
    I('br_s', [1, 1], np.int64), I('br_e', [4, 4], np.int64),
    I('ones', np.ones((1, 1, 3, 3), np.uint8)),
    I('poly_w', poly),
]

graph = helper.make_graph(nodes, 'task185_linecolor_projection_roi',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', TensorProto.INT32, [1, 10, 30, 30])], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 16)], ir_version=10)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print(OUT)
