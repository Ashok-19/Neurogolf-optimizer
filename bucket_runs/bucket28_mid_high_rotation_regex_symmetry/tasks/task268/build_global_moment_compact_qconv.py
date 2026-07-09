from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = TASK.parents[1]
sys.path.insert(0, str(ROOT / 'ARC-GEN'))
sys.path.insert(0, str(TASK))
import arcgen_task


def solve_np(grid):
    a = np.asarray(grid, dtype=np.uint8)
    n = a.shape[0]
    bg = a == 0
    colored = ~bg
    row_has = colored.any(axis=1)
    col_has = colored.any(axis=0)
    top = int(np.argmax(row_has))
    bottom = int(n - 1 - np.argmax(row_has[::-1]))
    left = int(np.argmax(col_has))
    right = int(n - 1 - np.argmax(col_has[::-1]))

    top_bg = int(bg[top].sum())
    bottom_bg = int(bg[bottom].sum())
    left_bg = int(bg[:, left].sum())
    right_bg = int(bg[:, right].sum())
    open_top = top_bg > bottom_bg
    open_bottom = bottom_bg > top_bg
    open_left = left_bg > right_bg
    open_right = right_bg > left_bg
    open_ud = open_top or open_bottom
    open_lr = open_left or open_right

    gtop = top + (2 if open_lr else 0)
    gbottom = bottom - (2 if open_lr else 0)
    gleft = left + (2 if open_ud else 0)
    gright = right - (2 if open_ud else 0)

    rr = np.arange(n)[:, None]
    cc = np.arange(n)[None, :]
    bbox = (rr >= top) & (rr <= bottom) & (cc >= left) & (cc <= right)
    gap_cols = (cc >= gleft) & (cc <= gright)
    gap_rows = (rr >= gtop) & (rr <= gbottom)
    rows_out = ((rr < gtop) & open_top) | ((rr > gbottom) & open_bottom)
    cols_out = ((cc < gleft) & open_left) | ((cc > gright) & open_right)
    outside_axis = rows_out | cols_out

    diag_diff_const = (gleft - gtop) if (open_top or open_left) else (gright - gbottom)
    diag_sum_const = (gright + gtop) if (open_top or open_right) else (gleft + gbottom)
    diag_diff = (cc - diag_diff_const) == rr
    diag_sum = (diag_sum_const - cc) == rr
    fan = gap_cols | gap_rows | diag_diff | diag_sum
    fill = bg & (bbox | (outside_axis & fan))
    out = a.copy()
    out[fill] = 4
    return out.tolist()


def check_np(fresh_n=5000):
    d = json.loads((TASK / 'task.json').read_text())
    for split in ('train', 'test', 'arc-gen'):
        for i, pair in enumerate(d.get(split, [])):
            assert solve_np(pair['input']) == pair['output'], (split, i)
    seen = {str(pair['input']) for split in ('train', 'test', 'arc-gen') for pair in d.get(split, [])}
    fresh = 0
    for i in range(30000):
        if fresh >= fresh_n:
            break
        random.seed(10000 + i)
        try:
            pair = arcgen_task.generate()
        except Exception:
            continue
        key = str(pair['input'])
        if key in seen:
            continue
        seen.add(key)
        assert solve_np(pair['input']) == pair['output'], i
        fresh += 1
    assert fresh == fresh_n, fresh
    print('numpy exact public/arc-gen +', fresh, 'fresh')


def init(name, value):
    return numpy_helper.from_array(np.asarray(value), name)


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def build(path: Path):
    nodes = []
    inits = []

    def I(name, value):
        inits.append(init(name, value))
        return name

    def N(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], name=f'n{len(nodes):03d}_{op}', **attrs))
        return output

    I('slice_axes', np.array([1, 2, 3], np.int64))
    I('bg_starts', np.array([0, 0, 0], np.int64))
    I('bg_ends', np.array([1, 10, 10], np.int64))
    I('row_ids', np.arange(10, dtype=np.int8).reshape(1, 1, 10, 1))
    I('col_ids', np.arange(10, dtype=np.int8).reshape(1, 1, 1, 10))
    I('two_i8', np.array([[[[2]]]], np.int8))
    I('color_weights', np.arange(10, dtype=np.float32))
    I('bg_row_axes', np.array([3], np.int64))
    I('bg_col_axes', np.array([2], np.int64))
    I('all_axes', np.array([1, 2, 3], np.int64))

    # Background channel is the only dense compact input tensor. Grid size is
    # recovered globally from sqrt(sum(one-hot input)) instead of projection
    # maxima. The same total population and background population recover the
    # dynamic box color as a global weighted color moment.
    N('Slice', ['input', 'bg_starts', 'bg_ends', 'slice_axes'], 'bg_f32')
    N('Cast', ['bg_f32'], 'bg', to=TensorProto.BOOL)
    N('ReduceSum', ['bg_f32', 'bg_row_axes'], 'bg_rows', keepdims=1)
    N('ReduceSum', ['bg_f32', 'bg_col_axes'], 'bg_cols', keepdims=1)
    N('ReduceSum', ['input', 'all_axes'], 'total_cells', keepdims=1)
    N('Sqrt', ['total_cells'], 'grid_size_f')
    N('Cast', ['grid_size_f'], 'grid_size_i8', to=TensorProto.INT8)
    N('ReduceSum', ['bg_f32', 'all_axes'], 'bg_total', keepdims=1)
    N('Sub', ['total_cells', 'bg_total'], 'colored_count')
    N('Einsum', ['input', 'color_weights'], 'weighted_color_sum', equation='ncrw,c->n')
    N('Div', ['weighted_color_sum', 'colored_count'], 'box_color_f')
    N('Cast', ['box_color_f'], 'box_color_u8', to=TensorProto.UINT8)

    N('Less', ['bg_rows', 'grid_size_f'], 'row_less_bg')
    N('Less', ['bg_cols', 'grid_size_f'], 'col_less_bg')
    N('Less', ['row_ids', 'grid_size_i8'], 'row_in_grid')
    N('Less', ['col_ids', 'grid_size_i8'], 'col_in_grid')
    N('And', ['row_less_bg', 'row_in_grid'], 'in_row')
    N('And', ['col_less_bg', 'col_in_grid'], 'in_col')
    N('Cast', ['in_row'], 'row_active', to=TensorProto.UINT8)
    N('Cast', ['in_col'], 'col_active', to=TensorProto.UINT8)
    N('ArgMax', ['row_active'], 'top', axis=2, keepdims=1, select_last_index=0)
    N('Cast', ['top'], 'top_i8', to=TensorProto.INT8)
    N('ArgMax', ['row_active'], 'bottom', axis=2, keepdims=1, select_last_index=1)
    N('Cast', ['bottom'], 'bottom_i8', to=TensorProto.INT8)
    N('ArgMax', ['col_active'], 'left', axis=3, keepdims=1, select_last_index=0)
    N('Cast', ['left'], 'left_i8', to=TensorProto.INT8)
    N('ArgMax', ['col_active'], 'right', axis=3, keepdims=1, select_last_index=1)
    N('Cast', ['right'], 'right_i8', to=TensorProto.INT8)
    N('And', ['in_row', 'in_col'], 'bbox_span')

    N('GatherElements', ['bg_rows', 'top'], 'top_bg_count', axis=2)
    N('GatherElements', ['bg_rows', 'bottom'], 'bottom_bg_count', axis=2)
    N('GatherElements', ['bg_cols', 'left'], 'left_bg_count', axis=3)
    N('GatherElements', ['bg_cols', 'right'], 'right_bg_count', axis=3)
    N('Greater', ['top_bg_count', 'bottom_bg_count'], 'open_top')
    N('Greater', ['bottom_bg_count', 'top_bg_count'], 'open_bottom')
    N('Greater', ['left_bg_count', 'right_bg_count'], 'open_left')
    N('Greater', ['right_bg_count', 'left_bg_count'], 'open_right')
    N('Or', ['open_top', 'open_bottom'], 'open_ud')
    N('Or', ['open_left', 'open_right'], 'open_lr')
    N('Cast', ['open_lr'], 'open_lr_i8', to=TensorProto.INT8)
    N('Cast', ['open_ud'], 'open_ud_i8', to=TensorProto.INT8)
    N('Mul', ['open_lr_i8', 'two_i8'], 'open_lr2_i8')
    N('Mul', ['open_ud_i8', 'two_i8'], 'open_ud2_i8')
    N('Add', ['top_i8', 'open_lr2_i8'], 'gtop')
    N('Sub', ['bottom_i8', 'open_lr2_i8'], 'gbottom')
    N('Add', ['left_i8', 'open_ud2_i8'], 'gleft')
    N('Sub', ['right_i8', 'open_ud2_i8'], 'gright')

    N('LessOrEqual', ['gleft', 'col_ids'], 'col_ge_gleft')
    N('LessOrEqual', ['col_ids', 'gright'], 'col_le_gright')
    N('And', ['col_ge_gleft', 'col_le_gright'], 'gap_cols_span')
    N('LessOrEqual', ['gtop', 'row_ids'], 'row_ge_gtop')
    N('LessOrEqual', ['row_ids', 'gbottom'], 'row_le_gbottom')
    N('And', ['row_ge_gtop', 'row_le_gbottom'], 'gap_rows_span')
    N('Less', ['row_ids', 'gtop'], 'top_rows')
    N('Greater', ['row_ids', 'gbottom'], 'bottom_rows')
    N('Less', ['col_ids', 'gleft'], 'left_cols')
    N('Greater', ['col_ids', 'gright'], 'right_cols')
    N('And', ['top_rows', 'open_top'], 'top_rows_open')
    N('And', ['bottom_rows', 'open_bottom'], 'bottom_rows_open')
    N('And', ['left_cols', 'open_left'], 'left_cols_open')
    N('And', ['right_cols', 'open_right'], 'right_cols_open')
    N('Or', ['top_rows_open', 'bottom_rows_open'], 'rows_out')
    N('Or', ['left_cols_open', 'right_cols_open'], 'cols_out')
    N('Or', ['rows_out', 'cols_out'], 'outside_axis')

    N('Sub', ['gleft', 'gtop'], 'diag_tl_const')
    N('Sub', ['gright', 'gbottom'], 'diag_br_const')
    N('Add', ['gright', 'gtop'], 'diag_tr_const')
    N('Add', ['gleft', 'gbottom'], 'diag_bl_const')
    N('Or', ['open_top', 'open_left'], 'use_tl_diag')
    N('Or', ['open_top', 'open_right'], 'use_tr_diag')
    N('Cast', ['use_tl_diag'], 'diag_diff_sel', to=TensorProto.INT8)
    N('Sub', ['diag_tl_const', 'diag_br_const'], 'diag_diff_delta')
    N('Mul', ['diag_diff_sel', 'diag_diff_delta'], 'diag_diff_step')
    N('Add', ['diag_br_const', 'diag_diff_step'], 'diag_diff_const')
    N('Cast', ['use_tr_diag'], 'diag_sum_sel', to=TensorProto.INT8)
    N('Sub', ['diag_tr_const', 'diag_bl_const'], 'diag_sum_delta')
    N('Mul', ['diag_sum_sel', 'diag_sum_delta'], 'diag_sum_step')
    N('Add', ['diag_bl_const', 'diag_sum_step'], 'diag_sum_const')
    N('Sub', ['col_ids', 'diag_diff_const'], 'tmp_d')
    N('Equal', ['tmp_d', 'row_ids'], 'diag_diff')
    N('Sub', ['diag_sum_const', 'col_ids'], 'tmp_s')
    N('Equal', ['tmp_s', 'row_ids'], 'diag_sum')
    N('Or', ['gap_cols_span', 'gap_rows_span'], 'gap_any')
    N('Or', ['gap_any', 'diag_diff'], 'fan_a')
    N('Or', ['fan_a', 'diag_sum'], 'fan')
    N('And', ['outside_axis', 'fan'], 'outside')
    N('Or', ['bbox_span', 'outside'], 'fill_area')
    N('And', ['bg', 'fill_area'], 'fill')

    # Compact scalar labels and quadratic final-output renderer.
    N('Cast', ['bg'], 'bg_u8', to=TensorProto.UINT8)
    I('one_u8', np.array(1, np.uint8))
    I('yellow_code', np.array(5, np.uint8))
    N('Add', ['box_color_u8', 'one_u8'], 'box_code')
    N('Where', ['bbox_span', 'box_code', 'bg_u8'], 'box_or_bg')
    N('Where', ['fill', 'yellow_code', 'box_or_bg'], 'labels')
    N('Mul', ['labels', 'labels'], 'labels2')
    N('Concat', ['labels', 'labels2'], 'features', axis=1)

    I('s1', np.array(1.0, np.float32))
    I('xz0', np.array(0, np.uint8))
    I('wz0', np.array(0, np.int8))
    I('yz0', np.array(0, np.uint8))
    W = np.zeros((10, 2, 1, 1), np.int8)
    B = np.zeros((10,), np.int32)
    for ch in range(10):
        t = ch + 1
        W[ch, :, 0, 0] = [2 * t, -1]
        B[ch] = 1 - t * t
    I('W', W)
    I('B', B)
    N('QLinearConv', ['features', 's1', 'xz0', 'W', 's1', 'wz0', 's1', 'yz0', 'B'],
      'output', kernel_shape=[1, 1], pads=[0, 0, 20, 20])

    graph = helper.make_graph(
        nodes,
        'task268_global_moment_compact_qconv',
        [vi('input', TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi('output', TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 18)],
        ir_version=10,
        producer_name='neurogolf_structural_rewrite',
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print('saved', path, 'nodes', len(nodes), 'params', sum(int(np.prod(x.dims or [1])) for x in inits), 'bytes', path.stat().st_size)


if __name__ == '__main__':
    check_np()
    build(HERE / 'candidate_global_moment_compact_qconv.onnx')
