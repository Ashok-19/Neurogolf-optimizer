import unittest

import onnx

from neurogolf_optimizer.synthesis.dsl import Program, Step
from neurogolf_optimizer.synthesis.onnx_emit import emit_program
from neurogolf_optimizer.synthesis.onnx_templates import (
    emit_bbox_rectangle,
    emit_compact_recolor,
    emit_fill_bbox,
    emit_flip_h,
    emit_flip_v,
    emit_rotate90,
    emit_rotate180,
    emit_rotate270,
    emit_static_line,
    estimate_model_cost,
)


class OnnxEmitTests(unittest.TestCase):
    def test_identity_and_recolor_emit(self):
        onnx.checker.check_model(emit_program(Program("id", "identity", steps=[Step("Input")])))
        p = Program("r", "recolor", steps=[Step("Input"), Step("ReplaceColor", {"old": 1, "new": 2})])
        onnx.checker.check_model(emit_program(p))
        self.assertLess(estimate_model_cost(emit_compact_recolor({i: i for i in range(10)})), 40522)

    def test_flip_rotate_emitters_compile(self):
        for model in (emit_flip_h(), emit_flip_v(), emit_rotate90(), emit_rotate180(), emit_rotate270()):
            onnx.checker.check_model(model)

    def test_constant_grid_emit_shape(self):
        model = emit_program(Program("c", "tiling", steps=[Step("Constant", {"grid": [[1]]})]))
        onnx.checker.check_model(model)
        self.assertEqual(model.graph.output[0].name, "output")

    def test_bbox_and_line_emitters_compile(self):
        for model in (emit_bbox_rectangle(1, 2), emit_fill_bbox(1, 3), emit_static_line(4, row=7), emit_static_line(5, col=8)):
            onnx.checker.check_model(model)
            self.assertLess(estimate_model_cost(model), 40522)


if __name__ == "__main__":
    unittest.main()
