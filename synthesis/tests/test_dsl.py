import unittest

from neurogolf_optimizer.synthesis.dsl import Program, Step


class DslTests(unittest.TestCase):
    def test_run_python_recolor_crop(self):
        p = Program("p", "recolor", steps=[Step("Input"), Step("ReplaceColor", {"old": 1, "new": 2})])
        self.assertEqual(p.run_python([[1, 0]]), [[2, 0]])
        c = Program("c", "crop", steps=[Step("Input"), Step("Crop", {"bbox": (0, 1, 1, 2)})])
        self.assertEqual(c.run_python([[0, 3]]), [[3]])


if __name__ == "__main__":
    unittest.main()
