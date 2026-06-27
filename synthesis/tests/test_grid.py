import unittest

from neurogolf_optimizer.synthesis.grid import bounding_box, crop, flip_h, grid_equal_with_missing_zero, pad_to_30, replace_color, shape


class GridTests(unittest.TestCase):
    def test_grid_ops(self):
        g = [[0, 1], [2]]
        self.assertEqual(shape(g), (2, 2))
        self.assertEqual(bounding_box(g), (0, 0, 2, 2))
        self.assertEqual(crop(g, (0, 1, 1, 2)), [[1]])
        self.assertEqual(replace_color(g, 1, 3)[0][1], 3)
        self.assertEqual(len(pad_to_30(g)), 30)
        self.assertTrue(grid_equal_with_missing_zero([[1]], [[1, 0]]))


if __name__ == "__main__":
    unittest.main()
