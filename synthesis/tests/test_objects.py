import unittest

from neurogolf_optimizer.synthesis.objects import extract_color_objects, extract_objects


class ObjectTests(unittest.TestCase):
    def test_extract_objects(self):
        objs = extract_objects([[0, 1, 1], [0, 0, 2], [3, 0, 2]], connectivity=4)
        self.assertEqual(len(objs), 2)
        per_color = extract_color_objects([[0, 1, 1], [0, 0, 2], [3, 0, 2]], connectivity=4)
        self.assertEqual(len(per_color), 3)
        self.assertTrue(any(o.color == 1 and o.area == 2 for o in per_color))


if __name__ == "__main__":
    unittest.main()
