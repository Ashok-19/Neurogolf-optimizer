from pathlib import Path
import unittest

from neurogolf_optimizer.models import OptimizerConfig
from neurogolf_optimizer.scoring import delta_points, fresh_rate, is_accepted, is_near_miss, near_miss_category, points


class ScoringTests(unittest.TestCase):
    def test_points_and_delta(self):
        self.assertGreater(points(100), points(1000))
        self.assertGreater(delta_points(1000, 500), 0)

    def test_relaxed_acceptance(self):
        cfg = OptimizerConfig(repo_root=Path("."), fresh_min=0.9)
        validation = {"public_wrong": 0, "arc_gen_wrong": 0, "fresh_right": 180, "fresh_total": 200, "cost": 900, "exact": False}
        self.assertEqual(fresh_rate(validation), 0.9)
        self.assertTrue(is_accepted(validation, 1000, cfg))

    def test_near_miss(self):
        cfg = OptimizerConfig(repo_root=Path("."), fresh_min=0.9)
        validation = {"public_wrong": 0, "arc_gen_wrong": 3, "fresh_right": 190, "fresh_total": 200, "cost": 700}
        self.assertTrue(is_near_miss(validation, 1000, cfg))
        self.assertEqual(near_miss_category(validation, 1000, cfg), "A")

    def test_require_exact_and_fresh_exact(self):
        validation = {"public_wrong": 0, "arc_gen_wrong": 0, "fresh_right": 180, "fresh_total": 200, "cost": 900, "exact": False}
        self.assertFalse(is_accepted(validation, 1000, OptimizerConfig(repo_root=Path("."), require_exact=True)))
        self.assertFalse(is_accepted(validation, 1000, OptimizerConfig(repo_root=Path("."), require_fresh_exact=True)))


if __name__ == "__main__":
    unittest.main()
