import unittest

from neurogolf_optimizer.validate import parse_validation_stdout


class ValidationParserTests(unittest.TestCase):
    def test_parse_indented_json(self):
        stdout = """
noise
{
  "task": 66,
  "model": "experiments/x.onnx",
  "public_right": 3,
  "public_wrong": 0,
  "arc_gen_right": 2,
  "arc_gen_wrong": 0,
  "fresh_right": 180,
  "fresh_total": 200,
  "memory": 10,
  "params": 5,
  "cost": 15,
  "points": 22.0,
  "exact": false
}
"""
        result = parse_validation_stdout("task066", "experiments/x.onnx", stdout)
        self.assertEqual(result.cost, 15)
        self.assertEqual(result.public_wrong, 0)

    def test_parse_failure_is_structured(self):
        result = parse_validation_stdout("task001", "bad.onnx", "not json")
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.cost)


if __name__ == "__main__":
    unittest.main()
