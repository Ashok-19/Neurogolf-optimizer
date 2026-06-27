from pathlib import Path
import tempfile
import unittest

from neurogolf_optimizer.models import OptimizerConfig
from neurogolf_optimizer.synthesis.program_search import search_programs


class ProgramSearchTests(unittest.TestCase):
    def test_search_identity_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task000"
            (task / "task").mkdir(parents=True)
            (task / "task" / "task000.json").write_text('{"train":[{"input":[[1]],"output":[[1]]}],"test":[]}')
            cfg = OptimizerConfig(repo_root=Path(tmp), max_programs=20, max_python_candidates=10)
            out = search_programs(task, cfg)
            self.assertTrue(any(c.program.family == "identity" for c in out))


if __name__ == "__main__":
    unittest.main()
