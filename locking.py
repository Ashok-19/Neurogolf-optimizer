from __future__ import annotations

import os
import time
from pathlib import Path


class TaskLock:
    def __init__(self, experiments_dir: Path, timeout: float = 120.0):
        self.path = experiments_dir / ".optimizer_lock"
        self.timeout = timeout
        self.acquired = False

    def __enter__(self) -> "TaskLock":
        start = time.time()
        while True:
            try:
                os.mkdir(self.path)
                (self.path / "owner.txt").write_text(f"pid={os.getpid()} time={time.time()}\n")
                self.acquired = True
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"timed out waiting for task lock {self.path}")
                time.sleep(0.25)

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        try:
            for child in self.path.iterdir():
                child.unlink()
            self.path.rmdir()
        finally:
            self.acquired = False
