from __future__ import annotations

import json
import sqlite3
import time
import shutil
from pathlib import Path

from .models import ValidationResult


class ValidationCache:
    def __init__(self, cache_dir: Path, disabled: bool = False):
        self.cache_dir = cache_dir
        self.disabled = disabled
        self.sqlite_path = cache_dir / "validation_cache.sqlite"
        self.jsonl_path = cache_dir / "validation_cache.jsonl"
        self.conn: sqlite3.Connection | None = None
        if not disabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.sqlite_path, timeout=60.0, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=60000")
            self._ensure_schema()

    @classmethod
    def clear(cls, cache_dir: Path) -> None:
        for path in (cache_dir / "validation_cache.sqlite", cache_dir / "validation_cache.sqlite-wal", cache_dir / "validation_cache.sqlite-shm", cache_dir / "validation_cache.jsonl"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _execute(self, sql: str, params: tuple = ()):
        if self.conn is None:
            return None
        delay = 0.05
        for attempt in range(8):
            try:
                return self.conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(delay)
                delay *= 2
        return None

    def _ensure_schema(self) -> None:
        if self.conn is None:
            return
        table = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='validations'").fetchone()
        if table is not None:
            columns = [row[1] for row in self.conn.execute("PRAGMA table_info(validations)").fetchall()]
            if "profile" not in columns:
                self.conn.execute("ALTER TABLE validations RENAME TO validations_legacy")
                self.conn.execute(
                    "CREATE TABLE validations (task TEXT, hash TEXT, profile TEXT, result TEXT, created REAL, PRIMARY KEY(task, hash, profile))"
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO validations(task, hash, profile, result, created) "
                    "SELECT task, hash, 'legacy', result, created FROM validations_legacy"
                )
                self.conn.execute("DROP TABLE validations_legacy")
                return
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS validations (task TEXT, hash TEXT, profile TEXT, result TEXT, created REAL, PRIMARY KEY(task, hash, profile))"
        )

    def get(self, task: str, model_hash: str, profile: str = "default") -> ValidationResult | None:
        if self.disabled or self.conn is None:
            return None
        row = self._execute("SELECT result FROM validations WHERE task=? AND hash=? AND profile=?", (task, model_hash, profile)).fetchone()
        if not row:
            return None
        return ValidationResult(**json.loads(row[0]))

    def put(self, task: str, model_hash: str, result: ValidationResult, profile: str = "default") -> None:
        if self.disabled or self.conn is None:
            return
        payload = json.dumps(result.to_json_dict(), sort_keys=True)
        self._execute(
            "INSERT OR REPLACE INTO validations(task, hash, profile, result, created) VALUES (?, ?, ?, ?, ?)",
            (task, model_hash, profile, payload, time.time()),
        )
        with self.jsonl_path.open("a") as fh:
            fh.write(json.dumps({"task": task, "hash": model_hash, "profile": profile, "result": result.to_json_dict()}, sort_keys=True) + "\n")

    def stats(self) -> dict:
        if self.disabled or self.conn is None:
            return {"disabled": True, "entries": 0}
        row = self._execute("SELECT COUNT(*), COUNT(DISTINCT task), COUNT(DISTINCT profile) FROM validations").fetchone()
        size = self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
        return {
            "disabled": False,
            "entries": int(row[0]),
            "tasks": int(row[1]),
            "profiles": int(row[2]),
            "sqlite_path": str(self.sqlite_path),
            "bytes": size,
        }

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
