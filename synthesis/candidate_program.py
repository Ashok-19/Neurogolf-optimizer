from __future__ import annotations

from dataclasses import dataclass

from .dsl import Program


@dataclass
class ProgramCandidate:
    program: Program
    public_right: int
    public_total: int
    arc_right: int
    arc_total: int
    fresh_right: int
    fresh_total: int
    complexity: int
    estimated_cost: int | None = None
    compiled: bool = False
    compile_error: str | None = None

    @property
    def correctness(self) -> float:
        total = self.public_total + self.arc_total + self.fresh_total
        if total == 0:
            return 0.0
        return (self.public_right + self.arc_right + self.fresh_right) / total

    def to_dict(self) -> dict:
        return {
            "program": self.program.to_dict(),
            "public_right": self.public_right,
            "public_total": self.public_total,
            "arc_right": self.arc_right,
            "arc_total": self.arc_total,
            "fresh_right": self.fresh_right,
            "fresh_total": self.fresh_total,
            "complexity": self.complexity,
            "estimated_cost": self.estimated_cost,
            "compiled": self.compiled,
            "compile_error": self.compile_error,
            "correctness": self.correctness,
        }
