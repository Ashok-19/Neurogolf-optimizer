from __future__ import annotations

from dataclasses import dataclass

from .candidate_program import ProgramCandidate
from .dsl import Program, Step
from .scoring import rank_program
from .task_data import GridPair
from .verifier import verify_program


SUPPORTED_LINEAR_OPS = {
    "Recolor",
    "ReplaceColor",
    "Crop",
    "CropNonZeroBBox",
    "CropColorBBox",
    "Flip",
    "Rotate",
    "SelectColor",
    "DrawRectangle",
    "FillRectangle",
    "DrawLine",
    "Tile",
}


@dataclass
class ScoredProgram:
    program: Program
    public_right: int
    public_total: int
    arc_right: int
    arc_total: int
    fresh_right: int
    fresh_total: int
    complexity: int

    def to_candidate(self) -> ProgramCandidate:
        return ProgramCandidate(
            program=self.program,
            public_right=self.public_right,
            public_total=self.public_total,
            arc_right=self.arc_right,
            arc_total=self.arc_total,
            fresh_right=self.fresh_right,
            fresh_total=self.fresh_total,
            complexity=self.complexity,
        )


def _primitive_steps(programs: list[Program]) -> list[Step]:
    steps: list[Step] = []
    seen: set[str] = set()
    for program in programs:
        if len(program.steps) != 2 or program.steps[0].op != "Input":
            continue
        step = program.steps[1]
        if step.op not in SUPPORTED_LINEAR_OPS:
            continue
        key = f"{step.op}:{step.params}"
        if key in seen:
            continue
        seen.add(key)
        steps.append(step)
    return steps


def _program_name(steps: list[Step]) -> str:
    pieces = []
    for step in steps:
        suffix = ""
        if step.op in {"Flip", "Rotate"}:
            suffix = "_" + "_".join(f"{k}{v}" for k, v in sorted(step.params.items()))
        elif step.op in {"ReplaceColor", "CropColorBBox", "SelectColor"}:
            suffix = "_" + "_".join(str(v) for _k, v in sorted(step.params.items()))
        pieces.append(step.op.lower() + suffix)
    return "waterfall_" + "__".join(pieces)


def _compose(steps: list[Step]) -> Program:
    return Program(_program_name(steps), "waterfall", {"depth": len(steps)}, [Step("Input")] + list(steps))


def _score(program: Program, public: list[GridPair], arc: list[GridPair], fresh: list[GridPair]) -> ScoredProgram:
    pub_right, _ = verify_program(program, public, mode="official_tensor")
    arc_right = 0
    fresh_right = 0
    if pub_right:
        arc_right, _ = verify_program(program, arc, mode="official_tensor")
        fresh_right, _ = verify_program(program, fresh, mode="grid_relaxed")
    return ScoredProgram(
        program=program,
        public_right=pub_right,
        public_total=len(public),
        arc_right=arc_right,
        arc_total=len(arc),
        fresh_right=fresh_right,
        fresh_total=len(fresh),
        complexity=program.complexity_score(),
    )


def _beam_key(scored: ScoredProgram) -> tuple:
    cand = scored.to_candidate()
    public_wrong = cand.public_total - cand.public_right
    arc_wrong = cand.arc_total - cand.arc_right
    exact = public_wrong == 0 and arc_wrong == 0
    public_any = cand.public_right > 0
    fresh_rate = 0.0 if cand.fresh_total == 0 else cand.fresh_right / cand.fresh_total
    return (
        1 if exact else 0,
        1 if public_any else 0,
        -public_wrong,
        -arc_wrong,
        fresh_rate,
        -cand.complexity,
    )


def waterfall_programs(
    primitive_programs: list[Program],
    public: list[GridPair],
    arc: list[GridPair],
    fresh: list[GridPair],
    max_depth: int = 3,
    beam_size: int = 80,
    max_programs: int = 1000,
) -> list[ProgramCandidate]:
    steps = _primitive_steps(primitive_programs)
    if max_depth < 2 or not steps:
        return []
    beam: list[list[Step]] = [[]]
    scored_out: list[ScoredProgram] = []
    seen: set[str] = set()
    for depth in range(1, max_depth + 1):
        next_scored: list[ScoredProgram] = []
        attempted_this_depth = 0
        for prefix in beam:
            for step in steps:
                if attempted_this_depth >= max_programs:
                    break
                chain = prefix + [step]
                program = _compose(chain)
                key = str(program.to_dict())
                if key in seen:
                    continue
                seen.add(key)
                attempted_this_depth += 1
                try:
                    scored = _score(program, public, arc, fresh)
                except Exception:
                    continue
                next_scored.append(scored)
                if depth >= 2 and scored.public_right:
                    scored_out.append(scored)
            if attempted_this_depth >= max_programs:
                break
        if not next_scored:
            break
        # Keep imperfect intermediates because a later crop/recolor/transform can fix them.
        beam = [s.program.steps[1:] for s in sorted(next_scored, key=_beam_key, reverse=True)[:beam_size]]
    candidates = [s.to_candidate() for s in scored_out]
    candidates.sort(key=rank_program, reverse=True)
    return candidates
