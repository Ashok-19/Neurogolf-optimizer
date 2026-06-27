from __future__ import annotations

from .candidate_program import ProgramCandidate


def rank_program(candidate: ProgramCandidate) -> tuple:
    public_wrong = candidate.public_total - candidate.public_right
    arc_wrong = candidate.arc_total - candidate.arc_right
    official_exact = public_wrong == 0 and arc_wrong == 0
    public_exact_low_arc = public_wrong == 0 and arc_wrong <= 5
    public_near = public_wrong <= 1
    fresh_rate = 0.0 if candidate.fresh_total == 0 else candidate.fresh_right / candidate.fresh_total
    return (
        1 if official_exact else 0,
        1 if public_exact_low_arc else 0,
        1 if public_near else 0,
        -public_wrong,
        -arc_wrong,
        fresh_rate,
        -(candidate.estimated_cost or 10**12),
        -candidate.complexity,
    )
