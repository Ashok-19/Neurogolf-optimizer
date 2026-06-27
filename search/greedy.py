from __future__ import annotations

from ..models import Candidate


def greedy_select(candidates: list[Candidate]) -> Candidate | None:
    accepted = [c for c in candidates if c.accepted and c.cost is not None]
    if accepted:
        return min(accepted, key=lambda c: c.cost or 10**18)
    return None
