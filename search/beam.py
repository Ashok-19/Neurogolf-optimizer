from __future__ import annotations

from itertools import combinations

from ..models import Candidate
from ..models import Mutation


def rank_candidate(candidate: Candidate) -> tuple:
    accepted = 1 if candidate.accepted else 0
    near = 1 if candidate.near_miss else 0
    cost_score = -(candidate.cost or 10**18)
    fresh = 0.0
    if candidate.validation and candidate.validation.fresh_total:
        fresh = candidate.validation.fresh_right / candidate.validation.fresh_total
    return (accepted, near, cost_score, fresh, candidate.delta or 0.0)


def prune_beam(candidates: list[Candidate], beam_size: int) -> list[Candidate]:
    return sorted(candidates, key=rank_candidate, reverse=True)[:beam_size]


def compatible_mutations(mutations: list[Mutation]) -> bool:
    where_outputs: set[str] = set()
    ids: set[str] = set()
    for mutation in mutations:
        if mutation.id in ids:
            return False
        ids.add(mutation.id)
        meta = mutation.metadata
        output = meta.get("output")
        if output and mutation.family in {"where_ablation", "output_collapse"}:
            if output in where_outputs:
                return False
            where_outputs.add(output)
        for choice in meta.get("choices", []):
            out = choice.get("output")
            if out in where_outputs:
                return False
            where_outputs.add(out)
    return True


def beam_mutation_sets(base: list[Mutation], beam_candidates: list[Candidate], max_depth: int, beam_size: int) -> list[list[Mutation]]:
    pool: list[Mutation] = []
    seen: set[str] = set()
    for candidate in prune_beam(beam_candidates, beam_size):
        for mutation in candidate.mutations:
            if mutation.id not in seen:
                pool.append(mutation)
                seen.add(mutation.id)
    for mutation in sorted(base, key=lambda m: (m.priority, m.estimated_savings or 0), reverse=True):
        if mutation.id not in seen:
            pool.append(mutation)
            seen.add(mutation.id)
        if len(pool) >= max(beam_size * 4, 16):
            break
    out: list[list[Mutation]] = []
    for depth in range(2, max_depth + 1):
        for combo in combinations(pool, depth):
            muts = list(combo)
            if compatible_mutations(muts):
                out.append(muts)
            if len(out) >= beam_size * max_depth * 8:
                return out
    return out
