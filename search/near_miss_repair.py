from __future__ import annotations

from itertools import combinations

from ..models import Candidate, Mutation


def mutation_risk(mutation: Mutation) -> tuple:
    meta = mutation.metadata
    final = 1 if meta.get("is_graph_output") or meta.get("kind") in {"final_where", "final_identity"} else 0
    dead = int(meta.get("estimated_dead_nodes") or 0) + int(meta.get("estimated_dead_initializers") or 0) * 2
    family_risk = {
        "output_collapse": 100,
        "where_combo": 90,
        "where_ablation": 70,
        "table_shrink": 55,
        "dtype_lowering": 45,
        "concat_pad": 40,
    }.get(mutation.family, 20)
    return (final, family_risk, dead, mutation.priority)


def candidate_repair_score(candidate: Candidate) -> tuple:
    v = candidate.validation
    if v is None:
        return (10**9, 0.0, candidate.cost or 10**18)
    official_wrong = (v.public_wrong or 0) + (v.arc_gen_wrong or 0)
    fresh = 0.0 if not v.fresh_total else (v.fresh_right or 0) / v.fresh_total
    return (official_wrong, -fresh, candidate.cost or 10**18)


def subset_repairs(mutations: list[Mutation], max_repairs: int = 64) -> list[list[Mutation]]:
    if len(mutations) <= 1:
        return []
    risky = sorted(mutations, key=mutation_risk, reverse=True)
    repairs: list[list[Mutation]] = []
    # Restore the highest-risk mutation first, then pairs of risky mutations.
    for restore_count in (1, 2):
        if restore_count >= len(risky):
            continue
        for restored in combinations(risky, restore_count):
            restored_ids = {m.id for m in restored}
            repair = [m for m in mutations if m.id not in restored_ids]
            if repair and repair not in repairs:
                repairs.append(repair)
            if len(repairs) >= max_repairs:
                return repairs
    # Fill remaining budget with all len-1/len-2 subsets.
    for size in (len(mutations) - 1, len(mutations) - 2):
        if size <= 0:
            continue
        for combo in combinations(mutations, size):
            repair = list(combo)
            if repair not in repairs:
                repairs.append(repair)
            if len(repairs) >= max_repairs:
                return repairs
    return repairs
