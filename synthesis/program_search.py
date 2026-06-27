from __future__ import annotations

from pathlib import Path

from .candidate_program import ProgramCandidate
from .families import all_family_programs
from .features import task_features
from .scoring import rank_program
from .task_data import generate_fresh_pairs, load_arc_gen_pairs, load_public_pairs
from .verifier import verify_program
from .waterfall import waterfall_programs


def _split_pairs(task_dir: Path, fresh_n: int, seed: int):
    public = load_public_pairs(task_dir)
    arc = load_arc_gen_pairs(task_dir)
    fresh = generate_fresh_pairs(task_dir, fresh_n, seed=seed) if fresh_n else []
    return public, arc, fresh


def search_programs(task_dir: Path, config, families: list[str] | None = None) -> list[ProgramCandidate]:
    public, arc, fresh = _split_pairs(task_dir, getattr(config, "fresh_gen_count", 0), getattr(config, "seed", 42))
    available = public + arc
    if not available:
        return []
    feats = task_features(available)
    programs = all_family_programs(available, feats, config, families=families)
    max_programs = getattr(config, "max_programs", 2000)
    max_py = getattr(config, "max_python_candidates", 200)
    if getattr(config, "enable_waterfall_synthesis", True):
        wf = waterfall_programs(
            programs,
            public,
            arc,
            fresh,
            max_depth=getattr(config, "synthesis_max_depth", 3),
            beam_size=getattr(config, "waterfall_beam_size", getattr(config, "program_beam_size", 50)),
            max_programs=getattr(config, "max_waterfall_programs", max_programs),
        )
    else:
        wf = []
    candidates: list[ProgramCandidate] = []
    seen = set()
    best_near: list[ProgramCandidate] = []
    for program in programs:
        key = str(program.to_dict())
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > max_programs:
            break
        pub_right, _pub_fail = verify_program(program, public, mode="official_tensor")
        if pub_right == 0:
            arc_right, fresh_right = 0, 0
        else:
            arc_right, _arc_fail = verify_program(program, arc, mode="official_tensor")
            fresh_right, _fresh_fail = verify_program(program, fresh, mode="grid_relaxed")
        cand = ProgramCandidate(
            program=program,
            public_right=pub_right,
            public_total=len(public),
            arc_right=arc_right,
            arc_total=len(arc),
            fresh_right=fresh_right,
            fresh_total=len(fresh),
            complexity=program.complexity_score(),
        )
        public_wrong = cand.public_total - cand.public_right
        if cand.public_right == 0 and not getattr(config, "allow_bad_synth_probes", False):
            continue
        if public_wrong > 1 and not getattr(config, "allow_near_miss_synth", False):
            continue
        # Keep exact/near-exact candidates, plus a bounded pool of executable near misses for audit and repair.
        if not public or pub_right >= max(1, len(public) - 1):
            candidates.append(cand)
        elif pub_right > 0:
            best_near.append(cand)
        elif len(best_near) < 5:
            best_near.append(cand)
    candidates.extend(sorted(best_near, key=rank_program, reverse=True)[: max_py // 2])
    candidates.extend(wf[:max_py])
    candidates.sort(key=rank_program, reverse=True)
    return candidates[:max_py]
