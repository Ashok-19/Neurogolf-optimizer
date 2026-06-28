from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .analysis import analyze_model
from .cache import ValidationCache
from .graph_utils import load_model, model_hash, save_model
from .models import Candidate, Mutation, OptimizerConfig, TaskContext, ValidationResult
from .mutation_base import BaseMutationFamily, filter_family_names, mutation_id
from .mutations import default_families
from .mutations.synth_solver import SynthSolver
from .mutations.where_combo import WhereCombo
from .near_miss_lab import try_repair_model
from .promotion import promote_candidate
from .reporting import append_jsonl, candidate_summary, write_json
from .scoring import delta_points, fresh_rate, is_accepted, is_near_miss, near_miss_category
from .search.beam import beam_mutation_sets, prune_beam
from .search.evolutionary import sample_mutation_sets
from .search.near_miss_repair import subset_repairs
from .search.scheduler import read_baseline_cost
from .validate import validate_model


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class NeuroGolfOptimizer:
    def __init__(self, config: OptimizerConfig, cache: ValidationCache | None = None, run_id: str | None = None):
        self.config = config
        self.cache_dir = config.cache_dir or (config.repo_root / ".chatgpt/operations/neurogolf_optimizer/cache")
        if config.clear_cache:
            ValidationCache.clear(self.cache_dir)
        self.cache = cache or ValidationCache(self.cache_dir, disabled=config.no_cache)
        self.run_id = run_id or config.run_id or timestamp()

    def _baseline_stats(self, task_dir: Path) -> dict:
        path = task_dir / "baseline_stats.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def make_context(self, task_id: str) -> TaskContext:
        task_dir = self.config.repo_root / task_id
        experiments_dir = task_dir / "experiments"
        run_dir = experiments_dir / "optimizer_runs" / self.run_id
        experiments_dir.mkdir(exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        baseline = task_dir / "task" / f"{task_id}.onnx"
        baseline_cost = read_baseline_cost(task_dir)
        return TaskContext(
            task_id=task_id,
            task_dir=task_dir,
            baseline_model_path=baseline,
            baseline_stats=self._baseline_stats(task_dir),
            experiments_dir=experiments_dir,
            run_dir=run_dir,
            incumbent_model_path=baseline,
            incumbent_cost=baseline_cost,
            config=self.config,
            baseline_cost=baseline_cost,
        )

    def _load_state(self, ctx: TaskContext) -> dict:
        path = ctx.run_dir / "state.json"
        if self.config.resume and path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as exc:
                append_jsonl(ctx.run_dir / "errors.jsonl", {"event": "state_load_error", "error": repr(exc)})
        return {
            "visited_hashes": [],
            "failed_build_signatures": [],
            "accepted": [],
            "near_misses": [],
            "trials": 0,
            "candidate_index": 0,
            "incumbent_model_path": None,
            "incumbent_cost": None,
        }

    def _save_state(self, ctx: TaskContext, state: dict) -> None:
        state["incumbent_model_path"] = str(ctx.incumbent_model_path)
        state["incumbent_cost"] = ctx.incumbent_cost
        write_json(ctx.run_dir / "state.json", state)

    def choose_incumbent(self, ctx: TaskContext) -> tuple[Path, int, ValidationResult | None]:
        baseline_cost = ctx.baseline_cost or ctx.incumbent_cost
        if baseline_cost <= 0:
            baseline_result = validate_model(ctx.task_dir, ctx.baseline_model_path, self.config, self.cache)
            baseline_cost = baseline_result.cost or 10**18
            ctx.baseline_cost = baseline_cost
            ctx.incumbent_cost = baseline_cost
        candidate_path = ctx.experiments_dir / "candidate.onnx"
        if not candidate_path.exists():
            return ctx.baseline_model_path, baseline_cost, None
        result = validate_model(ctx.task_dir, candidate_path, self.config, self.cache)
        append_jsonl(ctx.run_dir / "results.jsonl", {"event": "existing_candidate", "validation": result.to_json_dict()})
        if is_accepted(result.to_json_dict(), baseline_cost, self.config):
            ctx.incumbent_model_path = candidate_path
            ctx.incumbent_cost = int(result.cost or baseline_cost)
            return candidate_path, ctx.incumbent_cost, result
        return ctx.baseline_model_path, baseline_cost, result

    def _families(self, ctx: TaskContext) -> list[BaseMutationFamily]:
        if self.config.synthesis_only:
            return [SynthSolver()]
        families = default_families()
        try:
            analysis = analyze_model(ctx.task_id, ctx.incumbent_model_path)
            tags = set(analysis.tags)
            priority_names: list[str] = []
            if "where_heavy" in tags or "final_selector" in tags:
                priority_names.extend(["where_ablation", "where_combo", "output_collapse"])
            if "table_heavy" in tags:
                priority_names.extend(["gather_shrink", "table_shrink"])
            if "dtype_heavy" in tags:
                priority_names.extend(["dtype_lowering", "bool_mask_rewrite", "cast_simplify"])
            if "concat_padding" in tags:
                priority_names.append("concat_pad")
            if "conv_heavy" in tags or "qlinear_heavy" in tags:
                priority_names.append("conv_rewrite")
            ordered: list[BaseMutationFamily] = []
            for name in priority_names:
                ordered.extend([f for f in families if f.name == name and f not in ordered])
            ordered.extend([f for f in families if f not in ordered])
            families = ordered
            write_json(ctx.run_dir / "analysis.json", analysis.__dict__)
        except Exception as exc:
            append_jsonl(ctx.run_dir / "errors.jsonl", {"event": "analysis_error", "error": repr(exc)})
        families = filter_family_names(families, self.config.families)
        if self.config.enable_synthesis and not any(f.name == "synth_solver" for f in families):
            families.insert(0, SynthSolver())
        if not self.config.families or "where_combo" in self.config.families or "where" in self.config.families:
            if not any(f.name == "where_combo" for f in families):
                families.append(WhereCombo())
        return families

    def _candidate_path(self, ctx: TaskContext, index: int, family: str) -> Path:
        safe_family = "".join(c if c.isalnum() or c in "._-" else "_" for c in family)
        return ctx.run_dir / f"candidate_{index:06d}_{safe_family}.onnx"

    def _mutation_signature(self, mutations: list[Mutation]) -> str:
        return "|".join(m.id for m in mutations)

    def _apply_mutation_list(self, ctx: TaskContext, parent_model, mutations: list[Mutation], families: dict[str, BaseMutationFamily]):
        current = parent_model
        for mutation in mutations:
            family = families.get(mutation.family)
            if family is None:
                return None, f"unknown family {mutation.family}"
            result = family.apply(ctx, current, mutation)
            if not result.ok or result.model is None:
                return None, result.error or "build failed"
            current = result.model
        return current, None

    def _build_candidate(
        self,
        ctx: TaskContext,
        parent_model_path: Path,
        parent_hash: str,
        parent_model,
        mutations: list[Mutation],
        families: dict[str, BaseMutationFamily],
        candidate_index: int,
        visited_hashes: set[str],
        failed_signatures: set[str],
    ) -> Candidate:
        sig = self._mutation_signature(mutations)
        if sig in failed_signatures:
            return Candidate(ctx.task_id, str(parent_model_path), "", mutations, None, "known failed build signature", None, None, None, False, False, parent_hash=parent_hash, depth=len(mutations))
        built, error = self._apply_mutation_list(ctx, parent_model, mutations, families)
        if error or built is None:
            failed_signatures.add(sig)
            return Candidate(ctx.task_id, str(parent_model_path), "", mutations, None, error, None, None, None, False, False, parent_hash=parent_hash, depth=len(mutations), estimated_savings=sum(m.estimated_savings or 0 for m in mutations))
        path = self._candidate_path(ctx, candidate_index, mutations[-1].family if mutations else "unknown")
        try:
            h = model_hash(built)
            if h in visited_hashes:
                return Candidate(ctx.task_id, str(parent_model_path), str(path), mutations, None, "duplicate model hash", h, None, None, False, False, parent_hash=parent_hash, depth=len(mutations), estimated_savings=sum(m.estimated_savings or 0 for m in mutations))
            visited_hashes.add(h)
            save_model(built, path)
            return Candidate(ctx.task_id, str(parent_model_path), str(path), mutations, None, None, h, None, None, False, False, parent_hash=parent_hash, depth=len(mutations), estimated_savings=sum(m.estimated_savings or 0 for m in mutations))
        except Exception as exc:
            failed_signatures.add(sig)
            return Candidate(ctx.task_id, str(parent_model_path), str(path), mutations, None, f"{type(exc).__name__}: {exc}", None, None, None, False, False, parent_hash=parent_hash, depth=len(mutations))

    def _validate_one(self, ctx: TaskContext, candidate: Candidate, incumbent_cost: int) -> Candidate:
        cache = ValidationCache(self.cache_dir, disabled=self.config.no_cache)
        try:
            validation = validate_model(ctx.task_dir, Path(candidate.model_path), self.config, cache)
        finally:
            cache.close()
        vdict = validation.to_json_dict()
        candidate.validation = validation
        candidate.cost = validation.cost
        candidate.delta = None if validation.cost is None else delta_points(incumbent_cost, validation.cost)
        candidate.accepted = is_accepted(vdict, incumbent_cost, self.config)
        candidate.near_miss = is_near_miss(vdict, incumbent_cost, self.config)
        candidate.near_miss_category = near_miss_category(vdict, incumbent_cost, self.config)
        return candidate

    def _validate_batch(self, ctx: TaskContext, candidates: list[Candidate]) -> list[Candidate]:
        validatable = [c for c in candidates if not c.build_error and c.model_path]
        results = [c for c in candidates if c.build_error or not c.model_path]
        if not validatable:
            return results
        workers = max(1, self.config.validation_workers)
        if workers == 1 or len(validatable) == 1:
            results.extend([self._validate_one(ctx, c, ctx.incumbent_cost) for c in validatable])
            return results
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._validate_one, ctx, c, ctx.incumbent_cost) for c in validatable]
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def _validation_from_repair(self, task: str, model_path: str, data: dict) -> ValidationResult:
        return ValidationResult(
            task=str(data.get("task", task)),
            model_path=str(data.get("model_path", model_path)),
            public_right=data.get("public_right"),
            public_wrong=data.get("public_wrong"),
            arc_gen_right=data.get("arc_gen_right"),
            arc_gen_wrong=data.get("arc_gen_wrong"),
            fresh_right=data.get("fresh_right"),
            fresh_total=data.get("fresh_total"),
            memory=data.get("memory"),
            params=data.get("params"),
            cost=data.get("cost"),
            points=data.get("points"),
            exact=data.get("exact"),
            returncode=int(data.get("returncode", 0) or 0),
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            error=data.get("error"),
        )

    def _near_miss_repair_candidates(self, ctx: TaskContext, candidate: Candidate, incumbent_cost: int) -> list[Candidate]:
        if not self.config.enable_near_miss_repair or self.config.max_near_miss_repairs <= 0:
            return []
        if not candidate.model_path or not Path(candidate.model_path).exists():
            return []
        repair_dir = ctx.run_dir / "auto_repairs" / Path(candidate.model_path).stem
        try:
            repairs = try_repair_model(
                ctx.task_dir,
                Path(candidate.model_path),
                repair_dir,
                self.config,
                self.cache,
                incumbent_cost,
                max_repairs=self.config.max_near_miss_repairs,
                source_mutations=candidate.mutations,
            )
        except Exception as exc:
            append_jsonl(ctx.run_dir / "errors.jsonl", {"event": "near_miss_repair_error", "model": candidate.model_path, "error": repr(exc)})
            return []
        write_json(repair_dir / "repair_results.json", repairs)
        out: list[Candidate] = []
        for repair in repairs:
            append_jsonl(ctx.run_dir / "repair_results.jsonl", repair)
            model_path = repair.get("model_path")
            validation_data = repair.get("validation")
            if not repair.get("ok") or not model_path or not isinstance(validation_data, dict):
                continue
            validation = self._validation_from_repair(ctx.task_id, str(model_path), validation_data)
            vdict = validation.to_json_dict()
            mutation = Mutation(
                id=mutation_id("near_miss_repair", candidate.model_hash or candidate.model_path, repair.get("repair")),
                family="near_miss_repair",
                description=f"auto repair {repair.get('repair')}",
                priority=1200.0,
                estimated_savings=None if validation.cost is None else max(0, incumbent_cost - int(validation.cost)),
                metadata={
                    "repair": repair.get("repair"),
                    "source_model": candidate.model_path,
                    "source_hash": candidate.model_hash,
                    "repair_family": repair.get("repair_family"),
                    "failure_class": repair.get("failure_class"),
                    "internal_or_dense": repair.get("internal_or_dense"),
                    "source_mutation_family": repair.get("source_mutation_family"),
                    "estimated_repair_cost": repair.get("estimated_repair_cost"),
                    "actual_validation_cost": repair.get("actual_validation_cost"),
                    "primary_repair": repair.get("primary_repair"),
                    "cost_gate": repair.get("cost_gate"),
                    "repair_chain": repair.get("repair_chain"),
                    "post_repair_cleanup_steps": repair.get("post_repair_cleanup_steps"),
                    "cost_before_cleanup": repair.get("cost_before_cleanup"),
                    "cost_after_cleanup": repair.get("cost_after_cleanup"),
                    "accepted_after_cleanup": repair.get("accepted_after_cleanup"),
                    "repair_diagnostic_labels": repair.get("repair_diagnostic_labels"),
                    "cost_diagnostics": repair.get("cost_diagnostics"),
                    "repair_metadata": repair.get("repair_metadata"),
                },
            )
            try:
                h = model_hash(Path(model_path))
            except Exception:
                h = None
            out.append(
                Candidate(
                    ctx.task_id,
                    candidate.model_path,
                    str(model_path),
                    candidate.mutations + [mutation],
                    validation,
                    None,
                    h,
                    validation.cost,
                    None if validation.cost is None else delta_points(incumbent_cost, validation.cost),
                    is_accepted(vdict, incumbent_cost, self.config),
                    is_near_miss(vdict, incumbent_cost, self.config),
                    near_miss_category(vdict, incumbent_cost, self.config),
                    parent_hash=candidate.model_hash,
                    depth=candidate.depth + 1,
                    estimated_savings=mutation.estimated_savings,
                )
            )
        return out

    def _diagnostic_retention_reasons(self, candidate: Candidate, incumbent_cost: int) -> list[str]:
        reasons: list[str] = []
        if candidate.accepted:
            return ["accepted"]
        for mutation in candidate.mutations:
            if mutation.family != "synth_solver":
                continue
            score = mutation.metadata.get("program_score") or {}
            if score.get("public_right") == score.get("public_total") and score.get("arc_right") == score.get("arc_total"):
                reasons.append("python_public_arc_exact_but_official_failed")
        if candidate.validation is None:
            return reasons
        v = candidate.validation.to_json_dict()
        if candidate.validation.public_wrong == 0:
            reasons.append("public_exact")
        if candidate.validation.arc_gen_wrong == 0:
            reasons.append("arc_exact")
        if candidate.near_miss_category in {"A", "B", "C", "D"}:
            reasons.append(f"near_miss_category_{candidate.near_miss_category}")
        if fresh_rate(v) >= self.config.fresh_min:
            reasons.append(f"fresh_ge_{self.config.fresh_min:.2f}")
        if candidate.cost is not None and candidate.cost < incumbent_cost:
            reasons.append("lower_than_incumbent_cost")
            if candidate.validation.public_wrong != 0 or candidate.validation.arc_gen_wrong != 0:
                reasons.append("positive_score_gain_but_official_failures")
        return reasons

    def _apply_rejection_retention(self, candidate: Candidate, incumbent_cost: int) -> None:
        path = Path(candidate.model_path) if candidate.model_path else None
        if candidate.accepted:
            candidate.retention_reason = "accepted"
            candidate.model_file_exists = bool(path and path.exists())
            return
        if not path:
            candidate.retention_reason = candidate.build_error or "no_model_file"
            candidate.model_file_exists = False
            return
        if self.config.keep_rejected:
            candidate.retention_reason = "keep_rejected_flag"
            candidate.model_file_exists = path.exists()
            return
        reasons = self._diagnostic_retention_reasons(candidate, incumbent_cost)
        if reasons and self.config.keep_diagnostic_rejected:
            candidate.retention_reason = "|".join(dict.fromkeys(reasons))
            candidate.model_file_exists = path.exists()
            return
        if self.config.delete_rejected and self.config.delete_hopeless_rejected:
            try:
                path.unlink()
            except OSError:
                pass
            candidate.retention_reason = "deleted_hopeless_rejected"
            candidate.model_file_exists = path.exists()
            return
        candidate.retention_reason = "retained_by_policy"
        candidate.model_file_exists = path.exists()

    def _promote(self, ctx: TaskContext, candidate: Candidate, old_cost: int) -> bool:
        if not candidate.accepted or not candidate.model_path or self.config.dry_run or not self.config.promote:
            return False
        src = Path(candidate.model_path)
        repair_meta = candidate.mutations[-1].metadata if candidate.mutations else {}
        metadata = {
            "repair_family": repair_meta.get("repair_family"),
            "repair_name": repair_meta.get("repair"),
            "repair_chain": repair_meta.get("repair_chain"),
            "source_mutation_family": repair_meta.get("source_mutation_family"),
            "optimizer_candidate": candidate_summary(candidate, old_cost),
        }
        result = promote_candidate(
            ctx.task_dir,
            src,
            self.config,
            self.cache,
            old_cost,
            baseline_cost=ctx.baseline_cost,
            metadata=metadata,
            run_best_path=ctx.run_dir / "best.onnx",
        )
        if not result.promoted:
            return False
        candidate.promoted = True
        if result.validation is not None:
            candidate.validation = result.validation
            candidate.cost = result.validation.cost
        if result.row is not None:
            (ctx.run_dir / "best_notes.md").write_text(json.dumps(result.row, indent=2, sort_keys=True) + "\n")
        return True

    def _discover_mutations(self, ctx: TaskContext, parent_model) -> tuple[list[Mutation], dict[str, BaseMutationFamily]]:
        families = self._families(ctx)
        family_map = {f.name: f for f in families}
        discovered: list[Mutation] = []
        for family in families:
            try:
                discovered.extend(family.discover(ctx, parent_model))
            except Exception as exc:
                append_jsonl(ctx.run_dir / "errors.jsonl", {"event": "discover_error", "family": family.name, "error": repr(exc)})
        unique = {m.id: m for m in discovered}
        mutations = sorted(unique.values(), key=lambda m: (m.priority, m.estimated_savings or 0), reverse=True)
        write_json(ctx.run_dir / "mutation_history.json", [m.to_json_dict() for m in mutations[:10000]])
        return mutations, family_map

    def _make_queue(self, base: list[Mutation], beam: list[Candidate], near: list[Candidate], iteration: int) -> list[list[Mutation]]:
        mode = self.config.search_mode
        queue: list[list[Mutation]] = []
        if mode in {"greedy", "beam", "hybrid"}:
            queue.extend([[m] for m in base])
        if mode in {"beam", "hybrid"}:
            queue.extend(beam_mutation_sets(base, beam + near, self.config.max_combo_size, self.config.beam_size))
        if mode in {"evolutionary", "hybrid"}:
            evo_budget = max(self.config.beam_size * 4, min(256, self.config.trials // 4 or 1))
            queue.extend(sample_mutation_sets(base, self.config.max_combo_size, evo_budget, self.config.seed + iteration))
        seen: set[str] = set()
        unique: list[list[Mutation]] = []
        for muts in queue:
            sig = self._mutation_signature(muts)
            if sig not in seen:
                seen.add(sig)
                unique.append(muts)
        unique.sort(
            key=lambda muts: (
                sum(m.estimated_savings or 0 for m in muts),
                sum(m.priority for m in muts),
                -len(muts),
            ),
            reverse=True,
        )
        return unique

    def optimize_task(self, task_id: str) -> dict:
        ctx = self.make_context(task_id)
        write_json(ctx.run_dir / "config.json", self.config.to_json_dict())
        accepted_rows: list[dict] = []
        near_rows: list[dict] = []
        failed_builds: list[dict] = []
        best_candidate: Candidate | None = None
        metrics = {
            "trials": 0,
            "build_success_count": 0,
            "build_fail_count": 0,
            "duplicate_count": 0,
            "validated_count": 0,
            "accepted_count": 0,
            "near_miss_count": 0,
            "synth_candidates_validated": 0,
            "synth_candidates_accepted": 0,
            "repair_candidates_validated": 0,
            "repair_candidates_accepted": 0,
        }
        start_time = time.time()
        try:
            state = self._load_state(ctx)
            incumbent_path, incumbent_cost, _existing = self.choose_incumbent(ctx)
            ctx.incumbent_model_path = Path(state.get("incumbent_model_path") or incumbent_path)
            ctx.incumbent_cost = int(state.get("incumbent_cost") or incumbent_cost)
            if not ctx.incumbent_model_path.exists():
                ctx.incumbent_model_path = incumbent_path
            visited_hashes = set(state.get("visited_hashes", []))
            failed_signatures = set(state.get("failed_build_signatures", []))
            visited_hashes.add(model_hash(ctx.incumbent_model_path))
            accepted_rows = list(state.get("accepted", []))
            near_rows = list(state.get("near_misses", []))
            trials = int(state.get("trials", 0))
            candidate_index = int(state.get("candidate_index", 0))
            no_improve = 0
            total_gain = sum(r.get("delta_points") or 0.0 for r in accepted_rows)
            beam: list[Candidate] = []
            near_candidates: list[Candidate] = []
            exhausted_signatures: set[str] = set()
            iteration = 0

            while trials < self.config.trials and no_improve < self.config.stop_after_no_improve:
                if self.config.time_limit_seconds and time.time() - start_time >= self.config.time_limit_seconds:
                    break
                if self.config.target_gain is not None and total_gain >= self.config.target_gain:
                    break
                parent_path = ctx.incumbent_model_path
                parent_hash = model_hash(parent_path)
                parent = load_model(parent_path)
                base_mutations, family_map = self._discover_mutations(ctx, parent)
                queue = self._make_queue(base_mutations, beam, near_candidates, iteration)
                queue = [muts for muts in queue if self._mutation_signature(muts) not in exhausted_signatures]
                if not queue:
                    break
                iteration += 1
                qpos = 0
                improved_this_parent = False
                while qpos < len(queue) and trials < self.config.trials and no_improve < self.config.stop_after_no_improve:
                    if self.config.time_limit_seconds and time.time() - start_time >= self.config.time_limit_seconds:
                        break
                    batch_sets: list[list[Mutation]] = []
                    while qpos < len(queue) and len(batch_sets) < max(1, self.config.validation_workers) and trials + len(batch_sets) < self.config.trials:
                        muts = queue[qpos]
                        qpos += 1
                        sig = self._mutation_signature(muts)
                        exhausted_signatures.add(sig)
                        if sig in failed_signatures:
                            continue
                        batch_sets.append(muts)
                    if not batch_sets:
                        continue
                    built_candidates: list[Candidate] = []
                    for muts in batch_sets:
                        candidate_index += 1
                        cand = self._build_candidate(ctx, parent_path, parent_hash, parent, muts, family_map, candidate_index, visited_hashes, failed_signatures)
                        if cand.build_error:
                            if "duplicate model hash" in cand.build_error:
                                metrics["duplicate_count"] += 1
                            else:
                                metrics["build_fail_count"] += 1
                            row = candidate_summary(cand, ctx.incumbent_cost)
                            failed_builds.append(row)
                            append_jsonl(ctx.run_dir / "failed_builds.jsonl", row)
                            no_improve += 1
                            continue
                        metrics["build_success_count"] += 1
                        trials += 1
                        built_candidates.append(cand)
                    for candidate in self._validate_batch(ctx, built_candidates):
                        metrics["validated_count"] += 1
                        is_synth_candidate = any(m.family == "synth_solver" for m in candidate.mutations)
                        if is_synth_candidate:
                            metrics["synth_candidates_validated"] += 1
                        if candidate.accepted:
                            metrics["accepted_count"] += 1
                            if is_synth_candidate:
                                metrics["synth_candidates_accepted"] += 1
                            old_cost = ctx.incumbent_cost
                            promoted = self._promote(ctx, candidate, old_cost)
                            candidate.promoted = promoted
                            self._apply_rejection_retention(candidate, ctx.incumbent_cost)
                            row = candidate_summary(candidate, ctx.incumbent_cost)
                            append_jsonl(ctx.run_dir / "results.jsonl", row)
                            accepted_rows.append(row)
                            append_jsonl(ctx.run_dir / "accepted.jsonl", row)
                            best_candidate = candidate
                            total_gain += row.get("delta_points") or 0.0
                            ctx.incumbent_model_path = Path(candidate.model_path)
                            ctx.incumbent_cost = int(candidate.cost or ctx.incumbent_cost)
                            no_improve = 0
                            improved_this_parent = True
                            break
                        self._apply_rejection_retention(candidate, ctx.incumbent_cost)
                        row = candidate_summary(candidate, ctx.incumbent_cost)
                        append_jsonl(ctx.run_dir / "results.jsonl", row)
                        if candidate.near_miss:
                            metrics["near_miss_count"] += 1
                            near_candidates.append(candidate)
                            near_rows.append(row)
                            append_jsonl(ctx.run_dir / "near_misses.jsonl", row)
                            repair_candidates = self._near_miss_repair_candidates(ctx, candidate, ctx.incumbent_cost)
                            metrics["repair_candidates_validated"] += len(repair_candidates)
                            trials += len(repair_candidates)
                            accepted_repair = False
                            for repair_candidate in repair_candidates:
                                if repair_candidate.model_hash:
                                    visited_hashes.add(repair_candidate.model_hash)
                                if repair_candidate.accepted:
                                    metrics["repair_candidates_accepted"] += 1
                                    metrics["accepted_count"] += 1
                                    old_cost = ctx.incumbent_cost
                                    promoted = self._promote(ctx, repair_candidate, old_cost)
                                    repair_candidate.promoted = promoted
                                    self._apply_rejection_retention(repair_candidate, ctx.incumbent_cost)
                                    repair_row = candidate_summary(repair_candidate, ctx.incumbent_cost)
                                    append_jsonl(ctx.run_dir / "results.jsonl", repair_row)
                                    accepted_rows.append(repair_row)
                                    append_jsonl(ctx.run_dir / "accepted.jsonl", repair_row)
                                    best_candidate = repair_candidate
                                    total_gain += repair_row.get("delta_points") or 0.0
                                    ctx.incumbent_model_path = Path(repair_candidate.model_path)
                                    ctx.incumbent_cost = int(repair_candidate.cost or ctx.incumbent_cost)
                                    no_improve = 0
                                    improved_this_parent = True
                                    accepted_repair = True
                                    break
                                self._apply_rejection_retention(repair_candidate, ctx.incumbent_cost)
                                repair_row = candidate_summary(repair_candidate, ctx.incumbent_cost)
                                append_jsonl(ctx.run_dir / "results.jsonl", repair_row)
                                if repair_candidate.near_miss:
                                    near_candidates.append(repair_candidate)
                                    near_rows.append(repair_row)
                                    append_jsonl(ctx.run_dir / "near_misses.jsonl", repair_row)
                            if accepted_repair:
                                break
                            for repair in subset_repairs(candidate.mutations, max_repairs=16):
                                rsig = self._mutation_signature(repair)
                                if rsig not in exhausted_signatures:
                                    queue.insert(qpos, repair)
                        elif candidate.validation and candidate.validation.cost is not None:
                            if self.config.keep_high_cost_probes or candidate.validation.cost < ctx.incumbent_cost:
                                beam.append(candidate)
                        if candidate.validation and candidate.validation.cost is not None and not candidate.accepted:
                            no_improve += 1
                    beam = prune_beam(beam + near_candidates, self.config.beam_size)
                    near_candidates = prune_beam([c for c in near_candidates if c.near_miss], self.config.beam_size)
                    state.update(
                        {
                            "visited_hashes": list(visited_hashes)[-200000:],
                            "failed_build_signatures": list(failed_signatures)[-self.config.max_build_failures :],
                            "accepted": accepted_rows,
                            "near_misses": near_rows[-1000:],
                            "trials": trials,
                            "candidate_index": candidate_index,
                        }
                    )
                    self._save_state(ctx, state)
                    if self.config.progress_every and trials % self.config.progress_every == 0:
                        append_jsonl(ctx.run_dir / "progress.jsonl", {"trials": trials, "no_improve": no_improve, "incumbent_cost": ctx.incumbent_cost, "beam": len(beam), "near": len(near_candidates)})
                    if improved_this_parent:
                        break
                if improved_this_parent:
                    exhausted_signatures.clear()
                    continue
                # Queue exhausted for current parent; regenerate with updated beams until all possible queues are exhausted.
                if qpos >= len(queue):
                    continue

            write_json(ctx.run_dir / "accepted.json", accepted_rows)
            write_json(ctx.run_dir / "near_misses.json", near_rows)
            write_json(ctx.run_dir / "failed_builds.json", failed_builds[-1000:])
            metrics["trials"] = trials
            synth_metrics_path = ctx.run_dir / "synthesis_discovery_metrics.json"
            if synth_metrics_path.exists():
                try:
                    metrics.update(json.loads(synth_metrics_path.read_text()))
                except Exception:
                    pass
            metrics.setdefault("synth_programs_generated", 0)
            metrics.setdefault("synth_programs_python_exact", 0)
            metrics.setdefault("synth_programs_compilable", 0)
            metrics["useful_trial_rate"] = (metrics["validated_count"] / trials) if trials else 0.0
            write_json(ctx.run_dir / "campaign_metrics.json", metrics)
            summary = {
                "task": task_id,
                "run_dir": str(ctx.run_dir),
                "baseline_cost": ctx.baseline_cost,
                "final_incumbent_cost": ctx.incumbent_cost,
                "trials": trials,
                "no_improve": no_improve,
                "accepted": accepted_rows,
                "near_misses": near_rows,
                "best_model": None if best_candidate is None else best_candidate.model_path,
                "promoted": bool(best_candidate and best_candidate.promoted),
                "search_mode": self.config.search_mode,
            }
            write_json(ctx.run_dir / "summary.json", summary)
            if best_candidate is not None and Path(best_candidate.model_path).exists():
                shutil.copy2(best_candidate.model_path, ctx.run_dir / "best.onnx")
            return summary
        except Exception as exc:
            error = {"task": task_id, "run_dir": str(ctx.run_dir), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
            append_jsonl(ctx.run_dir / "errors.jsonl", error)
            write_json(ctx.run_dir / "summary.json", error)
            return error
