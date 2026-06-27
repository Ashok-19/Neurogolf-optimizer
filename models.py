from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizerConfig:
    repo_root: Path
    fresh_min: float = 0.90
    trials: int = 1000
    beam_size: int = 20
    max_combo_size: int = 5
    workers: int = 1
    validation_workers: int = 1
    timeout: int = 1200
    promote: bool = False
    dry_run: bool = False
    require_exact: bool = False
    require_fresh_exact: bool = False
    allow_relaxed_fresh: bool = True
    public_arc_only: bool = False
    no_cache: bool = False
    cache_dir: Path | None = None
    clear_cache: bool = False
    cache_stats: bool = False
    seed: int = 42
    max_candidates_per_family: int = 10000
    max_saved_models_per_task: int = 5000
    stop_after_no_improve: int = 1000
    target_gain: float | None = None
    time_limit_seconds: int | None = None
    allow_nonstrict_shapes: bool = False
    quick: bool = True
    validation_mode: str = "quick"
    search_mode: str = "hybrid"
    families: list[str] = field(default_factory=list)
    resume: bool = False
    run_id: str | None = None
    preset: str | None = None
    keep_rejected: bool = False
    delete_rejected: bool = True
    delete_hopeless_rejected: bool = True
    keep_diagnostic_rejected: bool = True
    keep_near_miss_models: bool = True
    keep_high_cost_probes: bool = False
    max_build_failures: int = 10000
    progress_every: int = 25
    enable_synthesis: bool = False
    synthesis_only: bool = False
    synthesis_families: list[str] = field(default_factory=list)
    max_programs: int = 2000
    max_python_candidates: int = 200
    max_onnx_compiles: int = 20
    fresh_gen_count: int = 0
    program_beam_size: int = 50
    dsl_max_depth: int = 4
    enable_waterfall_synthesis: bool = True
    synthesis_max_depth: int = 3
    waterfall_beam_size: int = 80
    max_waterfall_programs: int = 1000
    allow_constant_output: bool = False
    allow_public_specialization: bool = False
    allow_bad_synth_probes: bool = False
    allow_near_miss_synth: bool = False
    enable_near_miss_repair: bool = True
    max_near_miss_repairs: int = 8

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repo_root"] = str(self.repo_root)
        if self.cache_dir is not None:
            data["cache_dir"] = str(self.cache_dir)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        data = dict(data)
        data["repo_root"] = Path(data["repo_root"])
        if data.get("cache_dir") is not None:
            data["cache_dir"] = Path(data["cache_dir"])
        return cls(**data)


@dataclass
class ValidationResult:
    task: str
    model_path: str
    public_right: int | None
    public_wrong: int | None
    arc_gen_right: int | None
    arc_gen_wrong: int | None
    fresh_right: int | None
    fresh_total: int | None
    memory: int | None
    params: int | None
    cost: int | None
    points: float | None
    exact: bool | None
    returncode: int
    stdout: str
    stderr: str
    error: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mutation:
    id: str
    family: str
    description: str
    priority: float
    estimated_savings: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildResult:
    model: Any | None
    ok: bool
    error: str | None = None
    changed: bool = False
    estimated_savings: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    task: str
    parent_model: str
    model_path: str
    mutations: list[Mutation]
    validation: ValidationResult | None
    build_error: str | None
    model_hash: str | None
    cost: int | None
    delta: float | None
    accepted: bool
    near_miss: bool
    near_miss_category: str | None = None
    parent_hash: str | None = None
    depth: int = 1
    estimated_savings: int | None = None
    promoted: bool = False
    retention_reason: str | None = None
    model_file_exists: bool | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mutations"] = [m.to_json_dict() for m in self.mutations]
        if self.validation:
            data["validation"] = self.validation.to_json_dict()
        return data


@dataclass
class TaskContext:
    task_id: str
    task_dir: Path
    baseline_model_path: Path
    baseline_stats: dict[str, Any]
    experiments_dir: Path
    run_dir: Path
    incumbent_model_path: Path
    incumbent_cost: int
    config: OptimizerConfig
    baseline_cost: int | None = None


@dataclass
class TaskAnalysis:
    task_id: str
    node_count: int
    op_histogram: dict[str, int]
    initializer_count: int
    largest_initializers: list[dict[str, Any]]
    dtype_histogram: dict[str, int]
    where_count: int
    concat_count: int
    conv_count: int
    qlinear_conv_count: int
    conv_transpose_count: int
    gather_count: int
    scatter_count: int
    estimated_float32_intermediate_bytes: int
    estimated_bool_uint8_tensors: int
    large_lookup_table_candidates: list[dict[str, Any]]
    large_zero_false_tensor_candidates: list[dict[str, Any]]
    final_output_producer_pattern: str | None
    tags: list[str]
    param_estimate: int = 0
    duplicate_initializers: list[dict[str, Any]] = field(default_factory=list)
    where_graph_output_count: int = 0
    concat_padding_patterns: list[dict[str, Any]] = field(default_factory=list)
    gather_table_patterns: list[dict[str, Any]] = field(default_factory=list)
    conv_details: list[dict[str, Any]] = field(default_factory=list)
    largest_intermediates: list[dict[str, Any]] = field(default_factory=list)
