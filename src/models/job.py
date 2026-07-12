"""Pydantic models for Job state, metrics, and artifacts.

State is the single source of truth (persisted as state.json per job).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobPhase(str, Enum):
    """Execution phases matching the pipeline's step-by-step trace."""
    QUEUED = "Queued"
    ANALYSIS = "Analysis"
    PORTING = "Porting"
    VALIDATING = "Validating"
    BENCHMARKING = "Benchmarking"
    OPTIMIZING = "Optimizing"
    REPORTING = "Reporting"
    COMPLETED = "Completed"
    FAILED = "Failed"


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SeedId(str, Enum):
    VECTOR_ADD = "vectorAdd"
    TILED_MATMUL = "tiledMatmul"
    REDUCTION = "reduction"
    # Intentional compile-time-gap test case for the repair loop (see
    # src/baseline/pipeline.py's _attempt_hipcc_repair()) — not a
    # benchmark seed. Direct-filename lookup in
    # WorkspaceManager.copy_seed() finds seeds/repairDemo.cu without
    # needing an entry in that method's fallback mapping dict.
    REPAIR_DEMO = "repairDemo"
    # Synthetic multi-file project fixture — seeds/multiFileDemo/ is a
    # DIRECTORY, not a single .cu file (main.cu calls into helper.cu via
    # helper.cuh). Exercises WorkspaceManager.copy_seed()'s directory-copy
    # path and run_hipify()/run_hipcc()'s existing multi-source handling
    # end to end. Not a benchmark seed — see seeds/multiFileDemo/main.cu's
    # header comment.
    MULTI_FILE_DEMO = "multiFileDemo"


class AgentMessage(BaseModel):
    """Structured message for the pipeline's execution trace."""
    id: int
    agent: str
    timestamp: str
    type: Literal["thought", "action", "observation"]
    content: str


class RawMetrics(BaseModel):
    """Raw values captured from amd-smi (or simulated in mock mode).

    Every field is Optional and defaults to None, meaning "not captured" —
    never a guessed/placeholder number. A real run only ever populates a
    field when it was actually parsed from real amd-smi output; if parsing
    misses a field, it stays None and the report renders it as "Not
    captured" rather than silently showing a plausible-looking fake value.
    """
    gpu_utilization_percent: Optional[float] = None
    power_watts_avg: Optional[float] = None
    power_watts_peak: Optional[float] = None
    temperature_c: Optional[float] = None
    memory_used_mb: Optional[float] = None
    clock_sclk_mhz: Optional[float] = None
    clock_mclk_mhz: Optional[float] = None


class DerivedMetrics(BaseModel):
    """Derived efficiency numbers.

    theoretical_peak_gbs / theoretical_peak_tflops are computed at RUNTIME
    from the actual GPU present — see
    src/tools/execution.py's detect_gpu_theoretical_peaks(), which queries
    rocminfo (compute units, max engine clock) + amd-smi (max memory
    clock, memory bus width) and derives both numbers from first
    principles. There is no hardcoded per-SKU table on the primary path:
    it works the same on gfx1100, gfx942, or any future architecture. Both
    fields (and efficiency_percent/efficiency_tflops_percent along with
    them) stay None only if that live query AND its small fallback table
    both come up empty for the detected architecture — never a guessed or
    borrowed-from-another-GPU number. theoretical_peak_source records
    which path actually produced the value ("runtime" / "fallback_table" /
    "mock" / "unavailable") and theoretical_peak_calculation is a
    judge-readable one-liner showing the exact inputs and formula used
    (see GpuTheoreticalPeaks.bandwidth_formula_str() /
    tflops_formula_str()) — both rendered directly in the report so the
    number can be checked, not just trusted. achieved_* fields are only
    ever populated from real parsed binary output, never guessed,
    independent of any of this.
    """
    achieved_bw_gbs: Optional[float] = None
    theoretical_peak_gbs: Optional[float] = None  # populated by runtime hardware query, or left None
    efficiency_percent: Optional[float] = None

    achieved_tflops: Optional[float] = None
    theoretical_peak_tflops: Optional[float] = None  # populated by runtime hardware query, or left None
    efficiency_tflops_percent: Optional[float] = None

    # Which path actually produced theoretical_peak_gbs/tflops, and a
    # human-readable rendering of the exact calculation — see
    # detect_gpu_theoretical_peaks() in src/tools/execution.py.
    theoretical_peak_source: Optional[str] = None
    theoretical_peak_calculation: Optional[str] = None

    # None = the seed binary's own hipEventElapsedTime() line was not found
    # in its stdout (e.g. it crashed before printing) — never a wall-clock
    # or other substitute wearing the "kernel time" label.
    kernel_time_ms: Optional[float] = None
    bytes_moved: Optional[float] = None
    flops: Optional[float] = None


class JobMetrics(BaseModel):
    """Full metrics payload for a job (raw + derived + timeseries)."""
    raw: RawMetrics = Field(default_factory=RawMetrics)
    derived: DerivedMetrics = Field(default_factory=DerivedMetrics)
    timeseries: list[dict[str, Any]] = Field(default_factory=list)  # [{t, util, power, temp, ...}]
    captured_at: datetime = Field(default_factory=datetime.utcnow)


class JobState(BaseModel):
    """Complete job record. Immutable updates preferred (create new or careful patch)."""
    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:12]}")
    seed_id: SeedId
    phase: JobPhase = JobPhase.QUEUED
    status: JobStatus = JobStatus.RUNNING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # THIS job's own recorded mode — set once in run_baseline() from the
    # same MOCK constant (src/tools/execution.py) that governs everything
    # else in that run, and never touched again afterward. Deliberately
    # NOT re-derived from whatever the server happens to be running as
    # later (e.g. when a job is replayed) — a real job stays recorded as
    # real even if viewed from a server that's currently in MOCK mode.
    # Default "MOCK" exists only so older state.json files written before
    # this field existed still load (Pydantic falls back to the default
    # on a missing key) — see the one-time backfill for the 4 real
    # pre-existing job dirs this was retrofitted onto.
    mode: Literal["MOCK", "REAL"] = "MOCK"

    # Execution trace
    messages: list[AgentMessage] = Field(default_factory=list)
    completed_phases: list[JobPhase] = Field(default_factory=list)

    # Results
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    report_md_path: Optional[str] = None
    artifacts_tar_path: Optional[str] = None
    hip_out_dir: Optional[str] = None

    # Porting details (for report + replay)
    hipify_command: Optional[str] = None
    hipcc_command: Optional[str] = None
    # The actual GPU architecture (e.g. "gfx1100", "gfx942") hipcc compiled
    # for on this run — auto-detected at runtime (see
    # src/tools/execution.py's detect_gpu_arch()), never assumed. None if
    # detection failed and compilation fell back to --offload-arch=native.
    gpu_arch: Optional[str] = None
    repair_loops: int = 0

    # CUDA library #includes detected in source (cublas_v2.h/cudnn.h/
    # thrust/*/nccl.h — see src/tools/execution.py's
    # detect_cuda_library_includes()) and their real ROCm equivalents.
    # Empty for every job that doesn't reference any of the 4 — which is
    # every one of the 4 original seeds. One dict per detected library:
    # {cuda_header, rocm_name, rocm_header, link_flag, available,
    # availability_detail, caveat} — all JSON-primitive fields, no nested
    # models, since this is a report-facing fact list, not something
    # anything else in the pipeline branches on.
    library_detections: list[dict[str, Any]] = Field(default_factory=list)

    # Validation
    validation_passed: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    tolerance: float = 1e-5

    # Populated ONLY when the binary printed something to stdout but
    # src/tools/execution.py's parse_binary_output_for_metrics() matched
    # NONE of its known formats (empty dict back) — see
    # src/baseline/pipeline.py's run_baseline() for where this is set.
    # None for every one of the 4 original seeds + multiFileDemo, whose
    # output is always recognized. Exists so a report reader sees the
    # actual unrecognized text directly instead of a bare "Not captured"
    # with no further context prompting a trip to logs/run.log.
    unrecognized_output_snippet: Optional[str] = None

    error: Optional[str] = None

    # Where the workspace lives on disk (absolute path on the instance)
    workspace_dir: Optional[str] = None


class CreateJobRequest(BaseModel):
    seed_id: SeedId


class JobResponse(BaseModel):
    job_id: str
    seed_id: SeedId
    phase: JobPhase
    status: JobStatus
    messages: list[AgentMessage]
    metrics: JobMetrics
    completed_phases: list[JobPhase]
    duration_seconds: float | None = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    rocm_available: bool
    gpu_name: Optional[str] = None
    rocm_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
