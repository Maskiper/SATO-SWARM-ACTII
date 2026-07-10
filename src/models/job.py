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

    theoretical_peak_gbs / theoretical_peak_tflops are a real GPU's
    published spec-sheet numbers (documented constants, not a
    measurement) — but which numbers depend entirely on which GPU
    architecture is actually detected at runtime (see JobState.gpu_arch
    and src/baseline/pipeline.py's GPU_THEORETICAL_PEAKS). There is no
    hardcoded default here: an MI300X's ~5300 GB/s peak would silently
    misrepresent efficiency on, say, a gfx1100 card, so both fields stay
    None (and efficiency_percent/efficiency_tflops_percent along with
    them) unless the detected architecture has a known, verified entry in
    that lookup table. achieved_* fields are only ever populated from
    real parsed binary output, never guessed, independent of this.
    """
    achieved_bw_gbs: Optional[float] = None
    theoretical_peak_gbs: Optional[float] = None  # populated by arch-keyed lookup, or left None
    efficiency_percent: Optional[float] = None

    achieved_tflops: Optional[float] = None
    theoretical_peak_tflops: Optional[float] = None  # populated by arch-keyed lookup, or left None
    efficiency_tflops_percent: Optional[float] = None

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

    # Validation
    validation_passed: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    tolerance: float = 1e-5

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
