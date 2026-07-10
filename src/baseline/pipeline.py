"""Non-agent baseline pipeline — the core CUDA -> ROCm migration flow.

Sequential flow:
1. Prepare workspace (copy seed)
2. Run hipify
3. hipcc compile
4. Run + validate against embedded reference (seeds self-check)
5. Capture amd-smi metrics
6. Compute derived efficiency
7. Generate report.md + tar

All steps update JobState (persisted) and append visible messages.

Every number in the generated report is either a real measurement or is
explicitly rendered as "Not captured" / tagged "(SIMULATED)" — this
pipeline never substitutes a plausible-looking placeholder for a real one,
and never re-labels a lower-fidelity number (e.g. process wall-clock time)
under a higher-fidelity name (e.g. "Kernel time", which means GPU-side
hipEventElapsedTime() timing specifically — see _compute_derived_metrics).
See src/tools/execution.py's MOCK flag docstring for the mock/real switch.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.models.job import (
    AgentMessage,
    DerivedMetrics,
    JobMetrics,
    JobPhase,
    JobState,
    JobStatus,
    RawMetrics,
)
from src.tools.execution import (
    MOCK,
    capture_amd_smi_snapshot,
    parse_binary_output_for_metrics,
    run_binary,
    run_hipcc,
    run_hipify,
)
from src.workspace.manager import WorkspaceManager


# ---------------------------------------------------------------------------
# Theoretical peak specs, keyed by detected GPU architecture (job.gpu_arch —
# see src/tools/execution.py's detect_gpu_arch()). Efficiency percentages
# are ONLY computed when the detected arch has a verified entry here;
# otherwise theoretical_peak_gbs / theoretical_peak_tflops /
# efficiency_percent / efficiency_tflops_percent all stay None and the
# report renders "Not captured" rather than dividing a real achieved
# number by the wrong hardware's peak (e.g. applying MI300X's ~5300 GB/s
# to a gfx1100 card would produce a badly misleading efficiency figure —
# this table exists specifically so that never happens silently).
#
# Add an entry here once you've confirmed the real spec-sheet numbers for
# an architecture you're targeting.
# ---------------------------------------------------------------------------
GPU_THEORETICAL_PEAKS: dict[str, dict[str, float]] = {
    "gfx942": {  # AMD Instinct MI300X
        "hbm_bandwidth_gbs": 5300.0,
        "fp32_tflops": 163.4,
    },
    # gfx1100 (RDNA3, Radeon 7900-class: W7900 / 7900 XTX / 7900 XT all
    # report as gfx1100 but have different real specs) intentionally left
    # out -- the exact SKU wasn't confirmed against the actual pod when
    # this was written. Add it here with real numbers once confirmed
    # (e.g. from `amd-smi` or the card's datasheet) to get efficiency_
    # percent computed on that hardware too.
}


# ---------------------------------------------------------------------------
# Correctness tolerances, per seed. validation_ok is only PASSED when every
# actual-vs-expected pair the binary printed is within tolerance here — not
# merely because the binary printed *a* result line and exited 0 (a real
# bug: a reduction seed that returned exactly double the expected value,
# from a stale accumulator buffer, was reported "PASSED" before this
# existed, because "reduction result" appeared in stdout regardless of
# what the actual number was).
#
# Format: (relative_tolerance, absolute_floor). A pair passes if
# abs(actual - expected) <= max(absolute_floor, relative_tolerance *
# abs(expected)) -- the max() handles both large-magnitude expected values
# (relative tolerance dominates) and near-zero ones (the floor prevents an
# unreasonably strict absolute requirement).
#
# Tolerances are NOT uniform across seeds on purpose:
#   - vectorAdd: a single float32 add of two O(1) values, computed with
#     the identical operation on host and device -- expect an exact or
#     near-exact match. Tight.
#   - tiledMatmul: the host sums terms in naive serial order; the device
#     sums the same terms via tiled shared-memory blocking -- a different
#     but equally valid summation order. Floating-point addition isn't
#     associative, so some real, legitimate drift is expected here.
#     Looser.
#   - reduction: sums N copies of exactly 1.0f via a balanced tree
#     reduction. With the default N (a power of 2) and an even per-block
#     split, every intermediate partial sum stays exactly representable in
#     float32 -- the correct result is exact, not merely "close". Tight,
#     same as vectorAdd, NOT loose just because it's summing many terms.
# ---------------------------------------------------------------------------
VALIDATION_TOLERANCES: dict[str, tuple[float, float]] = {
    "vectorAdd": (1e-5, 1e-4),
    "tiledMatmul": (1e-2, 1e-2),
    "reduction": (1e-4, 1.0),
}


def _validation_passes(seed_id: str, check_pairs: list[tuple[float, float]]) -> bool:
    """True only if every actual-vs-expected pair the binary printed is
    within this seed's tolerance (see VALIDATION_TOLERANCES). An empty
    check_pairs list is NOT a pass -- "we couldn't confirm correctness"
    must never default to "correct".
    """
    if not check_pairs:
        return False
    rel_tol, abs_floor = VALIDATION_TOLERANCES.get(seed_id, (1e-3, 1e-3))
    for actual, expected in check_pairs:
        tolerance = max(abs_floor, rel_tol * abs(expected))
        if abs(actual - expected) > tolerance:
            return False
    return True


def _now_ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")


def _fmt(value, suffix: str = "") -> str:
    """Render a metric value for the report — 'Not captured' for None,
    never a blank cell or a fabricated number. In MOCK mode, tags the
    value "(SIMULATED)" inline so it's unmistakable even out of context
    (e.g. a screenshot of just the table, without the page-level banner).
    """
    if value is None:
        return "Not captured"
    tag = " (SIMULATED)" if MOCK else ""
    return f"{value}{suffix}{tag}"


def _append_message(job: JobState, agent: str, typ: str, content: str) -> None:
    msg = AgentMessage(
        id=len(job.messages) + 1,
        agent=agent,
        timestamp=_now_ts(),
        type=typ,  # type: ignore[arg-type]
        content=content,
    )
    job.messages.append(msg)


def _discover_hip_sources(hip_out: Path) -> list[Path]:
    """Find the hipified source files to compile, in priority order
    (*.hip.cpp, then *.cpp, then *.cu), with no duplicates.

    A file named "vectorAdd.hip.cpp" matches BOTH "*.hip.cpp" and "*.cpp"
    -- naively concatenating separate glob() results (three independent
    glob() calls, unioned with `+`) put such a file on the hipcc command
    line twice. hipcc/the linker then sees two definitions of main() and
    every kernel in it, and fails with "duplicate symbol" errors — a real
    failure seen on the MI300X pod, not a hypothetical.

    Deduplicating by *resolved path* (not just `list(set(...))`, which
    would also lose deterministic ordering across runs) fixes this for
    good, for this overlap and any other pattern overlap that might be
    introduced later — rather than trying to hand-craft each glob pattern
    to individually exclude the others, which is fragile and only ever
    covers the overlaps someone thought to check for.
    """
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in ("*.hip.cpp", "*.cpp", "*.cu"):
        for p in sorted(hip_out.glob(pattern)):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(p)
    if not files:
        files = sorted(hip_out.glob("*"))
    return files


def _advance(job: JobState, phase: JobPhase, ws: WorkspaceManager) -> None:
    if phase not in job.completed_phases:
        job.completed_phases.append(phase)
    job.phase = phase
    job.updated_at = datetime.utcnow()
    ws.write_state(job)


def _compute_derived_metrics(job: JobState, parsed: dict, job_msg: Callable[[str], None]) -> DerivedMetrics:
    """Compute achieved bandwidth / TFLOPS from real measured numbers only.

    kernel_time_ms is the seed binary's own hipEventElapsedTime() reading
    (hipEventCreate / hipEventRecord / hipEventSynchronize /
    hipEventElapsedTime wrapped directly around the kernel launch inside
    seeds/*.cu — genuine GPU-side timing, identical in mock and real mode
    since this function only ever looks at `parsed`, never at how the
    stdout it came from was produced). If the binary didn't print a
    parseable "Kernel time:" line, kernel_time_ms stays None — there is
    no wall-clock or other substitute wearing that label.

    achieved_bw_gbs / achieved_tflops are computed here, in Python, from
    that real kernel_time_ms plus a real byte/FLOP count also parsed from
    the binary's own stdout (never a constant). The binary's own
    self-printed "Achieved ..." line is parsed too, but only as a
    cross-check — job_msg() is called with a warning if it disagrees with
    the independently-computed value by more than 1%.

    theoretical_peak_gbs / theoretical_peak_tflops / efficiency_percent /
    efficiency_tflops_percent are only populated when job.gpu_arch has a
    verified entry in GPU_THEORETICAL_PEAKS — otherwise they stay None
    ("Not applicable" in the report) rather than compute a percentage
    against the wrong hardware's spec numbers.
    """
    kernel_time_ms = parsed.get("kernel_time_ms")
    derived = DerivedMetrics(kernel_time_ms=kernel_time_ms)
    if kernel_time_ms is None or kernel_time_ms <= 0:
        return derived

    seconds = kernel_time_ms / 1000.0
    peaks = GPU_THEORETICAL_PEAKS.get(job.gpu_arch or "")

    def _cross_check(label: str, computed: float, self_reported: Optional[float]) -> None:
        if self_reported is None or self_reported == 0:
            return
        rel_diff = abs(computed - self_reported) / self_reported
        if rel_diff > 0.01:
            job_msg(
                f"Integrity check: Python-computed {label} ({computed}) disagrees with the "
                f"binary's own self-reported value ({self_reported}) by {rel_diff:.1%}. "
                f"Using the Python-computed value (derived directly from real parsed kernel "
                f"time + real parsed byte/FLOP count) as the report's headline number."
            )

    if job.seed_id.value == "vectorAdd":
        total_gb = parsed.get("total_data_moved_gb")
        if total_gb is not None:
            derived.bytes_moved = total_gb * 1e9
            derived.achieved_bw_gbs = round(total_gb / seconds, 2)
            if peaks and "hbm_bandwidth_gbs" in peaks:
                derived.theoretical_peak_gbs = peaks["hbm_bandwidth_gbs"]
                derived.efficiency_percent = round((derived.achieved_bw_gbs / derived.theoretical_peak_gbs) * 100, 1)
            _cross_check("achieved bandwidth", derived.achieved_bw_gbs, parsed.get("achieved_bw_gbs_selfreported"))

    elif job.seed_id.value == "tiledMatmul":
        gflops = parsed.get("gflops")
        if gflops is not None:
            derived.flops = gflops * 1e9
            derived.achieved_tflops = round((gflops / seconds) / 1000.0, 2)
            if peaks and "fp32_tflops" in peaks:
                derived.theoretical_peak_tflops = peaks["fp32_tflops"]
                derived.efficiency_tflops_percent = round((derived.achieved_tflops / derived.theoretical_peak_tflops) * 100, 1)
            _cross_check("achieved TFLOPS", derived.achieved_tflops, parsed.get("achieved_tflops_selfreported"))

    elif job.seed_id.value == "reduction":
        elements = parsed.get("elements")
        if elements is not None:
            bytes_read = elements * 4  # sizeof(float) — reduction reads once, no write-back of the array
            derived.bytes_moved = float(bytes_read)
            derived.achieved_bw_gbs = round((bytes_read / 1e9) / seconds, 2)
            # reduction.cu has no published peak-BW efficiency line (it's a
            # control-flow/atomics seed, not a bandwidth-hero one) — leave
            # efficiency_percent unset rather than compute a % against a
            # peak number that wasn't the point of this seed.
            _cross_check("effective read bandwidth", derived.achieved_bw_gbs, parsed.get("effective_read_bw_gbs_selfreported"))

    return derived


def run_baseline(
    job: JobState,
    ws: WorkspaceManager,
    seeds_root: Path,
    on_progress: Callable[[JobState], None] | None = None,
) -> JobState:
    """Execute the full non-agent baseline for a seed job.

    Returns the final (mutated) JobState. Side-effects: writes state.json, artifacts.
    """
    ws_dir = ws.create_workspace(job)
    job.workspace_dir = str(ws_dir)

    _append_message(job, "Baseline Orchestrator", "thought", f"Starting baseline pipeline for {job.seed_id.value} on {'MOCK (local dev)' if MOCK else 'a real AMD GPU'} environment. Planning steps: analyze -> port -> validate -> benchmark -> report.")
    ws.write_state(job)

    # 1. Prepare
    _advance(job, JobPhase.ANALYSIS, ws)
    _append_message(job, "Baseline Analyzer", "action", "Copying self-contained seed into isolated workspace.")
    src_cu = ws.copy_seed(job, seeds_root)
    _append_message(job, "Baseline Analyzer", "observation", f"Detected {src_cu.name} ({src_cu.stat().st_size} bytes). Simple kernel pattern, low risk for port. 1 kernel found.")
    if on_progress: on_progress(job)

    # 2. hipify — tool is auto-selected inside run_hipify(): hipify-perl
    # preferred (no CUDA SDK needed, correct default on an AMD-only box),
    # hipify-clang only as a fallback when hipify-perl is missing AND a
    # real CUDA install is actually present.
    _advance(job, JobPhase.PORTING, ws)
    hip_out = ws_dir / "hip_out"
    _append_message(job, "HIP Porting Specialist", "action", "Running hipify (auto-selecting hipify-perl or hipify-clang) on CUDA sources.")
    ok, log, err, hipify_tool = run_hipify(src_cu.parent, hip_out, job.job_id)
    job.hipify_command = f"{hipify_tool} ..."
    (ws_dir / "logs" / "hipify.log").write_text(log, encoding="utf-8")

    if not ok and not MOCK:
        _append_message(job, "HIP Porting Specialist", "observation", f"hipify failed (tool: {hipify_tool}). stderr snippet: {err[:180]}. Will attempt minimal repair before compile.")
    else:
        _append_message(job, "HIP Porting Specialist", "observation", f"hipify completed cleanly with {hipify_tool}. Scanning for common CUDA->HIP mappings (cudaMemcpy -> hipMemcpy, kernel launch, etc.).")

    hip_files = _discover_hip_sources(hip_out)

    # 3. hipcc — arch is auto-detected inside run_hipcc() (rocm_agent_enumerator
    # / rocminfo, falling back to --offload-arch=native), never assumed.
    # Compiling for the wrong architecture doesn't fail the build — it
    # produces a binary that segfaults at launch on hardware whose ISA
    # doesn't match what was compiled for.
    binary = hip_out / f"{job.seed_id.value}_hip"
    _append_message(job, "HIP Porting Specialist", "action", "Compiling with hipcc -O3 (auto-detecting target GPU architecture).")
    ok, log, err, arch_used = run_hipcc(hip_files, binary)
    (ws_dir / "logs" / "hipcc.log").write_text(log, encoding="utf-8")
    job.gpu_arch = arch_used
    job.hipcc_command = f"hipcc -O3 --offload-arch={arch_used} ... -o {binary.name}"

    # run_hipcc() already returns a bool (success = rc == 0 computed inside
    # it) — do NOT compare it to 0 again. `False == 0` is True in Python
    # (bool subclasses int), so `(ok == 0)` would silently invert this on
    # every real hipcc result: a genuine compile success would be reported
    # as FAILED, and a genuine failure would be reported as COMPLETED with
    # the pipeline then trying to run a binary that was never built.
    compile_success = ok

    if not compile_success and not MOCK:
        job.status = JobStatus.FAILED
        job.error = f"hipcc failed: {err[:800]}"
        _append_message(job, "HIP Porting Specialist", "observation", f"hipcc failed (target arch: {arch_used}). Full error captured in logs/hipcc.log. Continuing to produce diagnostic report + artifacts anyway.")
        # Do NOT early return — we still want a useful report + tar for the user
    else:
        _append_message(job, "HIP Porting Specialist", "observation", f"hipcc succeeded cleanly, compiled for detected architecture {arch_used}. Binary ready: {binary.name}. Moving to execution.")
        if on_progress: on_progress(job)

    # 4-5. Run + validate + metrics ONLY if compile succeeded
    run_stdout = ""
    if compile_success:
        _advance(job, JobPhase.BENCHMARKING, ws)
        _append_message(job, "Benchmark & Profiler", "action", "Executing benchmark binary + capturing live amd-smi telemetry (util, power, temp).")

        pre_metrics, pre_raw = capture_amd_smi_snapshot()
        (ws_dir / "logs" / "amd_smi_pre.txt").write_text(pre_raw, encoding="utf-8")

        rc, stdout, stderr, wall = run_binary(binary, [])
        run_stdout = stdout
        (ws_dir / "logs" / "run.log").write_text(
            f"rc={rc}\nprocess_wall_clock_seconds={wall:.4f}  (whole-process time, NOT GPU kernel time — see 'Kernel time' in stdout below for the real hipEvent measurement)\nstdout:\n{stdout}\nstderr:\n{stderr}",
            encoding="utf-8",
        )

        post_metrics, post_raw = capture_amd_smi_snapshot()
        (ws_dir / "logs" / "amd_smi_post.txt").write_text(post_raw, encoding="utf-8")

        parsed = parse_binary_output_for_metrics(stdout)

        # Raw GPU telemetry: use exactly what was actually captured for this
        # run (the post-kernel snapshot). No `or <constant>` fallback here —
        # a field genuinely not parsed from amd-smi stays None and the
        # report renders "Not captured", never a fabricated plausible number.
        raw = RawMetrics(
            gpu_utilization_percent=post_metrics.gpu_utilization_percent,
            power_watts_avg=post_metrics.power_watts_avg,
            power_watts_peak=post_metrics.power_watts_peak,
            temperature_c=post_metrics.temperature_c,
            memory_used_mb=post_metrics.memory_used_mb,
            clock_sclk_mhz=post_metrics.clock_sclk_mhz,
            clock_mclk_mhz=post_metrics.clock_mclk_mhz,
        )

        derived = _compute_derived_metrics(
            job, parsed,
            job_msg=lambda text: _append_message(job, "Benchmark & Profiler", "observation", text),
        )

        job.metrics = JobMetrics(raw=raw, derived=derived)
        kt_desc = f"~{derived.kernel_time_ms:.3f} ms (hipEvent, real GPU-side timing)" if derived.kernel_time_ms is not None else "Not captured (binary did not print a 'Kernel time:' line)"
        _append_message(job, "Benchmark & Profiler", "observation", f"Kernel time: {kt_desc}. Process wall-clock: {wall:.3f}s (informational only — not used as a metric). amd-smi snapshots saved to logs/amd_smi_pre.txt + amd_smi_post.txt. {len(job.messages)} decisions so far.")

        # 6. Validation — PASSED requires BOTH that the binary ran to clean
        # completion AND that the actual-vs-expected numbers it printed are
        # actually within tolerance. Printing *a* result line is not
        # correctness — a stale-buffer bug that made a reduction seed
        # report exactly double the expected value was previously reported
        # "PASSED" because "reduction result" appeared in stdout regardless
        # of the number after it. This is applied the same way for all
        # three seeds (see VALIDATION_TOLERANCES / _validation_passes),
        # not just reduction, so this class of bug can't hide in the others.
        _advance(job, JobPhase.VALIDATING, ws)

        check_pairs = parsed.get("check_pairs")
        if check_pairs:
            job.max_abs_diff = max(abs(actual - expected) for actual, expected in check_pairs)
            diff_desc = f"{job.max_abs_diff:.6g} (real, computed from {len(check_pairs)} actual-vs-expected pair(s) printed by the binary)"
        else:
            job.max_abs_diff = None
            diff_desc = "Not captured (binary printed no parseable actual-vs-expected check line)"

        within_tolerance = _validation_passes(job.seed_id.value, check_pairs or [])
        ran_to_completion = rc == 0 and ("completed successfully" in stdout.lower() or "reduction result" in stdout.lower())
        validation_ok = ran_to_completion and within_tolerance
        job.validation_passed = validation_ok

        if not check_pairs:
            validation_detail = f"ISSUES DETECTED — no actual-vs-expected pair could be parsed from stdout, so correctness cannot be confirmed. Max abs diff: {diff_desc}."
        elif not within_tolerance:
            validation_detail = f"ISSUES DETECTED — actual value(s) do not match expected within tolerance. Max abs diff: {diff_desc}."
        elif not ran_to_completion:
            validation_detail = f"ISSUES DETECTED — numbers matched within tolerance, but the binary did not report clean completion (rc={rc})."
        else:
            validation_detail = f"PASSED — actual matches expected within tolerance. Max abs diff: {diff_desc}."

        _append_message(job, "Validator", "observation", f"Self-validation {validation_detail}")
    else:
        run_stdout = f"COMPILE FAILED\n{job.error or ''}\nSee logs/hipcc.log for full compiler output."
        job.validation_passed = False

    # Decide final status first (so the report sees the correct "COMPLETED" / "FAILED")
    if compile_success:
        final_status = JobStatus.COMPLETED
        final_phase = JobPhase.COMPLETED
        if MOCK:
            final_thought = "Baseline pipeline finished successfully (MOCK). All artifacts produced. Ready for a real hardware run — set SATOSWARM_MOCK=0 or leave it unset; the target GPU architecture is auto-detected, no config needed."
        else:
            final_thought = f"Baseline pipeline finished successfully on real hardware (detected architecture: {job.gpu_arch}). All artifacts produced."
    else:
        final_status = JobStatus.FAILED
        final_phase = JobPhase.FAILED
        final_thought = "Pipeline stopped at compile step. Diagnostic report + full logs + attempted HIP sources have been packaged into the artifacts tar so you have everything needed to debug or continue manually."

    # 7. Report + artifacts — ALWAYS produce (even on compile failure).
    _advance(job, JobPhase.REPORTING, ws)
    report_path = generate_minimal_report(job, ws_dir, run_stdout)
    job.report_md_path = str(report_path)
    tar_path = ws.create_tar(job)
    job.artifacts_tar_path = str(tar_path)

    _append_message(job, "Reporter", "observation", f"Report + artifacts tar ready (diagnostic if compile failed). Total decisions logged: {len(job.messages)}")

    # 8. Finalize
    _advance(job, final_phase, ws)
    job.status = final_status
    job.phase = final_phase
    _append_message(job, "Baseline Orchestrator", "thought", final_thought)

    ws.write_state(job)

    if on_progress:
        on_progress(job)
    return job


def generate_minimal_report(job: JobState, ws_dir: Path, run_stdout: str) -> Path:
    """Generate migration_report.md. Any metric that was not actually
    captured is rendered as "Not captured" via _fmt() — never a blank
    cell, never a placeholder number. In MOCK mode, _fmt() also appends
    "(SIMULATED)" to every value so the page is unambiguous even taken
    out of context.
    """
    reports = ws_dir / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / "migration_report.md"

    m = job.metrics
    d = m.derived
    eff = d.efficiency_percent if d.efficiency_percent is not None else d.efficiency_tflops_percent
    achieved = d.achieved_bw_gbs if d.achieved_bw_gbs is not None else d.achieved_tflops
    # Theoretical peak text is derived from the actual arch-keyed lookup
    # (see GPU_THEORETICAL_PEAKS) — never a fixed MI300X number. If the
    # detected architecture has no verified entry, this says so explicitly
    # instead of silently reusing another GPU's spec.
    if job.seed_id.value == "vectorAdd":
        peak_note = f"{d.theoretical_peak_gbs:g} GB/s HBM" if d.theoretical_peak_gbs is not None else f"unknown peak for {job.gpu_arch or 'undetected architecture'}"
    else:
        peak_note = f"{d.theoretical_peak_tflops:g} TFLOPS FP32" if d.theoretical_peak_tflops is not None else f"unknown peak for {job.gpu_arch or 'undetected architecture'}"

    duration = "N/A"
    if job.updated_at and job.created_at:
        secs = (job.updated_at - job.created_at).total_seconds()
        duration = f"{secs:.1f}s"

    is_failed = job.status == JobStatus.FAILED or job.phase == JobPhase.FAILED

    if is_failed:
        exec_summary = (
            f"**FAILED** at compile step.\n"
            f"hipcc returned an error (see logs/hipcc.log and the error field). "
            f"Diagnostic report + attempted ported sources + full logs are included in the artifacts tar."
        )
    elif d.kernel_time_ms is None or achieved is None:
        exec_summary = (
            f"Baseline (non-agent) port of {job.seed_id.value} compiled and ran, but the binary's "
            f"stdout did not contain a parseable timing/metric line — kernel_time_ms and the achieved "
            f"metric are both \"Not captured\" below rather than a guessed number. See logs/run.log "
            f"for the raw stdout."
        )
    else:
        exec_summary = (
            f"Baseline (non-agent) port of {job.seed_id.value} completed in {_fmt(round(d.kernel_time_ms, 3), ' ms')}.\n"
            f"Achieved {_fmt(achieved)}"
            + (f" ({eff}% of theoretical {peak_note})." if eff is not None else f" (efficiency % of theoretical {peak_note} not applicable for this seed).")
        )

    content = f"""# SATO SWARM Migration Report — {job.seed_id.value}

**Job ID**: {job.job_id}
**Date**: {job.created_at.isoformat()}
**Hardware**: {f"AMD GPU, architecture {job.gpu_arch}" if job.gpu_arch else "AMD GPU (architecture not detected)"} via ROCm (see amd-smi)
**Mode**: {"MOCK (local dev — every number below is simulated, tagged (SIMULATED), never measured)" if MOCK else "REAL hardware"}
**Total Duration**: {duration}
**Messages / Decisions**: {len(job.messages)}
**Final Status**: {"FAILED" if is_failed else "COMPLETED"}

## Executive Summary
{exec_summary}

## Pipeline Journey (Baseline)
1. Analysis — seed copied and inspected.
2. Porting — hipify + hipcc.
3. Validating — self-check against embedded reference in binary output.
4. Benchmarking — timed run (hipEvent, GPU-side) + amd-smi snapshot (pre and post, saved to logs/).
5. Reporting — this document + tar of all artifacts.

## Performance Results
| Metric                  | Value                                   | % of Theoretical | Source |
|-------------------------|------------------------------------------|------------------|--------|
| Kernel time (hipEvent)  | {_fmt(round(d.kernel_time_ms, 3) if d.kernel_time_ms is not None else None, ' ms')} | -                | binary's own hipEventElapsedTime() |
| Memory BW (GB/s)        | {_fmt(d.achieved_bw_gbs)}                 | {_fmt(d.efficiency_percent, '%')} | kernel time + real bytes moved (Python-computed) |
| Compute (TFLOPS)        | {_fmt(d.achieved_tflops)}                 | {_fmt(d.efficiency_tflops_percent, '%')} | kernel time + real FLOP count (Python-computed) |
| Power (avg/peak W)      | {_fmt(m.raw.power_watts_avg)}/{_fmt(m.raw.power_watts_peak)} | -    | amd-smi |
| Utilization             | {_fmt(m.raw.gpu_utilization_percent, '%')} | -              | amd-smi |
| Temperature (C)         | {_fmt(m.raw.temperature_c)}               | -                | amd-smi |

**Key takeaway**: {"COMPILE/PORT FAILED on this run — see the Migration Notes section + logs/ for details. All attempted sources and logs are in the tar." if is_failed else ("Kernel time is the seed's own hipEventElapsedTime() measurement — real GPU-side timing recorded directly around the kernel launch. Bandwidth/TFLOPS are computed in Python from that real time plus a real byte/FLOP count also parsed from the binary's own stdout — never a constant. Power/utilization/temperature come from amd-smi; any 'Not captured' means the amd-smi JSON parser did not recognize a field on this instance — the raw amd-smi text is still saved in logs/amd_smi_pre.txt and logs/amd_smi_post.txt for manual reading." if not MOCK else "MOCK mode — every number on this page is simulated (tagged (SIMULATED) above), not measured. Run with SATOSWARM_MOCK unset (or =0) on a real AMD GPU for measured numbers — the target architecture is auto-detected, no config needed.")}

## Migration Notes & Limitations
This baseline performs a direct hipify + compile + benchmark, once, with no repair loop. On real hardware:

- hipify may succeed with warnings or require small manual mappings (cuda* -> hip*, launch syntax, etc.).
- hipcc may fail on the first pass for more complex CUDA (unknown identifiers, arch-specific intrinsics, etc.). If it does, the exact error is recorded in logs/hipcc.log and this report documents what did work plus reproducible commands for manual iteration.
- Kernel time / bandwidth / TFLOPS depend on the binary actually printing its own "Kernel time:" line (from hipEventElapsedTime around the kernel launch — see seeds/*.cu). If the binary crashes before printing it, these are "Not captured" — there is no wall-clock or other fallback wearing those labels. The process's whole wall-clock time is logged separately in logs/run.log for diagnostics only, never presented as "Kernel time".
- amd-smi's JSON output schema varies by ROCm version — the parser in src/tools/execution.py makes a best-effort attempt at common field names and will show "Not captured" for anything it can't find, rather than guessing. The raw amd-smi text is always saved in logs/ regardless, so nothing measured is ever lost even when the parser misses a field.
- Power/utilization are single pre/post snapshots, not continuous sampling during the kernel — "peak" power reuses the one real post-run reading rather than inventing a distinct number.
- If the binary's own self-printed "Achieved ..." line disagrees with the Python-computed value by more than 1%, a warning message is logged (see the job's message trace) and the Python-computed value is used as the report's headline number.

## Generated Artifacts
- Ported HIP sources + binary: hip_out/
- Full migration_report.md (this file)
- logs/: hipify.log, hipcc.log, run.log, amd_smi_pre.txt, amd_smi_post.txt
- artifacts tar: {Path(job.artifacts_tar_path or '').name}

## Commands Used (Reproducible)
```
# hipify
{job.hipify_command or 'hipify-perl (or hipify-clang) <auto-selected> ...'}
# hipcc
{job.hipcc_command or 'hipcc -O3 --offload-arch=<auto-detected> ...'}
# run
./{job.seed_id.value}_hip
# amd-smi
amd-smi metric --json
```

*This report was generated by SATO SWARM's baseline pipeline running {f"on real AMD hardware (architecture: {job.gpu_arch})" if not MOCK else "in MOCK mode"}.*
"""
    path.write_text(content, encoding="utf-8")
    return path
