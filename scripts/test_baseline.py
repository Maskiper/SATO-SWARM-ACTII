#!/usr/bin/env python3
"""
SATO SWARM — Baseline Pipeline Test Runner

Usage:
    # Local development (mock — no AMD hardware required)
    SATOSWARM_MOCK=1 python scripts/test_baseline.py vectorAdd

    # On a real AMD GPU instance (after preflight verification)
    python scripts/test_baseline.py vectorAdd
    # (SATOSWARM_MOCK unset, or explicitly =0, means REAL mode. The target
    #  GPU architecture is auto-detected at compile time — gfx942, gfx1100,
    #  whatever's actually present.)

This runs the complete non-agent baseline end-to-end:
- Creates isolated workspace
- Copies seed
- hipify -> hipcc (or mock)
- Executes + captures metrics
- Validates
- Generates migration_report.md + artifacts tar
- Prints the report and key numbers
"""

import sys
from pathlib import Path

# Ensure we can import src when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.job import JobState, SeedId
from src.workspace.manager import WorkspaceManager
from src.baseline.pipeline import run_baseline
from src.tools.execution import MOCK  # single source of truth for mock/real — see execution.py


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_baseline.py <seed_id>")
        print("Available seeds: vectorAdd, tiledMatmul, reduction")
        sys.exit(1)

    seed_arg = sys.argv[1]
    try:
        seed_id = SeedId(seed_arg)
    except ValueError:
        print(f"Invalid seed: {seed_arg}")
        print("Valid: vectorAdd, tiledMatmul, reduction")
        sys.exit(1)

    print("=" * 72)
    print("SATO SWARM Baseline Test")
    print(f"Seed: {seed_id.value}")
    print(f"Mode: {'MOCK (local dev)' if MOCK else 'REAL (hardware, arch auto-detected)'}")
    print("=" * 72)

    job = JobState(seed_id=seed_id)
    ws = WorkspaceManager(base_dir=Path("./jobs"))

    seeds_root = Path(__file__).resolve().parents[1] / "seeds"

    print("\n[1/3] Running full baseline pipeline...")
    final_job = run_baseline(job, ws, seeds_root)

    print("\n[2/3] Pipeline finished.")
    print(f"  Status: {final_job.status}")
    print(f"  Final phase: {final_job.phase}")
    print(f"  GPU architecture (auto-detected): {final_job.gpu_arch or 'not detected'}")
    print(f"  Messages logged: {len(final_job.messages)}")
    print(f"  Workspace: {final_job.workspace_dir}")

    # Show last few messages
    print("\n--- Last activity ---")
    for msg in final_job.messages[-5:]:
        print(f"  [{msg.timestamp}] {msg.agent}: {msg.content[:80]}")

    # Show derived metrics
    d = final_job.metrics.derived
    print("\n--- Key Metrics ---")
    if d.achieved_bw_gbs:
        if d.efficiency_percent is not None:
            eff = f"{d.efficiency_percent}% of {d.theoretical_peak_gbs:g} GB/s"
        else:
            eff = "not computed (no verified theoretical peak for this GPU architecture)"
        print(f"  Achieved BW: {d.achieved_bw_gbs:.2f} GB/s  |  Efficiency: {eff}")
    if d.achieved_tflops:
        eff = f"{d.efficiency_tflops_percent}%" if d.efficiency_tflops_percent is not None else "not computed"
        print(f"  Achieved TFLOPS: {d.achieved_tflops:.2f}  |  Efficiency: {eff}")
    kt = f"{d.kernel_time_ms:.3f} ms (hipEvent)" if d.kernel_time_ms is not None else "Not captured"
    print(f"  Kernel time: {kt}")
    print(f"  Validation passed: {final_job.validation_passed}")

    raw = final_job.metrics.raw
    print("\n--- GPU Telemetry (amd-smi) ---")
    print(f"  Utilization: {raw.gpu_utilization_percent if raw.gpu_utilization_percent is not None else 'Not captured'}")
    print(f"  Power (avg/peak W): {raw.power_watts_avg if raw.power_watts_avg is not None else 'Not captured'}/{raw.power_watts_peak if raw.power_watts_peak is not None else 'Not captured'}")
    print(f"  Temperature (C): {raw.temperature_c if raw.temperature_c is not None else 'Not captured'}")

    # Print the report
    print("\n[3/3] Generated Migration Report:")
    if final_job.report_md_path and Path(final_job.report_md_path).exists():
        report = Path(final_job.report_md_path).read_text(encoding="utf-8")
        print("-" * 72)
        print(report)
        print("-" * 72)
    else:
        print("  (Report path not set)")

    if final_job.artifacts_tar_path:
        print(f"\nArtifacts tar ready: {final_job.artifacts_tar_path}")

    print("\nBaseline test complete.")
    if MOCK:
        print("   This was a MOCK run. For real hardware: unset SATOSWARM_MOCK (or set it to 0)")
        print("   on a machine with hipify-clang, hipcc, and amd-smi on PATH. The target GPU")
        print("   architecture is auto-detected - no config needed for gfx942, gfx1100, etc.")
    print("=" * 72)


if __name__ == "__main__":
    main()
