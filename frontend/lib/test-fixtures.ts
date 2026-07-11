import type { JobState } from "./types";

// A representative, shape-accurate mock JobState -- every field present
// and typed exactly as src/models/job.py's JobState really serializes it
// (see lib/types.ts). Used only by component render tests; never
// imported by app code.
export function makeMockJob(overrides: Partial<JobState> = {}): JobState {
  return {
    job_id: "job_test0000001",
    seed_id: "vectorAdd",
    phase: "Benchmarking",
    status: "running",
    mode: "MOCK",
    created_at: "2026-07-11T01:00:00Z",
    updated_at: "2026-07-11T01:00:05Z",
    messages: [
      { id: 1, agent: "Baseline Orchestrator", timestamp: "01:00:00", type: "thought", content: "Starting baseline pipeline for vectorAdd." },
      { id: 2, agent: "Baseline Analyzer", timestamp: "01:00:01", type: "action", content: "Copying self-contained seed into isolated workspace." },
      { id: 3, agent: "HIP Porting Specialist", timestamp: "01:00:02", type: "action", content: "Compiling with hipcc -O3." },
    ],
    completed_phases: ["Analysis", "Porting"],
    metrics: {
      raw: {
        gpu_utilization_percent: 92.0,
        power_watts_avg: 680.0,
        power_watts_peak: 710.0,
        temperature_c: 67.0,
        memory_used_mb: 42000.0,
        clock_sclk_mhz: 2100.0,
        clock_mclk_mhz: 5200.0,
      },
      derived: {
        achieved_bw_gbs: 4601.23,
        theoretical_peak_gbs: 5300.0,
        efficiency_percent: 86.8,
        achieved_tflops: null,
        theoretical_peak_tflops: null,
        efficiency_tflops_percent: null,
        theoretical_peak_source: "mock",
        theoretical_peak_calculation: "5300 GB/s — MOCK mode, fixed calibration value, no hardware queried",
        kernel_time_ms: 0.652,
        bytes_moved: 3000000000,
        flops: null,
      },
      timeseries: [],
      captured_at: "2026-07-11T01:00:05Z",
    },
    report_md_path: null,
    artifacts_tar_path: null,
    hip_out_dir: null,
    hipify_command: "hipify-perl ...",
    hipcc_command: "hipcc -O3 --offload-arch=gfx942 ... -o vectorAdd_hip",
    gpu_arch: "gfx942",
    repair_loops: 0,
    validation_passed: null,
    max_abs_diff: null,
    tolerance: 1e-5,
    error: null,
    workspace_dir: "/repo/jobs/job_test0000001",
    ...overrides,
  };
}

export const MOCK_REPORT_MD = `# SATO SWARM Migration Report — vectorAdd

**Job ID**: job_test0000001
**Date**: 2026-07-11T01:00:05
**Final Status**: COMPLETED

## Executive Summary
Baseline (non-agent) port of vectorAdd completed in 0.652 ms.
Achieved 4601.23 (86.8% of theoretical 5300 GB/s).
`;

export const MOCK_FAILED_REPORT_MD = `# SATO SWARM Migration Report — repairDemo

**Job ID**: job_test0000002
**Final Status**: FAILED

## Executive Summary
**FAILED** at compile step.
`;
