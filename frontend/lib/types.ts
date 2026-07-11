// Mirrors src/models/job.py exactly. This is a routing/rendering layer
// only — every field here corresponds 1:1 to a real Pydantic field the
// backend actually serializes; nothing is invented on the frontend side.
// If a field is optional/nullable in Python, it's optional/nullable here
// too, and components must render "Not captured"/"Not applicable" for a
// null value themselves — never substitute a fabricated number.

export type JobPhase =
  | "Queued"
  | "Analysis"
  | "Porting"
  | "Validating"
  | "Benchmarking"
  | "Optimizing"
  | "Reporting"
  | "Completed"
  | "Failed";

export type JobStatus = "running" | "completed" | "failed";

export type SeedId = "vectorAdd" | "tiledMatmul" | "reduction" | "repairDemo";

export const SEED_IDS: SeedId[] = ["vectorAdd", "tiledMatmul", "reduction", "repairDemo"];

export type MessageType = "thought" | "action" | "observation";

export interface AgentMessage {
  id: number;
  agent: string;
  timestamp: string;
  type: MessageType;
  content: string;
}

export interface RawMetrics {
  gpu_utilization_percent: number | null;
  power_watts_avg: number | null;
  power_watts_peak: number | null;
  temperature_c: number | null;
  memory_used_mb: number | null;
  clock_sclk_mhz: number | null;
  clock_mclk_mhz: number | null;
}

export interface DerivedMetrics {
  achieved_bw_gbs: number | null;
  theoretical_peak_gbs: number | null;
  efficiency_percent: number | null;

  achieved_tflops: number | null;
  theoretical_peak_tflops: number | null;
  efficiency_tflops_percent: number | null;

  theoretical_peak_source: string | null;
  theoretical_peak_calculation: string | null;

  kernel_time_ms: number | null;
  bytes_moved: number | null;
  flops: number | null;
}

export interface JobMetrics {
  raw: RawMetrics;
  derived: DerivedMetrics;
  timeseries: Record<string, unknown>[];
  captured_at: string;
}

export interface JobState {
  job_id: string;
  seed_id: SeedId;
  phase: JobPhase;
  status: JobStatus;
  // THIS job's own recorded mode — set once when it ran, independent of
  // whatever the server is currently running as (e.g. a replayed real
  // job stays "REAL" even when viewed from a server running in MOCK
  // mode). See src/models/job.py's JobState.mode docstring.
  mode: "MOCK" | "REAL";
  created_at: string;
  updated_at: string;

  messages: AgentMessage[];
  completed_phases: JobPhase[];

  metrics: JobMetrics;
  report_md_path: string | null;
  artifacts_tar_path: string | null;
  hip_out_dir: string | null;

  hipify_command: string | null;
  hipcc_command: string | null;
  gpu_arch: string | null;
  repair_loops: number;

  validation_passed: boolean | null;
  max_abs_diff: number | null;
  tolerance: number;

  error: string | null;
  workspace_dir: string | null;
}

// Matches src/main.py's HealthResponse exactly (a route-local model, not
// job.py's differently-shaped HealthResponse).
export interface HealthResponse {
  system: string;
  mode: "MOCK" | "REAL";
  gpu_arch: string | null;
  memory_patterns: number;
  jobs_run_this_session: number;
  tool_registry_tools: number;
}

export interface JobCreateResponse {
  job_id: string;
  status: string;
}

export interface ReplayResponse {
  job: JobState;
  report_md: string | null;
}
