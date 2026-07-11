import type { JobState } from "@/lib/types";

function fmt(value: number | null | undefined, suffix = "", digits = 2): string {
  if (value === null || value === undefined) return "Not captured";
  return `${value.toFixed(digits)}${suffix}`;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
      <dt className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</dt>
      <dd className="mt-1 text-lg font-semibold text-gray-900 dark:text-gray-50">{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{hint}</p>}
    </div>
  );
}

interface MetricsDashboardProps {
  job: JobState;
}

export default function MetricsDashboard({ job }: MetricsDashboardProps) {
  const d = job.metrics.derived;
  const raw = job.metrics.raw;

  const achieved = d.achieved_bw_gbs ?? d.achieved_tflops;
  const achievedUnit = d.achieved_bw_gbs !== null ? "GB/s" : d.achieved_tflops !== null ? "TFLOPS" : "";
  const efficiency = d.efficiency_percent ?? d.efficiency_tflops_percent;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Metrics</h3>

      <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <Stat label="Kernel time" value={fmt(d.kernel_time_ms, " ms", 3)} />
        <Stat
          label="Achieved bandwidth/TFLOPS"
          value={achieved !== null ? `${achieved.toFixed(2)} ${achievedUnit}` : "Not captured"}
        />
        <Stat
          label="Efficiency"
          value={efficiency !== null ? `${efficiency.toFixed(1)}%` : "Not applicable"}
          hint={
            efficiency !== null
              ? d.theoretical_peak_source
                ? `source: ${d.theoretical_peak_source}`
                : undefined
              : "no theoretical peak was computed for this GPU"
          }
        />
        <Stat label="GPU utilization" value={fmt(raw.gpu_utilization_percent, "%", 1)} />
        <Stat
          label="Power (avg / peak)"
          value={`${fmt(raw.power_watts_avg, "", 0)} / ${fmt(raw.power_watts_peak, "", 0)} W`}
        />
        <Stat label="Temperature" value={fmt(raw.temperature_c, " °C", 0)} />
      </dl>

      {d.theoretical_peak_calculation && (
        <p className="text-xs text-gray-500 dark:text-gray-400 break-words">
          {d.theoretical_peak_calculation}
        </p>
      )}

      {/* Sourced from job.mode -- THIS job's own recorded mode, not the
          server's current mode (see JobState.mode's docstring) -- so a
          replayed real job still correctly says "real hardware" even
          when viewed from a server currently running in MOCK mode.
          Matches migration_report.md's own (SIMULATED) tagging convention
          (see src/baseline/pipeline.py's _fmt()) -- mock data never claims
          real hardware, real data never adds an unearned simulated caveat. */}
      <div
        data-testid="metrics-mode-footer"
        data-job-mode={job.mode}
        className={`rounded-md px-3 py-2 text-xs font-medium ${
          job.mode === "REAL"
            ? "bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
            : "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
        }`}
      >
        {job.mode === "REAL"
          ? "This job ran on real hardware — no simulation"
          : "This job's data is simulated (MOCK mode) — not measured"}
      </div>
    </div>
  );
}
