"use client";

import { useEffect, useState } from "react";
import { createJob, getHealth, getJobReport, replaySeed } from "@/lib/api";
import { SEED_IDS, type HealthResponse, type SeedId } from "@/lib/types";
import LiveJobView from "@/components/LiveJobView";
import ReportViewer from "@/components/ReportViewer";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [reportView, setReportView] = useState<{ jobId: string; markdown: string } | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [launching, setLaunching] = useState<SeedId | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e instanceof Error ? e.message : "failed to reach backend"));
  }, []);

  async function handleRun(seedId: SeedId) {
    setReplayError(null);
    setReportView(null);
    setLaunching(seedId);
    try {
      const { job_id } = await createJob(seedId);
      setActiveJobId(job_id);
    } catch (e) {
      setReplayError(e instanceof Error ? e.message : "failed to start job");
    } finally {
      setLaunching(null);
    }
  }

  async function handleReplay(seedId: SeedId) {
    setReplayError(null);
    setReportView(null);
    try {
      const { job } = await replaySeed(seedId);
      // /demo/replay is the entry point (existence check + data source for
      // the historical job), but once we know the real job_id, the exact
      // same live-view machinery (GET /jobs/{id}/status, .../report) works
      // unchanged for it -- state.json for these pulled-from-pod jobs is
      // just as real and just as readable as any freshly-created job's.
      setActiveJobId(job.job_id);
    } catch (e) {
      setReplayError(e instanceof Error ? e.message : `no real completed run available for ${seedId}`);
    }
  }

  async function handleViewReport() {
    if (!activeJobId) return;
    try {
      const markdown = await getJobReport(activeJobId);
      setReportView({ jobId: activeJobId, markdown });
    } catch (e) {
      setReplayError(e instanceof Error ? e.message : "report not ready yet");
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-8 space-y-8">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">SATO SWARM</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          CUDA → HIP baseline pipeline + repair loop
        </p>
        {health && (
          // This line is specifically about the SERVER's current mode
          // (what a NEW run would use) -- NOT about whatever job is
          // being viewed below, which shows its own recorded mode
          // instead (see LiveJobView's mode badge / MetricsDashboard's
          // footer, both sourced from job.mode).
          <p data-testid="health-summary" className="text-xs text-gray-500 dark:text-gray-400">
            Server: {health.mode} mode ({health.mode === "MOCK" ? "new runs simulated" : "new runs use real hardware"}) ·{" "}
            GPU: {health.gpu_arch ?? "not detected"} · {health.memory_patterns} memory patterns ·{" "}
            {health.tool_registry_tools} tools · {health.jobs_run_this_session} jobs this session
          </p>
        )}
        {healthError && (
          <p className="text-xs text-red-600 dark:text-red-400">
            Backend unreachable: {healthError}. Is `uvicorn src.main:app` running?
          </p>
        )}
      </header>

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Seeds</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {SEED_IDS.map((seed) => (
            <div
              key={seed}
              className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-2"
            >
              <p className="font-medium text-gray-900 dark:text-gray-50">{seed}</p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => handleRun(seed)}
                  disabled={launching === seed}
                  data-testid={`run-${seed}`}
                  className="rounded-md bg-slate-800 dark:bg-slate-200 dark:text-slate-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
                >
                  {launching === seed ? "Starting…" : "Run"}
                </button>
                <button
                  type="button"
                  onClick={() => handleReplay(seed)}
                  data-testid={`replay-${seed}`}
                  className="rounded-md border border-gray-300 dark:border-gray-600 px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300"
                >
                  Replay real run
                </button>
              </div>
            </div>
          ))}
        </div>
        {replayError && (
          <p data-testid="replay-error" className="text-sm text-red-600 dark:text-red-400">
            {replayError}
          </p>
        )}
      </section>

      {activeJobId && !reportView && health && (
        <section className="border-t border-gray-200 dark:border-gray-700 pt-6">
          <LiveJobView jobId={activeJobId} onViewReport={handleViewReport} />
        </section>
      )}

      {reportView && (
        <section className="border-t border-gray-200 dark:border-gray-700 pt-6">
          <ReportViewer
            jobId={reportView.jobId}
            reportMarkdown={reportView.markdown}
            onClose={() => setReportView(null)}
          />
        </section>
      )}
    </main>
  );
}
