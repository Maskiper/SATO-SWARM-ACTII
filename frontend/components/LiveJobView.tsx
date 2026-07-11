"use client";

import { useEffect, useState } from "react";
import { getJobArtifactsUrl, getJobStatus } from "@/lib/api";
import type { JobState } from "@/lib/types";
import AgentFeed from "./AgentFeed";
import MetricsDashboard from "./MetricsDashboard";
import PhaseTimeline from "./PhaseTimeline";

interface LiveJobViewProps {
  jobId: string;
  onViewReport?: () => void;
}

// job.mode is THIS job's own recorded mode (see JobState.mode's
// docstring) -- deliberately independent of whatever the server is
// currently running as, so a replayed real job still shows "REAL DATA"
// even when viewed from a server currently running in MOCK mode.
const MODE_BADGE: Record<JobState["mode"], { label: string; classes: string }> = {
  REAL: {
    label: "REAL DATA",
    classes: "bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  },
  MOCK: {
    label: "SIMULATED",
    classes: "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200",
  },
};

const STATUS_BANNER: Record<JobState["status"], { label: string; classes: string }> = {
  running: {
    label: "Running",
    classes: "bg-blue-50 text-blue-800 dark:bg-blue-950 dark:text-blue-200",
  },
  completed: {
    label: "Completed",
    classes: "bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  },
  failed: {
    label: "Failed",
    classes: "bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200",
  },
};

export default function LiveJobView({ jobId, onViewReport }: LiveJobViewProps) {
  const [job, setJob] = useState<JobState | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const j = await getJobStatus(jobId);
        if (!cancelled) {
          setJob(j);
          setPollError(null);
        }
        return j;
      } catch (e) {
        if (!cancelled) setPollError(e instanceof Error ? e.message : "failed to load job status");
        return null;
      }
    };

    let interval: ReturnType<typeof setInterval> | null = null;
    poll().then((j) => {
      if (cancelled) return;
      if (j && (j.status === "completed" || j.status === "failed")) return;
      interval = setInterval(async () => {
        const latest = await poll();
        if (latest && (latest.status === "completed" || latest.status === "failed") && interval) {
          clearInterval(interval);
        }
      }, 1000);
    });

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [jobId]);

  if (pollError && !job) {
    return (
      <div className="rounded-md bg-red-50 dark:bg-red-950 p-4 text-red-800 dark:text-red-200 text-sm">
        Couldn&apos;t load job {jobId}: {pollError}
      </div>
    );
  }

  if (!job) {
    return <div className="text-sm text-gray-500 dark:text-gray-400">Loading job {jobId}…</div>;
  }

  const banner = STATUS_BANNER[job.status];
  const reportReady = job.report_md_path !== null;
  const artifactsReady = job.artifacts_tar_path !== null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50">
            {job.seed_id} — {job.job_id}
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            GPU arch: {job.gpu_arch ?? "not yet detected"}
            {job.repair_loops > 0 && ` · repair loops used: ${job.repair_loops}`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            data-testid="status-banner"
            data-status={job.status}
            className={`rounded-full px-3 py-1 text-sm font-medium ${banner.classes}`}
          >
            {banner.label}
          </span>
          <span
            data-testid="mode-badge"
            data-job-mode={job.mode}
            className={`rounded-full px-3 py-1 text-sm font-medium ${MODE_BADGE[job.mode].classes}`}
          >
            {MODE_BADGE[job.mode].label}
          </span>
        </div>
      </div>

      {job.status === "failed" && job.error && (
        <div
          data-testid="job-error"
          className="rounded-md border border-red-300 bg-red-50 dark:bg-red-950 dark:border-red-800 p-3 text-sm text-red-800 dark:text-red-200"
        >
          <strong>Error:</strong> {job.error}
        </div>
      )}

      <PhaseTimeline job={job} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-80 lg:h-96">
          <AgentFeed key={jobId} jobId={jobId} />
        </div>
        <div>
          <MetricsDashboard job={job} />
        </div>
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          disabled={!reportReady}
          onClick={onViewReport}
          data-testid="view-report-button"
          className="rounded-md bg-slate-800 dark:bg-slate-200 dark:text-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 disabled:cursor-not-allowed"
        >
          View report
        </button>
        <a
          href={artifactsReady ? getJobArtifactsUrl(jobId) : undefined}
          data-testid="download-artifacts-button"
          aria-disabled={!artifactsReady}
          className={`rounded-md border border-slate-300 dark:border-slate-600 px-4 py-2 text-sm font-medium text-slate-800 dark:text-slate-200 ${
            artifactsReady ? "hover:bg-slate-50 dark:hover:bg-slate-800" : "opacity-40 pointer-events-none"
          }`}
        >
          Download artifacts
        </a>
      </div>
    </div>
  );
}
