"use client";

import ReactMarkdown from "react-markdown";
import { getJobArtifactsUrl } from "@/lib/api";

interface ReportViewerProps {
  jobId: string;
  reportMarkdown: string;
  onClose?: () => void;
}

// The report's own "**Final Status**: FAILED" / "**Final Status**:
// COMPLETED" line (see src/baseline/pipeline.py's generate_minimal_report())
// is the source of truth here -- deliberately NOT a separately-passed
// job.status prop, so this component tells the truth even if it's only
// ever given the raw markdown text (e.g. a replayed historical report)
// with no live JobState alongside it.
const FINAL_STATUS_RE = /\*\*Final Status\*\*:\s*(FAILED|COMPLETED)/;

export default function ReportViewer({ jobId, reportMarkdown, onClose }: ReportViewerProps) {
  const match = reportMarkdown.match(FINAL_STATUS_RE);
  const isFailed = match?.[1] === "FAILED";

  return (
    <div className="space-y-4">
      <div
        data-testid="report-header"
        data-failed={isFailed}
        className={`flex items-center justify-between rounded-md px-4 py-3 ${
          isFailed
            ? "bg-red-50 dark:bg-red-950 text-red-900 dark:text-red-100"
            : "bg-emerald-50 dark:bg-emerald-950 text-emerald-900 dark:text-emerald-100"
        }`}
      >
        <h2 className="text-sm font-semibold">
          Migration report — {isFailed ? "FAILED" : "COMPLETED"}
        </h2>
        <div className="flex gap-2">
          <a
            href={getJobArtifactsUrl(jobId)}
            data-testid="report-download-artifacts"
            className="rounded-md border border-current px-3 py-1 text-xs font-medium hover:opacity-75"
          >
            Download artifacts
          </a>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-current px-3 py-1 text-xs font-medium hover:opacity-75"
            >
              Close
            </button>
          )}
        </div>
      </div>

      <div
        data-testid="report-markdown"
        className="prose prose-sm dark:prose-invert max-w-none rounded-md border border-gray-200 dark:border-gray-700 p-4"
      >
        <ReactMarkdown>{reportMarkdown}</ReactMarkdown>
      </div>
    </div>
  );
}
