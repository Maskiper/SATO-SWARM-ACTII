import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LiveJobView from "@/components/LiveJobView";
import { makeMockJob } from "@/lib/test-fixtures";

const mockGetJobStatus = vi.fn();
const mockStreamJob = vi.fn();
const mockGetJobArtifactsUrl = vi.fn((jobId: string) => `http://127.0.0.1:8000/jobs/${jobId}/artifacts`);

vi.mock("@/lib/api", () => ({
  getJobStatus: (...args: unknown[]) => mockGetJobStatus(...args),
  streamJob: (...args: unknown[]) => mockStreamJob(...args),
  getJobArtifactsUrl: (...args: [string]) => mockGetJobArtifactsUrl(...args),
}));

describe("LiveJobView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStreamJob.mockImplementation(() => () => {});
  });

  it("renders the real job's status banner, phases, and metrics once loaded", async () => {
    mockGetJobStatus.mockResolvedValue(makeMockJob({ status: "completed", phase: "Reporting", completed_phases: ["Analysis", "Porting", "Validating", "Benchmarking", "Reporting"] }));

    render(<LiveJobView jobId="job_test0000001" />);

    await waitFor(() => {
      expect(screen.getByTestId("status-banner")).toHaveTextContent("Completed");
    });
    expect(screen.getByText(/vectorAdd — job_test0000001/)).toBeInTheDocument();
  });

  it("shows the job's REAL error text for a failed job, not a generic message", async () => {
    mockGetJobStatus.mockResolvedValue(
      makeMockJob({
        status: "failed",
        phase: "Porting",
        error: "hipcc failed: use of undeclared identifier 'cudaCtxResetPersistingL2Cache'",
      }),
    );

    render(<LiveJobView jobId="job_test0000002" />);

    await waitFor(() => {
      expect(screen.getByTestId("job-error")).toHaveTextContent(
        "cudaCtxResetPersistingL2Cache",
      );
    });
  });

  it("disables the report/artifacts buttons until they're actually ready", async () => {
    mockGetJobStatus.mockResolvedValue(makeMockJob({ report_md_path: null, artifacts_tar_path: null }));

    render(<LiveJobView jobId="job_test0000001" />);

    await waitFor(() => {
      expect(screen.getByTestId("view-report-button")).toBeDisabled();
    });
    expect(screen.getByTestId("download-artifacts-button")).toHaveAttribute("aria-disabled", "true");
  });

  it("enables the report/artifacts buttons once the backend reports them ready", async () => {
    mockGetJobStatus.mockResolvedValue(
      makeMockJob({
        status: "completed",
        report_md_path: "/repo/jobs/job_test0000001/reports/migration_report.md",
        artifacts_tar_path: "/repo/jobs/job_test0000001/reports/job_test0000001_artifacts.tar.gz",
      }),
    );

    render(<LiveJobView jobId="job_test0000001" />);

    await waitFor(() => {
      expect(screen.getByTestId("view-report-button")).not.toBeDisabled();
    });
    expect(screen.getByTestId("download-artifacts-button")).toHaveAttribute("aria-disabled", "false");
  });

  it("shows a REAL DATA badge for a real job, sourced from job.mode — not any server-level prop", async () => {
    mockGetJobStatus.mockResolvedValue(makeMockJob({ mode: "REAL", gpu_arch: "gfx1100" }));

    render(<LiveJobView jobId="job_374d6e8c51d1" />);

    await waitFor(() => {
      const badge = screen.getByTestId("mode-badge");
      expect(badge).toHaveAttribute("data-job-mode", "REAL");
      expect(badge).toHaveTextContent("REAL DATA");
    });
  });

  it("shows a SIMULATED badge for a mock job", async () => {
    mockGetJobStatus.mockResolvedValue(makeMockJob({ mode: "MOCK" }));

    render(<LiveJobView jobId="job_test0000001" />);

    await waitFor(() => {
      const badge = screen.getByTestId("mode-badge");
      expect(badge).toHaveAttribute("data-job-mode", "MOCK");
      expect(badge).toHaveTextContent("SIMULATED");
    });
  });
});
