import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ReportViewer from "@/components/ReportViewer";
import { MOCK_FAILED_REPORT_MD, MOCK_REPORT_MD } from "@/lib/test-fixtures";

describe("ReportViewer", () => {
  it("renders the real markdown content via react-markdown (not hand-rolled)", () => {
    render(<ReportViewer jobId="job_test0000001" reportMarkdown={MOCK_REPORT_MD} />);
    // react-markdown turns "# ..." into a real <h1>, not a literal "#" string
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "SATO SWARM Migration Report — vectorAdd",
    );
    expect(screen.getByText(/Achieved 4601.23/)).toBeInTheDocument();
  });

  it('shows a green (non-failed) header when the report\'s own "Final Status" says COMPLETED', () => {
    render(<ReportViewer jobId="job_test0000001" reportMarkdown={MOCK_REPORT_MD} />);
    const header = screen.getByTestId("report-header");
    expect(header).toHaveAttribute("data-failed", "false");
    expect(header).toHaveTextContent("COMPLETED");
  });

  it('shows a red-tinted header ONLY when the report\'s own "Final Status" says FAILED', () => {
    render(<ReportViewer jobId="job_test0000002" reportMarkdown={MOCK_FAILED_REPORT_MD} />);
    const header = screen.getByTestId("report-header");
    expect(header).toHaveAttribute("data-failed", "true");
    expect(header).toHaveTextContent("FAILED");
  });

  it("provides a download-artifacts link scoped to the real job id", () => {
    render(<ReportViewer jobId="job_test0000001" reportMarkdown={MOCK_REPORT_MD} />);
    const link = screen.getByTestId("report-download-artifacts");
    expect(link).toHaveAttribute("href", expect.stringContaining("job_test0000001"));
  });
});
