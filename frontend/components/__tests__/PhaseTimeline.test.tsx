import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PhaseTimeline from "@/components/PhaseTimeline";
import { makeMockJob } from "@/lib/test-fixtures";

describe("PhaseTimeline", () => {
  it("renders all 5 real pipeline phases", () => {
    render(<PhaseTimeline job={makeMockJob()} />);
    for (const phase of ["Analysis", "Porting", "Validating", "Benchmarking", "Reporting"]) {
      expect(screen.getByText(phase)).toBeInTheDocument();
    }
  });

  it("marks the current phase distinctly from completed/pending ones", () => {
    const job = makeMockJob({ phase: "Benchmarking", completed_phases: ["Analysis", "Porting", "Validating"] });
    render(<PhaseTimeline job={job} />);
    expect(screen.getByTestId("phase-step-Analysis")).toHaveAttribute("data-state", "done");
    expect(screen.getByTestId("phase-step-Benchmarking")).toHaveAttribute("data-state", "current");
    expect(screen.getByTestId("phase-step-Reporting")).toHaveAttribute("data-state", "pending");
  });

  it("shows a failed indicator on the phase where a FAILED job actually stopped", () => {
    const job = makeMockJob({
      phase: "Porting",
      status: "failed",
      completed_phases: ["Analysis"],
      error: "hipcc failed: use of undeclared identifier 'cudaCtxResetPersistingL2Cache'",
    });
    render(<PhaseTimeline job={job} />);
    expect(screen.getByTestId("phase-step-Porting")).toHaveAttribute("data-state", "failed");
  });

  it('shows the "repair loop engaged" badge only when a real Repair Loop message exists', () => {
    const withRepair = makeMockJob({
      phase: "Porting",
      messages: [
        ...makeMockJob().messages,
        { id: 4, agent: "Repair Loop", timestamp: "01:00:03", type: "observation", content: "Matched pattern gap_cudaCtxResetPersistingL2Cache." },
      ],
    });
    render(<PhaseTimeline job={withRepair} />);
    expect(screen.getByTestId("repair-loop-badge")).toBeInTheDocument();
  });

  it('does NOT show the "repair loop engaged" badge when no Repair Loop message exists, even for repairDemo', () => {
    const noRepair = makeMockJob({ seed_id: "repairDemo", phase: "Porting" });
    render(<PhaseTimeline job={noRepair} />);
    expect(screen.queryByTestId("repair-loop-badge")).not.toBeInTheDocument();
  });
});
