import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MetricsDashboard from "@/components/MetricsDashboard";
import { makeMockJob } from "@/lib/test-fixtures";

describe("MetricsDashboard", () => {
  it("renders real captured values, not fabricated ones", () => {
    render(<MetricsDashboard job={makeMockJob()} />);
    expect(screen.getByText("0.652 ms")).toBeInTheDocument();
    expect(screen.getByText("4601.23 GB/s")).toBeInTheDocument();
    expect(screen.getByText("86.8%")).toBeInTheDocument();
    expect(screen.getByText("92.0%")).toBeInTheDocument();
  });

  it('shows "Not captured" for null raw telemetry, never a fabricated number', () => {
    const job = makeMockJob({
      metrics: {
        ...makeMockJob().metrics,
        raw: {
          gpu_utilization_percent: null,
          power_watts_avg: null,
          power_watts_peak: null,
          temperature_c: null,
          memory_used_mb: null,
          clock_sclk_mhz: null,
          clock_mclk_mhz: null,
        },
      },
    });
    render(<MetricsDashboard job={job} />);
    expect(screen.getAllByText("Not captured").length).toBeGreaterThan(0);
  });

  it('shows "Not applicable" for efficiency when no theoretical peak was computed', () => {
    const job = makeMockJob({
      metrics: {
        ...makeMockJob().metrics,
        derived: {
          ...makeMockJob().metrics.derived,
          efficiency_percent: null,
          efficiency_tflops_percent: null,
          theoretical_peak_source: null,
        },
      },
    });
    render(<MetricsDashboard job={job} />);
    expect(screen.getByText("Not applicable")).toBeInTheDocument();
  });

  it("shows the honest per-job MOCK indicator when job.mode is MOCK, not a real-hardware claim", () => {
    render(<MetricsDashboard job={makeMockJob({ mode: "MOCK" })} />);
    const footer = screen.getByTestId("metrics-mode-footer");
    expect(footer).toHaveAttribute("data-job-mode", "MOCK");
    expect(footer).toHaveTextContent(/this job's data is simulated/i);
    expect(footer).not.toHaveTextContent(/ran on real hardware/i);
  });

  it("shows the real-hardware indicator when job.mode is REAL, regardless of server mode", () => {
    render(<MetricsDashboard job={makeMockJob({ mode: "REAL" })} />);
    const footer = screen.getByTestId("metrics-mode-footer");
    expect(footer).toHaveAttribute("data-job-mode", "REAL");
    expect(footer).toHaveTextContent(/this job ran on real hardware/i);
  });
});
