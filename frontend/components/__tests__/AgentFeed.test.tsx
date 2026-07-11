import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentFeed from "@/components/AgentFeed";
import { makeMockJob } from "@/lib/test-fixtures";

const mockStreamJob = vi.fn();
const mockGetJobStatus = vi.fn();

vi.mock("@/lib/api", () => ({
  streamJob: (...args: unknown[]) => mockStreamJob(...args),
  getJobStatus: (...args: unknown[]) => mockGetJobStatus(...args),
}));

describe("AgentFeed", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetJobStatus.mockResolvedValue(makeMockJob());
  });

  it("renders real AgentMessage entries delivered via the SSE mock", async () => {
    mockStreamJob.mockImplementation((_jobId, handlers) => {
      handlers.onMessage({
        id: 1,
        agent: "Baseline Orchestrator",
        timestamp: "01:00:00",
        type: "thought",
        content: "Starting baseline pipeline for vectorAdd.",
      });
      return () => {};
    });

    render(<AgentFeed jobId="job_test0000001" />);

    await waitFor(() => {
      expect(screen.getByText("Starting baseline pipeline for vectorAdd.")).toBeInTheDocument();
    });
    const message = screen.getByTestId("agent-message");
    expect(message).toHaveAttribute("data-agent", "Baseline Orchestrator");
  });

  it("shows a polling-fallback notice when the stream reports an error, without losing messages", async () => {
    mockStreamJob.mockImplementation((_jobId, handlers) => {
      handlers.onMessage({ id: 1, agent: "Baseline Orchestrator", timestamp: "01:00:00", type: "thought", content: "Starting." });
      handlers.onError("stream connection error");
      return () => {};
    });

    render(<AgentFeed jobId="job_test0000001" />);

    await waitFor(() => {
      expect(screen.getByText(/polling fallback active/i)).toBeInTheDocument();
    });
    expect(screen.getByText("Starting.")).toBeInTheDocument();
  });

  it("shows an empty state before any message has arrived", () => {
    mockStreamJob.mockImplementation(() => () => {});
    render(<AgentFeed jobId="job_test0000001" />);
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument();
  });
});
