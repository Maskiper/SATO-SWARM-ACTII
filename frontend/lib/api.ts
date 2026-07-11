import type {
  AgentMessage,
  HealthResponse,
  JobCreateResponse,
  JobState,
  JobStatus,
  ReplayResponse,
  SeedId,
} from "./types";

// Local-only backend (see src/main.py's own docstring) — same origin
// assumption on both sides. Override via NEXT_PUBLIC_API_BASE_URL if the
// backend ever runs on a different port during development.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(`API error ${status}: ${detail}`);
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON — keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health");
}

export async function createJob(seedId: SeedId): Promise<JobCreateResponse> {
  return getJson<JobCreateResponse>(
    `/jobs?seed_id=${encodeURIComponent(seedId)}`,
    { method: "POST" },
  );
}

export async function getJobStatus(jobId: string): Promise<JobState> {
  return getJson<JobState>(`/jobs/${encodeURIComponent(jobId)}/status`);
}

export async function getJobReport(jobId: string): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/report`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // not JSON
    }
    throw new ApiError(res.status, detail);
  }
  return res.text();
}

export function getJobArtifactsUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/artifacts`;
}

export async function replaySeed(seedId: SeedId): Promise<ReplayResponse> {
  return getJson<ReplayResponse>(
    `/demo/replay?seed_id=${encodeURIComponent(seedId)}`,
    { method: "POST" },
  );
}

export interface StreamHandlers {
  onMessage: (message: AgentMessage) => void;
  onDone?: (status: JobStatus) => void;
  onError?: (detail: string) => void;
}

/**
 * Wraps the backend's SSE endpoint (GET /jobs/{id}/stream). Returns a
 * cleanup function that closes the connection — callers MUST call it on
 * unmount, same as any other subscription. AgentFeed also has a polling
 * fallback (see that component) for browsers/proxies where EventSource
 * doesn't behave, since the backend contract here is "poll state.json and
 * push new messages down SSE" — a plain reconnect-based fallback is a
 * legitimate substitute for the same underlying data, not a different
 * source of truth.
 */
export function streamJob(jobId: string, handlers: StreamHandlers): () => void {
  const source = new EventSource(
    `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/stream`,
  );

  source.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data) as AgentMessage;
      handlers.onMessage(message);
    } catch {
      // malformed event data — ignore this one event, keep the stream open
    }
  };

  source.addEventListener("done", (event) => {
    try {
      const data = JSON.parse((event as MessageEvent).data);
      handlers.onDone?.(data.status as JobStatus);
    } finally {
      source.close();
    }
  });

  source.addEventListener("error", (event) => {
    const messageEvent = event as MessageEvent;
    let detail = "stream connection error";
    if (messageEvent.data) {
      try {
        detail = JSON.parse(messageEvent.data).error ?? detail;
      } catch {
        // ignore parse failure, use default detail
      }
    }
    handlers.onError?.(detail);
  });

  return () => source.close();
}
