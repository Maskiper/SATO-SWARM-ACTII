"use client";

import { useEffect, useRef, useState } from "react";
import { getJobStatus, streamJob } from "@/lib/api";
import type { AgentMessage } from "@/lib/types";

// Every agent name run_baseline() actually appends messages under (see
// src/baseline/pipeline.py's _append_message() call sites) -- "Repair
// Loop" only ever appears for SeedId.REPAIR_DEMO jobs where hipcc failed
// and the repair loop engaged (see PhaseTimeline's badge, which reads
// this same real message data rather than hardcoding seed-specific
// behavior).
const AGENT_COLORS: Record<string, string> = {
  "Baseline Orchestrator": "border-slate-400 bg-slate-50 text-slate-900 dark:bg-slate-800 dark:text-slate-100",
  "Baseline Analyzer": "border-blue-400 bg-blue-50 text-blue-900 dark:bg-blue-950 dark:text-blue-100",
  "HIP Porting Specialist": "border-violet-400 bg-violet-50 text-violet-900 dark:bg-violet-950 dark:text-violet-100",
  "Repair Loop": "border-amber-400 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-100",
  "Benchmark & Profiler": "border-emerald-400 bg-emerald-50 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-100",
  Validator: "border-cyan-400 bg-cyan-50 text-cyan-900 dark:bg-cyan-950 dark:text-cyan-100",
  Reporter: "border-fuchsia-400 bg-fuchsia-50 text-fuchsia-900 dark:bg-fuchsia-950 dark:text-fuchsia-100",
};
const DEFAULT_COLOR = "border-gray-400 bg-gray-50 text-gray-900 dark:bg-gray-800 dark:text-gray-100";

const TYPE_LABEL: Record<AgentMessage["type"], string> = {
  thought: "thought",
  action: "action",
  observation: "observation",
};

interface AgentFeedProps {
  jobId: string;
}

function mergeMessages(prev: AgentMessage[], incoming: AgentMessage[]): AgentMessage[] {
  const byId = new Map(prev.map((m) => [m.id, m]));
  for (const m of incoming) byId.set(m.id, m);
  return Array.from(byId.values()).sort((a, b) => a.id - b.id);
}

export default function AgentFeed({ jobId }: AgentFeedProps) {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Primary path: SSE, backed by src/main.py's pure file-watching bridge
  // over the same state.json WorkspaceManager already writes.
  //
  // No manual setMessages([])/setConnectionError(null) reset here on
  // jobId change: LiveJobView renders this with key={jobId}, so React
  // remounts (not re-renders) this component per job, which naturally
  // resets useState's initial values -- the idiomatic fix for
  // react-hooks/set-state-in-effect rather than calling setState
  // synchronously inside the effect body.
  useEffect(() => {
    const cleanup = streamJob(jobId, {
      onMessage: (m) => setMessages((prev) => mergeMessages(prev, [m])),
      onError: (detail) => setConnectionError(detail),
    });
    return cleanup;
  }, [jobId]);

  // Polling fallback: reads the exact same source of truth (GET
  // .../status, which reads state.json) SSE does -- not a second,
  // divergent data source, just a redundant delivery path for
  // browsers/proxies where EventSource misbehaves. Runs continuously
  // alongside SSE (cheap, idempotent merge by message id) rather than
  // only activating after a detected failure, so a partially-working SSE
  // connection can't silently under-deliver messages.
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const job = await getJobStatus(jobId);
        if (!cancelled) setMessages((prev) => mergeMessages(prev, job.messages));
      } catch {
        // transient — next interval tries again
      }
    };
    const interval = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [jobId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-1 pb-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Agent Feed</h3>
        {connectionError && (
          <span className="text-xs text-amber-600 dark:text-amber-400">
            live stream interrupted — polling fallback active
          </span>
        )}
      </div>
      <div
        ref={scrollRef}
        data-testid="agent-feed-scroll"
        className="flex-1 overflow-y-auto space-y-2 pr-1"
      >
        {messages.length === 0 && (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic">
            No messages yet.
          </p>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            data-testid="agent-message"
            data-agent={m.agent}
            className={`rounded-md border-l-4 px-3 py-2 text-sm ${AGENT_COLORS[m.agent] ?? DEFAULT_COLOR}`}
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-semibold">{m.agent}</span>
              <span className="text-xs opacity-70 whitespace-nowrap">
                {m.timestamp} · {TYPE_LABEL[m.type]}
              </span>
            </div>
            <p className="mt-1 whitespace-pre-wrap break-words">{m.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
