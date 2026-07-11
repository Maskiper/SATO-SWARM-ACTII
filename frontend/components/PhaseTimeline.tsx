import type { JobPhase, JobState } from "@/lib/types";

// ACT II's actual 5 pipeline phases (src/models/job.py's JobPhase has
// QUEUED/OPTIMIZING/COMPLETED/FAILED too, but run_baseline() only ever
// _advance()s through these five as real steps -- QUEUED is pre-start,
// OPTIMIZING is defined but never reached by run_baseline(), and
// COMPLETED/FAILED are terminal outcomes, not steps in this stepper).
const PHASES: JobPhase[] = ["Analysis", "Porting", "Validating", "Benchmarking", "Reporting"];

interface PhaseTimelineProps {
  job: JobState;
}

export default function PhaseTimeline({ job }: PhaseTimelineProps) {
  const completed = new Set(job.completed_phases);
  const failed = job.status === "failed";

  // Derived from the job's REAL messages, not hardcoded to repairDemo --
  // any seed where the repair loop actually engaged shows this badge,
  // any seed where it didn't (including repairDemo runs where hipcc
  // simply succeeded on the first try) does not.
  const repairLoopEngaged = job.messages.some((m) => m.agent === "Repair Loop");

  return (
    <div>
      <ol className="flex items-center w-full" data-testid="phase-timeline">
        {PHASES.map((phase, idx) => {
          const isCurrent = job.phase === phase;
          const isDone = completed.has(phase) && !isCurrent;
          const isFailedHere = failed && isCurrent;

          let circleClasses =
            "flex items-center justify-center w-8 h-8 rounded-full border-2 text-xs font-semibold shrink-0";
          if (isFailedHere) {
            circleClasses += " border-red-500 bg-red-500 text-white";
          } else if (isDone || (isCurrent && !failed && job.status === "completed")) {
            circleClasses += " border-emerald-500 bg-emerald-500 text-white";
          } else if (isCurrent) {
            circleClasses += " border-blue-500 bg-blue-500 text-white animate-pulse";
          } else {
            circleClasses += " border-gray-300 text-gray-400 dark:border-gray-600 dark:text-gray-500";
          }

          return (
            <li key={phase} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1">
                <div className={circleClasses} data-testid={`phase-step-${phase}`} data-state={
                  isFailedHere ? "failed" : isDone ? "done" : isCurrent ? "current" : "pending"
                }>
                  {isFailedHere ? "!" : isDone ? "✓" : idx + 1}
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400 whitespace-nowrap">
                  {phase}
                </span>
                {phase === "Porting" && repairLoopEngaged && (
                  <span
                    data-testid="repair-loop-badge"
                    className="mt-0.5 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                  >
                    repair loop engaged
                  </span>
                )}
              </div>
              {idx < PHASES.length - 1 && (
                <div
                  className={`h-0.5 flex-1 mx-2 ${
                    completed.has(phase) ? "bg-emerald-500" : "bg-gray-300 dark:bg-gray-600"
                  }`}
                />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
