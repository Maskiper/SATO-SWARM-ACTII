"""SATO SWARM — FastAPI routing layer.

Thin wrapper ONLY: every endpoint calls directly into the same
already-proven functions the CLI (scripts/test_baseline.py) already
uses — src/baseline/pipeline.py's run_baseline(), WorkspaceManager,
PortingMemory, src/tools/execution.py's MOCK/detect_gpu_arch(). No
pipeline logic is reimplemented here. See tests/test_main.py's
CRITICAL equivalence test, which diffs a CLI run against an API run of
the same seed to prove this wrapper doesn't silently diverge from the
already-proven behavior.

Job state is never duplicated in memory — every read (status, report,
artifacts, replay, SSE) re-reads the SAME state.json WorkspaceManager
already writes, via WorkspaceManager.load_state()/get_workspace(). This
also means report/artifact paths are always resolved locally from
job_id (base_dir / job_id / ...), never trusted from a path STRING
stored inside state.json — job dirs pulled from the real pod store an
absolute POD path there (confirmed: "/workspace/SATO-SWARM-ACTII/jobs/..."
in the 4 real job dirs used by /demo/replay below), which doesn't exist
on whatever machine happens to be running this API.

Local-only: binds 127.0.0.1 (see the __main__ block), CORS restricted to
localhost dev-server origins. Not intended to be exposed off this
machine as-is — if that's ever needed, ALLOWED_ORIGINS below needs to
change to the actual deployed frontend origin, and the bind host/auth
story needs real thought; flagging that here rather than deciding it.

Run with (from repo root):
    SATOSWARM_MOCK=1 uvicorn src.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from src.agents.tools import ToolRegistry
from src.baseline.pipeline import run_baseline
from src.memory.loader import PortingMemory
from src.models.job import JobState, JobStatus, SeedId
from src.tools.execution import MOCK, detect_gpu_arch
from src.workspace.manager import WorkspaceManager

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS_ROOT = REPO_ROOT / "seeds"
WS = WorkspaceManager(base_dir=REPO_ROOT / "jobs")

# Real, historical, already-completed real-hardware job dirs (pulled back
# from the pod) — used ONLY by /demo/replay, which never runs a new job.
# If any of these aren't present in this checkout, that endpoint reports
# it plainly (404 with an explicit message) rather than substituting
# anything else.
REPLAY_JOB_IDS: dict[str, str] = {
    SeedId.VECTOR_ADD.value: "job_374d6e8c51d1",
    SeedId.TILED_MATMUL.value: "job_7eeb1f8358f8",
    SeedId.REDUCTION.value: "job_09ef95c5f62b",
    SeedId.REPAIR_DEMO.value: "job_1684fdb652d5",
}

# In-memory, process-lifetime only — "jobs run this session" per /health's
# spec, not a persisted cross-restart count. Never used as a source of
# truth for anything except this one counter; every other endpoint reads
# state.json, never this.
_jobs_run_this_session = 0

app = FastAPI(title="SATO SWARM")

# Local-only demo: CORS is permissive to localhost dev-server origins
# ONLY. If this backend is ever run somewhere reachable beyond
# localhost, this list needs to be locked down to the actual deployed
# frontend origin instead — flagging this, not deciding it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    system: str
    mode: str
    gpu_arch: Optional[str]
    memory_patterns: int
    jobs_run_this_session: int
    tool_registry_tools: int


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class AdminCleanupResponse(BaseModel):
    cleaned: int
    skipped_replay_jobs: int


def _report_path(job_id: str) -> Path:
    return WS.get_workspace(job_id) / "reports" / "migration_report.md"


def _artifacts_path(job_id: str) -> Path:
    return WS.get_workspace(job_id) / "reports" / f"{job_id}_artifacts.tar.gz"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        system="SATO SWARM",
        mode="MOCK" if MOCK else "REAL",
        gpu_arch=detect_gpu_arch(),
        memory_patterns=len(PortingMemory()),
        jobs_run_this_session=_jobs_run_this_session,
        tool_registry_tools=len(ToolRegistry.TOOL_NAMES),
    )


@app.post("/jobs", response_model=JobCreateResponse)
def create_job(
    background_tasks: BackgroundTasks,
    seed_id: SeedId = Query(..., description="vectorAdd | tiledMatmul | reduction | repairDemo"),
) -> JobCreateResponse:
    global _jobs_run_this_session
    job = JobState(seed_id=seed_id)
    # Create the workspace (+ initial state.json) synchronously, before
    # responding — so a client that immediately polls GET .../status right
    # after this call back never sees a spurious 404 while the background
    # task is still waiting for a thread. run_baseline() itself calls
    # create_workspace() again as its own first step; that's a harmless,
    # idempotent no-op re-run (see WorkspaceManager.create_workspace()),
    # not a duplicated side effect.
    WS.create_workspace(job)
    _jobs_run_this_session += 1
    background_tasks.add_task(
        run_baseline, job, WS, SEEDS_ROOT, on_progress=lambda j: WS.write_state(j)
    )
    return JobCreateResponse(job_id=job.job_id, status="queued")


@app.get("/jobs/{job_id}/status", response_model=JobState)
def job_status(job_id: str) -> JobState:
    job = WS.load_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id!r}")
    return job


@app.get("/jobs/{job_id}/report")
def job_report(job_id: str) -> PlainTextResponse:
    job = WS.load_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id!r}")
    path = _report_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not ready yet — job hasn't reached the Reporting phase.")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> FileResponse:
    job = WS.load_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id!r}")
    path = _artifacts_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifacts tar not ready yet — job hasn't reached the Reporting phase.")
    return FileResponse(path, media_type="application/gzip", filename=path.name)


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    """Pure file-watching bridge: polls the SAME state.json
    WorkspaceManager already writes, pushes any AgentMessage entries not
    yet seen down the SSE connection. run_baseline() has no idea this
    endpoint exists — it just keeps calling ws.write_state() the same
    way it always has (plus the on_progress hook passed in create_job(),
    which is an existing, already-supported extension point, not a
    modification to run_baseline() itself).

    Poll interval is a plain trade-off, not a proof: 750ms is inside the
    "~500ms-1s" the spec asked for. Because run_baseline() only calls
    ws.write_state() at specific checkpoints (phase transitions, plus
    wherever on_progress() fires), messages can still arrive at this
    endpoint in small batches rather than strictly one at a time — that
    granularity is inherited from the existing, unmodified pipeline, not
    something this endpoint controls.
    """
    async def event_generator():
        last_count = 0
        terminal = {JobStatus.COMPLETED.value, JobStatus.FAILED.value}
        while True:
            job = WS.load_state(job_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'error': f'unknown job_id: {job_id}'})}\n\n"
                return
            if len(job.messages) > last_count:
                for m in job.messages[last_count:]:
                    yield f"data: {m.model_dump_json()}\n\n"
                last_count = len(job.messages)
            if job.status.value in terminal:
                yield f"event: done\ndata: {json.dumps({'status': job.status.value})}\n\n"
                return
            await asyncio.sleep(0.75)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/demo/replay")
def demo_replay(
    seed_id: SeedId = Query(..., description="vectorAdd | tiledMatmul | reduction | repairDemo"),
) -> JSONResponse:
    """Returns the report + JobState from an ACTUAL, already-completed
    real-hardware job for this seed — never runs anything new. If the
    job directory isn't present in this checkout, says so plainly
    (404 + exact expected path) rather than substituting mock data or
    any other seed's numbers.
    """
    job_id = REPLAY_JOB_IDS.get(seed_id.value)
    if job_id is None:
        raise HTTPException(status_code=404, detail=f"No replay job configured for seed {seed_id.value!r}.")

    ws_dir = WS.get_workspace(job_id)
    if not ws_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Real completed job jobs/{job_id}/ for seed {seed_id.value!r} is not present in "
                f"this checkout. Pull it from the pod first: "
                f"scp <POD_HOST>:~/sato-swarm-actii/jobs/{job_id} -r ."
            ),
        )
    job = WS.load_state(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"jobs/{job_id}/state.json is missing or unreadable.")

    report_path = _report_path(job_id)
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else None

    return JSONResponse({"job": json.loads(job.model_dump_json()), "report_md": report_md})


@app.post("/admin/cleanup", response_model=AdminCleanupResponse)
def admin_cleanup(
    max_age_hours: int = Query(24, description="Delete job directories under jobs/ older than this many hours"),
) -> AdminCleanupResponse:
    """Deletes job directories under jobs/ older than max_age_hours
    (default 24 — one day; chosen to match WorkspaceManager.cleanup_old_
    jobs()'s pre-existing default rather than invent a new number).

    NEVER deletes the 4 real captured job directories /demo/replay
    depends on (REPLAY_JOB_IDS' values) — passed explicitly as
    protected_job_ids, skipped unconditionally regardless of age. This
    matters concretely, not just in theory: those 4 real captures are,
    as of this writing, already more than 24 hours old, so an
    unprotected age sweep at the default setting would delete them on
    its very first call.

    Count-based truncation (WorkspaceManager.cleanup_old_jobs()'s
    max_jobs) is deliberately NOT applied here — this endpoint's own
    contract is age-only (the one parameter it exposes); silently also
    deleting extra jobs by count would be a surprise side effect of an
    age-scoped request.
    """
    cleaned, skipped_replay_jobs = WS.cleanup_old_jobs(
        max_age_hours=max_age_hours,
        max_jobs=None,
        protected_job_ids=set(REPLAY_JOB_IDS.values()),
    )
    return AdminCleanupResponse(cleaned=cleaned, skipped_replay_jobs=skipped_replay_jobs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
