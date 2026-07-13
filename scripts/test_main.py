#!/usr/bin/env python3
"""
SATO SWARM — FastAPI Backend Test Suite

Tests every src/main.py endpoint against a real in-process ASGI app
(FastAPI's TestClient — no live uvicorn process needed), plus the
CRITICAL equivalence test: run vectorAdd via the CLI path
(run_baseline() called directly, the exact same way
scripts/test_baseline.py does) and via the API path (TestClient POST
/jobs + poll), then diff the resulting JobState/report content
field-by-field. Any unexplained difference is reported as a real bug in
the wrapper, never silently reconciled — see EXPECTED_DIFFERENT_JOB_FIELDS
and normalize_report() for exactly which fields are expected to differ
(job_id, timestamps, paths — inherently different between ANY two
independent runs, even two CLI runs back to back) versus what must be
byte-identical.

Usage:
    SATOSWARM_MOCK=1 python scripts/test_main.py
"""

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import src.main as main_module
from src.baseline.pipeline import run_baseline
from src.main import app
from src.models.job import JobState, SeedId
from src.tools.execution import MOCK
from src.workspace.manager import WorkspaceManager

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS_ROOT = REPO_ROOT / "seeds"

client = TestClient(app)

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def poll_until_terminal(job_id: str, timeout_s: float = 30.0) -> dict | None:
    start = time.time()
    while time.time() - start < timeout_s:
        r = client.get(f"/jobs/{job_id}/status")
        if r.status_code != 200:
            return None
        data = r.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.2)
    return None


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — FastAPI Backend Test Suite")
    print("=" * 70)
    print(f"SATOSWARM_MOCK: {'1 (mock)' if MOCK else '0/unset (REAL)'}")
    if not MOCK:
        print("WARNING: this test assumes MOCK=1 for deterministic equivalence checking.")
    print()

    # --- 1. /health ---
    print("1. GET /health")
    r = client.get("/health")
    print(f"   -> {r.status_code} {r.json()}")
    check("200 OK", r.status_code == 200)
    body = r.json()
    check("system == 'SATO SWARM'", body.get("system") == "SATO SWARM")
    check("mode reflects real MOCK constant", body.get("mode") == ("MOCK" if MOCK else "REAL"))
    check("gpu_arch is a real live value (not hardcoded)", body.get("gpu_arch") is not None)
    check("memory_patterns is a positive int", isinstance(body.get("memory_patterns"), int) and body["memory_patterns"] > 0)
    check("tool_registry_tools == 8", body.get("tool_registry_tools") == 8)
    print()

    # --- 2. POST /jobs + full lifecycle ---
    print("2. POST /jobs?seed_id=vectorAdd + lifecycle")
    r = client.post("/jobs", params={"seed_id": "vectorAdd"})
    check("create returns 200", r.status_code == 200)
    job_id = r.json()["job_id"]
    check("status == 'queued'", r.json()["status"] == "queued")

    final = poll_until_terminal(job_id)
    check("job reaches a terminal state (genuinely pollable, not a stub)", final is not None and final["status"] in ("completed", "failed"))
    check("job completed successfully in mock mode", final is not None and final["status"] == "completed")

    r = client.get(f"/jobs/{job_id}/report")
    check("report available once job is done", r.status_code == 200)
    check("report matches the file on disk exactly", r.text == (WorkspaceManager(base_dir=REPO_ROOT / "jobs").get_workspace(job_id) / "reports" / "migration_report.md").read_text(encoding="utf-8"))

    r = client.get(f"/jobs/{job_id}/artifacts")
    check("artifacts available once job is done", r.status_code == 200)
    check("artifacts content non-empty", len(r.content) > 0)
    print()

    # --- 3. 404s ---
    print("3. Unknown job_id -> 404 on every job-scoped endpoint")
    for path in ("status", "report", "artifacts"):
        r = client.get(f"/jobs/does_not_exist_{path}/{path}")
        check(f"GET .../{path} for unknown job -> 404", r.status_code == 404)
    print()

    # --- 4. /demo/replay for all 4 seeds, real data ---
    print("4. POST /demo/replay for all 4 seeds")
    for seed in ("vectorAdd", "tiledMatmul", "reduction", "repairDemo"):
        r = client.post("/demo/replay", params={"seed_id": seed})
        print(f"   {seed}: {r.status_code}")
        check(f"{seed}: replay returns 200 (real job dir present)", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check(f"{seed}: replay job.seed_id matches", data["job"]["seed_id"] == seed)
            check(f"{seed}: replay report_md is real, non-empty content", bool(data["report_md"]) and len(data["report_md"]) > 100)
            check(f"{seed}: replay job.status == 'completed' (real hardware run)", data["job"]["status"] == "completed")
    print()

    # --- 5. /demo/replay for an unconfigured/missing seed scenario ---
    # (all 4 real seeds are present in this checkout right now, so this
    # only exercises the "unknown seed_id" 422 path, not the "dir missing"
    # 404 path -- that path is exercised implicitly whenever one of the
    # 4 real job dirs isn't present, which main.py handles explicitly,
    # not tested here since we can't safely delete real pod data to prove it)
    print("5. POST /demo/replay with invalid seed_id -> 422 (FastAPI enum validation)")
    r = client.post("/demo/replay", params={"seed_id": "not_a_real_seed"})
    print(f"   -> {r.status_code}")
    check("invalid seed_id rejected before reaching replay logic", r.status_code == 422)
    print()

    # --- 6. SSE stream actually delivers messages ---
    print("6. GET /jobs/{id}/stream — SSE delivers real messages")
    r2 = client.post("/jobs", params={"seed_id": "reduction"})
    stream_job_id = r2.json()["job_id"]
    received_ids: list[int] = []
    done_status = None
    with client.stream("GET", f"/jobs/{stream_job_id}/stream") as stream:
        check("SSE endpoint returns 200 with correct content-type", stream.status_code == 200 and "text/event-stream" in stream.headers.get("content-type", ""))
        current_event = "message"
        for line in stream.iter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                import json as _json
                payload = _json.loads(line.split(":", 1)[1].strip())
                if current_event == "done":
                    done_status = payload["status"]
                    break
                else:
                    received_ids.append(payload["id"])
                current_event = "message"
    final_reduction = poll_until_terminal(stream_job_id)
    expected_count = len(final_reduction["messages"]) if final_reduction else -1
    check("SSE delivered a 'done' event", done_status is not None)
    check("SSE 'done' status matches the job's real final status", done_status == (final_reduction["status"] if final_reduction else None))
    check(
        "SSE delivered every message with no gaps/duplicates (ids == 1..N)",
        received_ids == list(range(1, expected_count + 1)),
        f"got {received_ids}, expected 1..{expected_count}",
    )
    print()

    print("7. POST /admin/cleanup — deletes old jobs, NEVER deletes replay jobs")
    print("   (fully isolated from the real jobs/ directory — a temp dir throughout)")

    def make_fake_job_dir(base: Path, job_id: str, age_hours: float) -> Path:
        d = base / job_id
        (d / "reports").mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text("{}", encoding="utf-8")
        old_time = time.time() - age_hours * 3600
        os.utime(d, (old_time, old_time))  # set LAST — creating children above bumps mtime
        return d

    real_replay_ids = set(main_module.REPLAY_JOB_IDS.values())
    check("sanity: REPLAY_JOB_IDS has exactly 4 real job IDs", len(real_replay_ids) == 4, f"got {real_replay_ids}")

    # --- 7a. Direct WorkspaceManager.cleanup_old_jobs() call.
    #
    # Proves CAUSATION, not just correlation: the SAME real-REPLAY_JOB_IDS
    # -named directories, aged the SAME way (1000h, well past the 24h
    # threshold), are run through cleanup_old_jobs() TWICE in the same
    # temp dir — once WITHOUT protected_job_ids (must be deleted; this is
    # what rules out a false positive from a broken aging mechanism, e.g.
    # os.utime() silently not taking effect on this filesystem, which
    # would make "survives" below pass vacuously without protection ever
    # actually being exercised), then re-created and run again WITH
    # protected_job_ids (must survive). The delta between these two runs
    # in the same environment is the actual proof that protected_job_ids
    # is the reason they survive the second time. ---
    tmp_dir_a = Path(tempfile.mkdtemp(prefix="sato_cleanup_test_direct_"))
    try:
        tmp_ws_a = WorkspaceManager(base_dir=tmp_dir_a)

        # -- control: same names, same age, NO protection — must die --
        control_replay_dirs = [make_fake_job_dir(tmp_dir_a, jid, age_hours=1000) for jid in real_replay_ids]
        cleaned_control, skipped_control = tmp_ws_a.cleanup_old_jobs(max_age_hours=24, max_jobs=None, protected_job_ids=None)
        check(
            "direct control: WITHOUT protection, all 4 real-replay-ID-named dirs (aged 1000h) ARE deleted",
            cleaned_control == 4 and not any(d.exists() for d in control_replay_dirs),
            f"cleaned={cleaned_control}, still exist={[str(d) for d in control_replay_dirs if d.exists()]}",
        )
        check("direct control: skipped_replay_jobs == 0 with no protected_job_ids given", skipped_control == 0, f"got {skipped_control}")

        # -- now the actual protected scenario, same temp dir, same age --
        old_normal_a = make_fake_job_dir(tmp_dir_a, "job_old_normal0000a1", age_hours=48)
        recent_normal_a = make_fake_job_dir(tmp_dir_a, "job_recent_normal0a2", age_hours=1)
        old_replay_dirs_a = [make_fake_job_dir(tmp_dir_a, jid, age_hours=1000) for jid in real_replay_ids]

        cleaned_a, skipped_a = tmp_ws_a.cleanup_old_jobs(max_age_hours=24, max_jobs=None, protected_job_ids=real_replay_ids)
        check("direct: exactly 1 non-replay old dir reported cleaned", cleaned_a == 1, f"got {cleaned_a}")
        check("direct: skipped_replay_jobs == 4 (all real replay IDs present)", skipped_a == 4, f"got {skipped_a}")
        check("direct: the old NON-replay dir is actually gone from disk", not old_normal_a.exists())
        check("direct: the recent NON-replay dir is untouched", recent_normal_a.exists())
        check(
            "direct: the SAME real-replay-ID dirs deleted UNPROTECTED above now survive WITH protection — proves causation, not coincidence",
            all(d.exists() for d in old_replay_dirs_a),
            f"missing: {[str(d) for d in old_replay_dirs_a if not d.exists()]}",
        )
    finally:
        shutil.rmtree(tmp_dir_a, ignore_errors=True)

    # --- 7b. Wired through the REAL production endpoint — same causation
    # proof as 7a (control run without protection, then the real
    # protected run), but exercising POST /admin/cleanup itself, not just
    # the WorkspaceManager method it delegates to. src.main.WS (a
    # module-level singleton pointed at the real jobs/ dir — see
    # src/main.py) is monkeypatched to this isolated temp dir for the
    # duration of this block only, then restored — the real jobs/
    # directory (which holds the actual REPLAY_JOB_IDS captures) is
    # never touched or even opened by this test. The endpoint itself
    # takes no protected_job_ids parameter (it always derives it from
    # REPLAY_JOB_IDS server-side — see src/main.py's admin_cleanup()),
    # so the control run instead temporarily empties main_module.
    # REPLAY_JOB_IDS for that one call — the endpoint's own code is never
    # modified, only the data it reads, restored immediately after. ---
    tmp_dir_b = Path(tempfile.mkdtemp(prefix="sato_cleanup_test_endpoint_"))
    original_ws = main_module.WS
    try:
        main_module.WS = WorkspaceManager(base_dir=tmp_dir_b)

        # -- control: same real-replay-ID-named dirs, same age, but
        # REPLAY_JOB_IDS temporarily empty -- the endpoint's own
        # `protected_job_ids=set(REPLAY_JOB_IDS.values())` becomes an
        # empty set for this one call, so they must be deleted --
        control_replay_dirs_b = [make_fake_job_dir(tmp_dir_b, jid, age_hours=1000) for jid in real_replay_ids]
        original_replay_ids = main_module.REPLAY_JOB_IDS
        try:
            main_module.REPLAY_JOB_IDS = {}
            r_control = client.post("/admin/cleanup", params={"max_age_hours": 24})
        finally:
            main_module.REPLAY_JOB_IDS = original_replay_ids
        check("endpoint control: returns 200 with REPLAY_JOB_IDS temporarily empty", r_control.status_code == 200, r_control.text)
        body_control = r_control.json()
        check(
            "endpoint control: WITHOUT any protected IDs, all 4 real-replay-ID-named dirs (aged 1000h) ARE deleted",
            body_control.get("cleaned") == 4 and not any(d.exists() for d in control_replay_dirs_b),
            f"body={body_control}, still exist={[str(d) for d in control_replay_dirs_b if d.exists()]}",
        )
        check("endpoint control: skipped_replay_jobs == 0 with REPLAY_JOB_IDS empty", body_control.get("skipped_replay_jobs") == 0, f"got {body_control}")

        # -- now the real production endpoint, REPLAY_JOB_IDS intact,
        # same temp dir, same age --
        old_normal_b = make_fake_job_dir(tmp_dir_b, "job_old_normal0000b1", age_hours=48)
        recent_normal_b = make_fake_job_dir(tmp_dir_b, "job_recent_normal0b2", age_hours=1)
        old_replay_dirs_b = [make_fake_job_dir(tmp_dir_b, jid, age_hours=1000) for jid in real_replay_ids]

        r = client.post("/admin/cleanup", params={"max_age_hours": 24})
        check("endpoint: returns 200", r.status_code == 200, r.text)
        body = r.json()
        check("endpoint: response has exactly {cleaned, skipped_replay_jobs}", set(body.keys()) == {"cleaned", "skipped_replay_jobs"}, f"got {body}")
        check("endpoint: cleaned == 1", body.get("cleaned") == 1, f"got {body}")
        check("endpoint: skipped_replay_jobs == 4", body.get("skipped_replay_jobs") == 4, f"got {body}")
        check("endpoint: the old NON-replay dir is actually gone from disk", not old_normal_b.exists())
        check("endpoint: the recent NON-replay dir is untouched", recent_normal_b.exists())
        check(
            "endpoint: the SAME real-replay-ID dirs deleted UNPROTECTED above now survive through the REAL production endpoint — proves causation, not coincidence",
            all(d.exists() for d in old_replay_dirs_b),
            f"missing: {[str(d) for d in old_replay_dirs_b if not d.exists()]}",
        )
    finally:
        main_module.WS = original_ws
        shutil.rmtree(tmp_dir_b, ignore_errors=True)

    # --- 7c. Confirm the REAL jobs/ directory's actual replay captures
    # were never touched by any of the above (belt-and-suspenders — 7a/7b
    # never referenced the real WS/jobs dir at all, but this positively
    # confirms it rather than just trusting the isolation held). ---
    for jid in real_replay_ids:
        check(f"real jobs/{jid}/ untouched by this test", (REPO_ROOT / "jobs" / jid).exists())
    print()

    # =========================================================================
    print("=" * 70)
    print("CRITICAL — CLI vs API equivalence (mock vectorAdd)")
    print("=" * 70)

    print("Running via CLI path (run_baseline() called directly, same as scripts/test_baseline.py)...")
    cli_job_obj = JobState(seed_id=SeedId.VECTOR_ADD)
    cli_ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    cli_final = run_baseline(cli_job_obj, cli_ws, SEEDS_ROOT)
    cli_job = cli_final.model_dump(mode="json")
    cli_report_path = cli_ws.get_workspace(cli_job["job_id"]) / "reports" / "migration_report.md"
    cli_report = cli_report_path.read_text(encoding="utf-8") if cli_report_path.exists() else ""
    print(f"  CLI job_id: {cli_job['job_id']}")

    print("Running via API path (TestClient POST /jobs + poll)...")
    r = client.post("/jobs", params={"seed_id": "vectorAdd"})
    api_job_id = r.json()["job_id"]
    api_job = poll_until_terminal(api_job_id)
    assert api_job is not None, "API vectorAdd run never reached a terminal state"
    r = client.get(f"/jobs/{api_job_id}/report")
    api_report = r.text
    print(f"  API job_id: {api_job_id}")
    print()

    # Top-level fields expected to differ between ANY two independent runs
    # (random job_id, wall-clock timestamps, and everything derived from
    # either) -- not a bug to reconcile, just inherent non-determinism.
    EXPECTED_DIFFERENT_JOB_FIELDS = {
        "job_id",
        "created_at",
        "updated_at",
        "workspace_dir",
        "report_md_path",
        "artifacts_tar_path",
    }
    # metrics.captured_at is the SAME kind of wall-clock timestamp as
    # created_at/updated_at, just nested inside `metrics` rather than
    # top-level -- caught by this test's first real run (see the run
    # before this fix): metrics.raw/derived/timeseries were already
    # byte-identical between CLI and API, only captured_at differed, by
    # milliseconds. Compare metrics field-by-field so that one legitimate
    # timestamp doesn't mask a real diff anywhere else inside it.
    EXPECTED_DIFFERENT_METRICS_FIELDS = {"captured_at"}

    def compare_job_states(cli: dict, api: dict) -> list[str]:
        diffs = []
        for key in sorted(set(cli) | set(api)):
            if key in EXPECTED_DIFFERENT_JOB_FIELDS or key in ("messages", "metrics"):
                continue
            if cli.get(key) != api.get(key):
                diffs.append(f"  {key}: CLI={cli.get(key)!r}  !=  API={api.get(key)!r}")

        cli_metrics, api_metrics = cli.get("metrics", {}), api.get("metrics", {})
        for key in sorted(set(cli_metrics) | set(api_metrics)):
            if key in EXPECTED_DIFFERENT_METRICS_FIELDS:
                continue
            if cli_metrics.get(key) != api_metrics.get(key):
                diffs.append(f"  metrics.{key}: CLI={cli_metrics.get(key)!r}  !=  API={api_metrics.get(key)!r}")

        cli_msgs, api_msgs = cli.get("messages", []), api.get("messages", [])
        if len(cli_msgs) != len(api_msgs):
            diffs.append(f"  messages: CLI has {len(cli_msgs)} entries, API has {len(api_msgs)}")
        else:
            for i, (cm, am) in enumerate(zip(cli_msgs, api_msgs)):
                for field in ("id", "agent", "type", "content"):
                    if cm.get(field) != am.get(field):
                        diffs.append(f"  messages[{i}].{field}: CLI={cm.get(field)!r}  !=  API={am.get(field)!r}")
        return diffs

    REPORT_NORMALIZE = [
        (re.compile(r"\*\*Job ID\*\*: \S+"), "**Job ID**: <NORMALIZED>"),
        (re.compile(r"\*\*Date\*\*: \S+"), "**Date**: <NORMALIZED>"),
        (re.compile(r"\*\*Total Duration\*\*: \S+"), "**Total Duration**: <NORMALIZED>"),
        (re.compile(r"artifacts tar: \S+"), "artifacts tar: <NORMALIZED>"),
    ]

    def normalize_report(text: str) -> str:
        for pattern, repl in REPORT_NORMALIZE:
            text = pattern.sub(repl, text)
        return text

    job_diffs = compare_job_states(cli_job, api_job)
    check("JobState fields identical modulo job_id/timestamps/paths", len(job_diffs) == 0)
    if job_diffs:
        print("  DIFF (JobState):")
        for d in job_diffs:
            print(d)

    cli_report_norm = normalize_report(cli_report)
    api_report_norm = normalize_report(api_report)
    reports_match = cli_report_norm == api_report_norm
    check("migration_report.md identical modulo Job ID/Date/Duration/artifact filename", reports_match)
    if not reports_match:
        print("  DIFF (report.md, normalized):")
        cli_lines = cli_report_norm.splitlines()
        api_lines = api_report_norm.splitlines()
        for i, (a, b) in enumerate(zip(cli_lines, api_lines)):
            if a != b:
                print(f"    line {i}: CLI={a!r}")
                print(f"    line {i}: API={b!r}")
        if len(cli_lines) != len(api_lines):
            print(f"    line count: CLI={len(cli_lines)} API={len(api_lines)}")

    print()
    print("=" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
