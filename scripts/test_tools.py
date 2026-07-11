#!/usr/bin/env python3
"""
SATO SWARM — ToolRegistry Standalone Test

Exercises all 8 src/agents/tools.py ToolRegistry tools against a REAL job
workspace (created via the same WorkspaceManager + JobState path the real
pipeline uses), in mock mode — no pod needed. Also exercises the failure
paths (sandbox escape, unknown tool, apply_search_replace's not-found /
not-unique cases) since those are as much a part of the contract as the
happy path.

This does NOT touch src/baseline/pipeline.py and does not run the
pipeline — it only proves ToolRegistry itself is correct, standalone,
before it's wired into anything.

Usage:
    SATOSWARM_MOCK=1 python scripts/test_tools.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.tools import ToolRegistry
from src.models.job import JobState, SeedId
from src.tools.execution import MOCK
from src.workspace.manager import WorkspaceManager

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS_ROOT = REPO_ROOT / "seeds"

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


def shape_ok(r: dict) -> bool:
    return isinstance(r, dict) and set(r.keys()) == {"success", "result", "error"}


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — ToolRegistry Standalone Test")
    print("=" * 70)
    print(f"SATOSWARM_MOCK: {'1 (mock)' if MOCK else '0/unset (REAL — this test expects mock)'}")
    if not MOCK:
        print("WARNING: not running in mock mode. Set SATOSWARM_MOCK=1 and re-run.")
    print()

    ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    job = JobState(seed_id=SeedId.VECTOR_ADD)
    ws_dir = ws.create_workspace(job)
    ws.copy_seed(job, SEEDS_ROOT)
    print(f"Job ID:        {job.job_id}")
    print(f"Workspace dir: {ws_dir}")
    print()

    registry = ToolRegistry(job_id=job.job_id, workspace_dir=ws_dir)

    # --- 1. run_hipify ---
    print("1. run_hipify(source_dir='cuda_src', out_dir='hip_out')")
    r = registry.execute("run_hipify", source_dir="cuda_src", out_dir="hip_out")
    print(f"   -> {r}")
    check("returns {success, result, error} shape", shape_ok(r))
    check("succeeds in mock mode", r["success"] is True, r.get("error"))
    print()

    # --- 2. list_workspace_files (discover hipify's output) ---
    print("2. list_workspace_files(subdir='hip_out')")
    r = registry.execute("list_workspace_files", subdir="hip_out")
    print(f"   -> {r}")
    check("returns correct shape", shape_ok(r))
    check("finds hipify's output file", r["success"] and r["result"]["count"] >= 1)
    hip_files = r["result"]["files"] if r["success"] else []
    print()

    # --- 3. run_hipcc (fed directly from step 2's output) ---
    hip_source = hip_files[0] if hip_files else "hip_out/vectorAdd.hip.cpp"
    print(f"3. run_hipcc(hip_sources=[{hip_source!r}], out_binary='hip_out/vectorAdd_hip')")
    r = registry.execute("run_hipcc", hip_sources=[hip_source], out_binary="hip_out/vectorAdd_hip")
    print(f"   -> {r}")
    check("returns correct shape", shape_ok(r))
    check("succeeds in mock mode", r["success"] is True, r.get("error"))
    print()

    # --- 4. run_benchmark (binary was never actually written by mock hipcc) ---
    print("4. run_benchmark(binary='hip_out/vectorAdd_hip')")
    r = registry.execute("run_benchmark", binary="hip_out/vectorAdd_hip")
    print(f"   -> {r}")
    check("returns correct shape", shape_ok(r))
    check("succeeds in mock mode despite binary not existing on disk", r["success"] is True, r.get("error"))
    check("stdout contains a real Kernel time line", r["success"] and "Kernel time" in r["result"]["stdout"])
    print()

    # --- 5. capture_amd_smi ---
    print("5. capture_amd_smi()")
    r = registry.execute("capture_amd_smi")
    print(f"   -> {r}")
    check("returns correct shape", shape_ok(r))
    check("succeeds", r["success"] is True)
    check("metrics dict has gpu_utilization_percent", r["success"] and "gpu_utilization_percent" in r["result"]["metrics"])
    print()

    # --- 6. read_file ---
    print("6. read_file(relative_path='cuda_src/vectorAdd.cu')")
    r = registry.execute("read_file", relative_path="cuda_src/vectorAdd.cu")
    n = len(r["result"]["content"]) if r["success"] else 0
    print(f"   -> success={r['success']}, {n} bytes read, error={r['error']}")
    check("returns correct shape", shape_ok(r))
    check(
        "content matches the real seed file",
        r["success"] and "vectorAdd seed completed successfully." in r["result"]["content"],
    )
    print()

    # --- 7. apply_search_replace: happy path (unique match) ---
    print("7. apply_search_replace: unique match")
    r = registry.execute(
        "apply_search_replace",
        relative_path="cuda_src/vectorAdd.cu",
        old_text="vectorAdd seed completed successfully.",
        new_text="vectorAdd seed completed successfully. [PATCHED BY TOOL TEST]",
    )
    print(f"   -> {r}")
    check("returns correct shape", shape_ok(r))
    check("succeeds on a unique match", r["success"] is True, r.get("error"))

    r2 = registry.execute("read_file", relative_path="cuda_src/vectorAdd.cu")
    check(
        "the edit was actually written to disk",
        r2["success"] and "[PATCHED BY TOOL TEST]" in r2["result"]["content"],
    )
    print()

    # --- 8. apply_search_replace: text not found ---
    print("8. apply_search_replace: old_text not found")
    r = registry.execute(
        "apply_search_replace",
        relative_path="cuda_src/vectorAdd.cu",
        old_text="THIS_STRING_DOES_NOT_EXIST_ANYWHERE_12345",
        new_text="x",
    )
    print(f"   -> {r}")
    check(
        "fails cleanly (not a crash) with a 'not found' error",
        r["success"] is False and "not found" in (r["error"] or ""),
    )
    print()

    # --- 9. apply_search_replace: text not unique ---
    print("9. apply_search_replace: old_text matches multiple locations")
    r = registry.execute(
        "apply_search_replace",
        relative_path="cuda_src/vectorAdd.cu",
        old_text="CHECK_CUDA(cudaFree(",
        new_text="x",
    )
    print(f"   -> {r}")
    check(
        "fails cleanly (not a crash) with a 'not unique' error",
        r["success"] is False and "not unique" in (r["error"] or ""),
    )
    r3 = registry.execute("read_file", relative_path="cuda_src/vectorAdd.cu")
    check(
        "file untouched when the match was ambiguous",
        r3["success"] and r3["result"]["content"].count("CHECK_CUDA(cudaFree(") == 3,
    )
    print()

    # --- 10. list_workspace_files: full workspace ---
    print("10. list_workspace_files() — full workspace")
    r = registry.execute("list_workspace_files")
    print(f"   -> success={r['success']}, count={r['result']['count'] if r['success'] else 'n/a'}")
    check("returns correct shape", shape_ok(r))
    check(
        "sees the cuda_src seed file",
        r["success"] and any(f.endswith("cuda_src/vectorAdd.cu") for f in r["result"]["files"]),
    )
    print()

    # --- 11. write_agent_note x2 (confirms append, not overwrite) ---
    print("11. write_agent_note() x2")
    r1 = registry.execute("write_agent_note", note="First note from the standalone tool test.")
    r2 = registry.execute("write_agent_note", note="Second note — confirms append, not overwrite.")
    print(f"   -> {r1}")
    print(f"   -> {r2}")
    check("returns correct shape", shape_ok(r1))
    check("both calls succeed", r1["success"] and r2["success"])

    r3 = registry.execute("read_file", relative_path="notes/blackboard.md")
    both_present = r3["success"] and "First note" in r3["result"]["content"] and "Second note" in r3["result"]["content"]
    check("both notes present in notes/blackboard.md (appended, not overwritten)", both_present)
    print()

    # --- 12. Sandbox escape: relative traversal ---
    print("12. Sandbox escape attempt: '../../../../etc/passwd'")
    r = registry.execute("read_file", relative_path="../../../../etc/passwd")
    print(f"   -> {r}")
    check(
        "refused cleanly (not a crash, not a real read)",
        r["success"] is False and "outside" in (r["error"] or ""),
    )
    print()

    # --- 13. Sandbox escape: absolute path override ---
    print("13. Sandbox escape attempt: absolute path outside workspace")
    r = registry.execute("read_file", relative_path="C:\\Windows\\System32\\drivers\\etc\\hosts")
    print(f"   -> {r}")
    check("refused cleanly (not a crash, not a real read)", r["success"] is False)
    print()

    # --- 14. Unknown tool name ---
    print("14. execute('delete_everything') — unknown tool")
    r = registry.execute("delete_everything", path="/")
    print(f"   -> {r}")
    check(
        "fails cleanly with 'Unknown tool' error, not a crash",
        r["success"] is False and "Unknown tool" in (r["error"] or ""),
    )
    print()

    print("=" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
