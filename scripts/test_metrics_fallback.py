#!/usr/bin/env python3
"""
SATO SWARM — Unrecognized Metrics Format Fallback Test

Tests the Part C fallback: when a binary's stdout matches NONE of
src/tools/execution.py's parse_binary_output_for_metrics() formats, the
raw text is captured verbatim (job.unrecognized_output_snippet) and
surfaced directly in the report — instead of every field just reading
"Not captured" with no further context.

TEST A — parse_binary_output_for_metrics() unit-level: synthetic
non-matching text returns an empty dict, confirming this project's own
fallback trigger condition ("parsed == {}") is exactly right.

TEST B — wired: monkeypatches ONLY src.baseline.pipeline's own
run_binary reference (same monkeypatch-a-bare-name convention
scripts/test_repair_loop.py already uses for run_hipcc/run_hipify) to
return real, meaningful, but completely unrecognized stdout, then calls
the REAL, UNMODIFIED run_baseline() end-to-end.

REGRESSION — the 4 original seeds + multiFileDemo (run unpatched, real
MOCK-mode run_binary) must all have unrecognized_output_snippet is None
— their output is always recognized.

Usage:
    SATOSWARM_MOCK=1 python scripts/test_metrics_fallback.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.baseline.pipeline as pipeline_module
from src.baseline.pipeline import run_baseline
from src.models.job import JobState, SeedId
from src.tools.execution import MOCK, parse_binary_output_for_metrics
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


UNRECOGNIZED_STDOUT = (
    "Custom debug dump from a hypothetical future seed:\n"
    "foo=42, bar=99, status=OK\n"
    "(this line intentionally matches none of parse_binary_output_for_metrics()'s regexes)\n"
)


def test_parser_unit() -> None:
    print("=" * 70)
    print("A. parse_binary_output_for_metrics() — synthetic non-matching text")
    print("=" * 70)
    parsed = parse_binary_output_for_metrics(UNRECOGNIZED_STDOUT)
    check("completely unrecognized text parses to an empty dict", parsed == {}, f"got {parsed}")

    parsed_recognized = parse_binary_output_for_metrics("Kernel time: 1.000 ms\n")
    check("a real recognized line still parses normally (sanity check)", parsed_recognized == {"kernel_time_ms": 1.0}, f"got {parsed_recognized}")
    print()


def test_wired_via_run_baseline() -> None:
    print("=" * 70)
    print("B. Wired through the REAL, unmodified run_baseline()")
    print("=" * 70)

    def fake_run_binary_unrecognized(binary, args, timeout=120):
        return 0, UNRECOGNIZED_STDOUT, "", 0.01

    ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    original_run_binary = pipeline_module.run_binary
    pipeline_module.run_binary = fake_run_binary_unrecognized
    try:
        job = JobState(seed_id=SeedId.VECTOR_ADD)
        final = run_baseline(job, ws, SEEDS_ROOT)
    finally:
        pipeline_module.run_binary = original_run_binary

    print(f"Job ID: {final.job_id}")
    print(f"Status: {final.status}")
    print()

    check("job still COMPLETED (unrecognized output is not a failure)", str(final.status) == "JobStatus.COMPLETED", f"status={final.status}")
    check(
        "job.unrecognized_output_snippet captured the real text verbatim",
        final.unrecognized_output_snippet == UNRECOGNIZED_STDOUT.strip(),
        f"got {final.unrecognized_output_snippet!r}",
    )
    check("kernel_time_ms correctly stayed None (no 'Kernel time:' line in this text)", final.metrics.derived.kernel_time_ms is None)

    msg_found = any(
        "matched none of this pipeline's known metric/validation formats" in m.content for m in final.messages
    )
    check("a message announces the unrecognized format", msg_found)

    report_path = Path(final.report_md_path) if final.report_md_path else None
    report_text = report_path.read_text(encoding="utf-8") if report_path and report_path.exists() else ""
    check("report includes the 'Raw Binary Output (unrecognized format)' section", "## Raw Binary Output (unrecognized format)" in report_text)
    check("report shows the actual unrecognized text verbatim", "foo=42, bar=99, status=OK" in report_text)
    print()


def test_truncation() -> None:
    print("=" * 70)
    print("C. Long unrecognized output is truncated with an honest marker")
    print("=" * 70)

    long_stdout = "line of unrecognized debug output\n" * 200  # well over 2000 bytes

    def fake_run_binary_long(binary, args, timeout=120):
        return 0, long_stdout, "", 0.01

    ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    original_run_binary = pipeline_module.run_binary
    pipeline_module.run_binary = fake_run_binary_long
    try:
        job = JobState(seed_id=SeedId.TILED_MATMUL)
        final = run_baseline(job, ws, SEEDS_ROOT)
    finally:
        pipeline_module.run_binary = original_run_binary

    snippet = final.unrecognized_output_snippet or ""
    check("snippet is truncated, not the full multi-KB text", len(snippet) < len(long_stdout.strip()), f"snippet len={len(snippet)}, full len={len(long_stdout.strip())}")
    check("truncation is disclosed honestly, not silent", "truncated" in snippet and "logs/run.log" in snippet)
    print()


def test_regression_original_seeds_unaffected() -> None:
    print("=" * 70)
    print("REGRESSION CHECK — the 4 original seeds + multiFileDemo are untouched")
    print("=" * 70)
    ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    for seed in (SeedId.VECTOR_ADD, SeedId.TILED_MATMUL, SeedId.REDUCTION, SeedId.REPAIR_DEMO, SeedId.MULTI_FILE_DEMO):
        job = JobState(seed_id=seed)
        final = run_baseline(job, ws, SEEDS_ROOT)
        check(f"{seed.value}: unrecognized_output_snippet is None (real MOCK output always recognized)", final.unrecognized_output_snippet is None, f"got {final.unrecognized_output_snippet!r}")
    print()


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — Unrecognized Metrics Format Fallback Test")
    print("=" * 70)
    print(f"SATOSWARM_MOCK: {'1 (mock)' if MOCK else '0/unset (REAL — this test expects mock)'}")
    if not MOCK:
        print("WARNING: not running in mock mode. Set SATOSWARM_MOCK=1 and re-run.")
    print()

    test_parser_unit()
    test_wired_via_run_baseline()
    test_truncation()
    test_regression_original_seeds_unaffected()

    print("=" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
