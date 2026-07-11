#!/usr/bin/env python3
"""
SATO SWARM — PortingMemory Standalone Test

Confirms src/memory/loader.py's PortingMemory loads the seeded
memory/porting_patterns.jsonl correctly, that get_relevant_patterns()
actually retrieves the right pattern for a realistic hipcc-style error
message, that it does NOT false-positive on an unrelated error, and that
add_pattern() persists correctly. Does not touch src/baseline/pipeline.py
or run any repair loop — none exists yet.

Usage:
    python scripts/test_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.loader import PortingMemory

REPO_ROOT = Path(__file__).resolve().parents[1]

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


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — PortingMemory Standalone Test")
    print("=" * 70)

    patterns_path = REPO_ROOT / "memory" / "porting_patterns.jsonl"
    print(f"Loading: {patterns_path}")
    memory = PortingMemory(path=patterns_path)
    print(f"Loaded {len(memory)} patterns")
    print()

    check("loaded exactly the 4 seeded patterns", len(memory) == 4, f"got {len(memory)}")
    print()

    # --- realistic hipcc-style error quoting the verified repairDemo gap ---
    real_error = (
        "repairDemo.hip.cpp:101:16: error: use of undeclared identifier "
        "'cudaCtxResetPersistingL2Cache'\n"
        "  CHECK_CUDA(cudaCtxResetPersistingL2Cache());\n"
        "               ^\n"
        "1 error generated when compiling for gfx1100."
    )
    print("1. get_relevant_patterns() against a realistic hipcc error:")
    print(f"   {real_error.splitlines()[0]!r}")
    results = memory.get_relevant_patterns(real_error, top_k=5, min_confidence=0.65)
    print(f"   -> {len(results)} match(es): {[r['id'] for r in results]}")
    check("found at least one match", len(results) >= 1)
    check(
        "top match is the correct pattern (gap_cudaCtxResetPersistingL2Cache)",
        results and results[0]["id"] == "gap_cudaCtxResetPersistingL2Cache",
    )
    print()

    # --- unrelated error should NOT match anything ---
    unrelated_error = (
        "vectorAdd.hip.cpp:34:5: error: expected ';' after expression\n"
        "  int x = 5\n"
        "           ^\n"
        "1 error generated."
    )
    print("2. get_relevant_patterns() against an UNRELATED syntax error:")
    print(f"   {unrelated_error.splitlines()[0]!r}")
    results2 = memory.get_relevant_patterns(unrelated_error, top_k=5, min_confidence=0.65)
    print(f"   -> {len(results2)} match(es): {[r['id'] for r in results2]}")
    check("correctly finds zero matches (no false positive)", len(results2) == 0)
    print()

    # --- min_confidence actually gates results ---
    print("3. min_confidence=0.0 vs min_confidence=1.0 on the same real error:")
    loose = memory.get_relevant_patterns(real_error, top_k=5, min_confidence=0.0)
    strict = memory.get_relevant_patterns(real_error, top_k=5, min_confidence=1.0)
    print(f"   min_confidence=0.0 -> {len(loose)} matches")
    print(f"   min_confidence=1.0 -> {len(strict)} matches")
    check("min_confidence=0.0 returns at least as many as 1.0", len(loose) >= len(strict))
    print()

    # --- add_pattern: persist=False vs persist=True ---
    print("4. add_pattern()")
    before_count = len(memory)
    memory.add_pattern({"id": "test_staged_only", "cuda": "testOnlyStagedPattern"}, persist=False)
    check("persist=False adds to in-memory list", len(memory) == before_count + 1)

    reloaded_before_persist = PortingMemory(path=patterns_path)
    check(
        "persist=False did NOT touch the file on disk",
        len(reloaded_before_persist) == before_count,
        f"file has {len(reloaded_before_persist)}, expected {before_count}",
    )

    memory.add_pattern(
        {"id": "test_persisted_pattern", "cuda": "testPersistedPattern example marker"},
        persist=True,
    )
    reloaded_after_persist = PortingMemory(path=patterns_path)
    check(
        "persist=True actually appended a new line to the file",
        # only ONE add_pattern() call in this test used persist=True (the
        # persist=False one above never touched disk) -- so the file
        # should have exactly one more line than before_count.
        len(reloaded_after_persist) == before_count + 1,
        f"file has {len(reloaded_after_persist)}, expected {before_count + 1}",
    )
    print()

    # --- get_context_for_agent ---
    print("5. get_context_for_agent() (fresh reload, seeded patterns only):")
    fresh = PortingMemory(path=patterns_path)
    # Remove the test-only line we just persisted so the seed file stays clean
    lines = patterns_path.read_text(encoding="utf-8").splitlines()
    clean_lines = [l for l in lines if '"test_persisted_pattern"' not in l]
    patterns_path.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
    fresh = PortingMemory(path=patterns_path)
    context = fresh.get_context_for_agent()
    print(context)
    print()
    check("get_context_for_agent returns non-empty readable text", len(context) > 100)
    check("context mentions all 4 seeded pattern cuda fields", all(
        p["cuda"].split()[0] in context for p in [
            {"cuda": "cudaFuncGetName"}, {"cuda": "cudaGraphConditionalHandleCreate"},
            {"cuda": "cudaDeviceFlushGPUDirectRDMAWrites"}, {"cuda": "cudaCtxResetPersistingL2Cache"},
        ]
    ))
    check("cleaned up test-only persisted line, file back to 4 patterns", len(fresh) == 4, f"got {len(fresh)}")
    print()

    print("=" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
