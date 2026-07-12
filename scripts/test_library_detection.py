#!/usr/bin/env python3
"""
SATO SWARM — CUDA Library Detection Test

Exercises src/tools/execution.py's detect_cuda_library_includes() /
check_rocm_library_available() standalone (synthetic source snippets,
no pipeline needed), then end-to-end through the REAL, unmodified
run_baseline() (a synthetic multi-library seed built in a temp seeds
directory), and finally regression-checks that the 4 original seeds +
multiFileDemo are completely untouched by any of this — none of them
reference cublas_v2.h/cudnn.h/thrust/*/nccl.h, so job.library_detections
must stay empty and job.hipify_command/hipcc_command must show no --roc
or extra -l flag for any of them.

Usage:
    SATOSWARM_MOCK=1 python scripts/test_library_detection.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.pipeline import run_baseline
from src.models.job import JobState, SeedId
from src.tools.execution import MOCK, check_rocm_library_available, detect_cuda_library_includes
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


def _write(dir_: Path, name: str, content: str) -> Path:
    p = dir_ / name
    p.write_text(content, encoding="utf-8")
    return p


def test_detection_unit(tmp: Path) -> None:
    print("=" * 70)
    print("1. detect_cuda_library_includes() — synthetic source snippets")
    print("=" * 70)

    # --- no CUDA library headers at all ---
    d1 = tmp / "none"
    d1.mkdir()
    _write(d1, "plain.cu", '#include <cuda_runtime.h>\n__global__ void k() {}\n')
    r = detect_cuda_library_includes(d1)
    check("plain cuda_runtime.h-only source detects nothing", r == [], f"got {r}")

    # --- cublas_v2.h -> rocBLAS ---
    d2 = tmp / "cublas"
    d2.mkdir()
    _write(d2, "a.cu", '#include <cublas_v2.h>\nint main(){return 0;}\n')
    r = detect_cuda_library_includes(d2)
    check("cublas_v2.h detected", len(r) == 1 and r[0]["cuda_header"] == "cublas_v2.h")
    if r:
        check("cublas_v2.h maps to rocBLAS", r[0]["rocm_name"] == "rocBLAS")
        check("cublas_v2.h maps to rocblas.h", r[0]["rocm_header"] == "rocblas.h")
        check("cublas_v2.h gets -lrocblas", r[0]["link_flag"] == "-lrocblas")
        check("cublas_v2.h has no caveat", r[0]["caveat"] is None)
        check("found_in lists a.cu", r[0]["found_in"] == ["a.cu"])

    # --- cudnn.h -> MIOpen (with caveat) ---
    d3 = tmp / "cudnn"
    d3.mkdir()
    _write(d3, "b.cu", '#include <cudnn.h>\nint main(){return 0;}\n')
    r = detect_cuda_library_includes(d3)
    check("cudnn.h detected", len(r) == 1 and r[0]["cuda_header"] == "cudnn.h")
    if r:
        check("cudnn.h maps to MIOpen", r[0]["rocm_name"] == "MIOpen")
        check("cudnn.h maps to miopen/miopen.h", r[0]["rocm_header"] == "miopen/miopen.h")
        check("cudnn.h gets -lMIOpen", r[0]["link_flag"] == "-lMIOpen")
        check("cudnn.h HAS a partial-translation caveat", bool(r[0]["caveat"]) and "PARTIAL" in r[0]["caveat"])

    # --- thrust/* -> rocThrust, no rewrite, no link flag, dedup across 2 headers + 2 files ---
    d4 = tmp / "thrust"
    d4.mkdir()
    _write(d4, "c.cu", '#include <thrust/device_vector.h>\nint main(){return 0;}\n')
    _write(d4, "d.cu", '#include <thrust/sort.h>\nvoid f(){}\n')
    r = detect_cuda_library_includes(d4)
    check("thrust/* detected exactly once (not once per header)", len(r) == 1, f"got {r}")
    if r:
        check("thrust/* maps to rocThrust", r[0]["rocm_name"] == "rocThrust")
        check("thrust/* needs no header rewrite", r[0]["rocm_header"] is None)
        check("thrust/* needs no link flag (header-only)", r[0]["link_flag"] is None)
        check(
            "both distinct thrust headers recorded",
            r[0]["matched_headers"] == ["thrust/device_vector.h", "thrust/sort.h"],
            f"got {r[0]['matched_headers']}",
        )
        check("found_in lists both files", r[0]["found_in"] == ["c.cu", "d.cu"], f"got {r[0]['found_in']}")

    # --- nccl.h -> RCCL ---
    d5 = tmp / "nccl"
    d5.mkdir()
    _write(d5, "e.cu", '#include <nccl.h>\nint main(){return 0;}\n')
    r = detect_cuda_library_includes(d5)
    check("nccl.h detected", len(r) == 1 and r[0]["cuda_header"] == "nccl.h")
    if r:
        check("nccl.h maps to RCCL", r[0]["rocm_name"] == "RCCL")
        check("nccl.h maps to rccl/rccl.h", r[0]["rocm_header"] == "rccl/rccl.h")
        check("nccl.h gets -lrccl", r[0]["link_flag"] == "-lrccl")

    # --- multiple distinct libraries in one file ---
    d6 = tmp / "multi"
    d6.mkdir()
    _write(d6, "f.cu", '#include <cublas_v2.h>\n#include <nccl.h>\nint main(){return 0;}\n')
    r = detect_cuda_library_includes(d6)
    check(
        "two distinct libraries in one file both detected",
        {e["cuda_header"] for e in r} == {"cublas_v2.h", "nccl.h"},
        f"got {[e['cuda_header'] for e in r]}",
    )
    print()


def test_availability_check_mock() -> None:
    print("=" * 70)
    print("2. check_rocm_library_available() in MOCK mode")
    print("=" * 70)
    fake_linked = {"header_only": False, "so_glob": "librocblas.so*"}
    fake_header_only = {"header_only": True, "header_check_subdir": "thrust"}
    for label, entry in [("linked library entry", fake_linked), ("header-only entry", fake_header_only)]:
        available, detail = check_rocm_library_available(entry)
        check(
            f"{label}: MOCK mode never claims 'available' (no real filesystem query)",
            available is False and "MOCK mode" in detail,
            f"got ({available!r}, {detail!r})",
        )
    print()


def _build_synthetic_multi_library_seed(seeds_root: Path) -> None:
    """A .cu file that references cublas_v2.h AND thrust/*, mirroring how
    a real-world CUDA project would #include both a compute library and
    a container/algorithm library side by side. Not meant to actually
    compile against real cuBLAS/Thrust (no cuBLAS calls are made) — this
    only exercises detection + the hipify/hipcc flag wiring end-to-end
    through the real run_baseline(), the same way seeds/repairDemo.cu
    exercises one specific real gap without being a full application.

    Deliberately named vectorAdd.cu and paired with SeedId.VECTOR_ADD (a
    REAL enum member — see SeedId in src/models/job.py) rather than
    inventing a 6th, permanent SeedId just for this one test:
    run_baseline() takes seeds_root as a parameter precisely so a test
    can point it at a throwaway directory (scripts/test_repair_loop.py
    and scripts/test_main.py already do the same) — pairing a real
    SeedId with THIS temp seeds_root's own vectorAdd.cu (not the real
    seeds/vectorAdd.cu) is a normal, supported use of that parameter, not
    a workaround.
    """
    content = (
        "// Synthetic fixture for scripts/test_library_detection.py — NOT the real seeds/vectorAdd.cu.\n"
        "#include <stdio.h>\n"
        "#include <cuda_runtime.h>\n"
        "#include <cublas_v2.h>\n"
        "#include <thrust/device_vector.h>\n"
        "__global__ void noop() {}\n"
        "int main(void) {\n"
        "  noop<<<1,1>>>();\n"
        "  printf(\"libraryDetectionDemo seed completed successfully.\\n\");\n"
        "  return 0;\n"
        "}\n"
    )
    _write(seeds_root, "vectorAdd.cu", content)


def test_end_to_end_via_run_baseline() -> None:
    print("=" * 70)
    print("3. Wired through the REAL, unmodified run_baseline()")
    print("=" * 70)

    tmp_seeds = Path(tempfile.mkdtemp(prefix="sato_lib_detect_seeds_"))
    tmp_jobs = Path(tempfile.mkdtemp(prefix="sato_lib_detect_jobs_"))
    try:
        _build_synthetic_multi_library_seed(tmp_seeds)
        job = JobState(seed_id=SeedId.VECTOR_ADD)
        ws = WorkspaceManager(base_dir=tmp_jobs)
        final = run_baseline(job, ws, tmp_seeds)

        check("job completed (compile succeeded in mock mode)", str(final.status) == "JobStatus.COMPLETED", f"status={final.status}")
        check(
            "exactly 2 libraries detected (rocBLAS + rocThrust)",
            {d["rocm_name"] for d in final.library_detections} == {"rocBLAS", "rocThrust"},
            f"got {[d['rocm_name'] for d in final.library_detections]}",
        )
        check("hipify_command requests --roc", "--roc" in (final.hipify_command or ""), final.hipify_command)
        check("hipcc_command includes -lrocblas", "-lrocblas" in (final.hipcc_command or ""), final.hipcc_command)
        check(
            "hipcc_command does NOT include a thrust link flag (header-only)",
            "-lrocthrust" not in (final.hipcc_command or "") and "-lthrust" not in (final.hipcc_command or ""),
            final.hipcc_command,
        )
        detected_msg = any(
            "CUDA library reference(s) detected" in m.content for m in final.messages
        )
        check("a message announces the detected libraries", detected_msg)

        report_path = Path(final.report_md_path) if final.report_md_path else None
        report_text = report_path.read_text(encoding="utf-8") if report_path and report_path.exists() else ""
        check("report includes the CUDA Library Dependencies section", "## CUDA Library Dependencies" in report_text)
        check("report table lists rocBLAS", "rocBLAS" in report_text)
        check("report shows the MIOpen-style honest availability check wording", "Installed on this machine" in report_text)

        hip_out = Path(final.workspace_dir) / "hip_out"
        check(
            "hip_out/ has exactly the one hipified translation unit (no header to exclude here)",
            sorted(p.name for p in hip_out.glob("*.hip.cpp")) == ["vectorAdd.hip.cpp"],
            f"got {sorted(p.name for p in hip_out.iterdir())}",
        )
    finally:
        shutil.rmtree(tmp_seeds, ignore_errors=True)
        shutil.rmtree(tmp_jobs, ignore_errors=True)
    print()


def test_regression_original_seeds_unaffected() -> None:
    print("=" * 70)
    print("REGRESSION CHECK — the 4 original seeds + multiFileDemo are untouched")
    print("=" * 70)
    ws = WorkspaceManager(base_dir=REPO_ROOT / "jobs")
    for seed in (SeedId.VECTOR_ADD, SeedId.TILED_MATMUL, SeedId.REDUCTION, SeedId.REPAIR_DEMO, SeedId.MULTI_FILE_DEMO):
        job = JobState(seed_id=seed)
        final = run_baseline(job, ws, SEEDS_ROOT)
        check(f"{seed.value}: library_detections is empty", final.library_detections == [], f"got {final.library_detections}")
        check(f"{seed.value}: hipify_command has no --roc", "--roc" not in (final.hipify_command or ""), final.hipify_command)
        check(
            f"{seed.value}: hipcc_command has no library link flag",
            not any(flag in (final.hipcc_command or "") for flag in ("-lrocblas", "-lMIOpen", "-lrccl")),
            final.hipcc_command,
        )
    print()


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — CUDA Library Detection Test")
    print("=" * 70)
    print(f"SATOSWARM_MOCK: {'1 (mock)' if MOCK else '0/unset (REAL — this test expects mock)'}")
    if not MOCK:
        print("WARNING: not running in mock mode. Set SATOSWARM_MOCK=1 and re-run.")
    print()

    with tempfile.TemporaryDirectory(prefix="sato_lib_detect_unit_") as tmp_str:
        test_detection_unit(Path(tmp_str))
    test_availability_check_mock()
    test_end_to_end_via_run_baseline()
    test_regression_original_seeds_unaffected()

    print("=" * 70)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
