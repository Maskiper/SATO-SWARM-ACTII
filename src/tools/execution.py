"""ROCm tool wrappers: hipify, hipcc, amd-smi, and binary execution.

Thin subprocess wrappers around the real ROCm toolchain, with a single
environment-variable switch (SATOSWARM_MOCK) for local development without
AMD hardware. See the MOCK flag below for exact semantics — this is the
ONLY place SATOSWARM_MOCK is read; every other module imports the MOCK
constant from here rather than re-reading the environment variable, so
there is exactly one source of truth for which mode is active.

All real work must run on an actual ROCm-capable AMD GPU (e.g. MI300X).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from src.models.job import RawMetrics


# ---------------------------------------------------------------------------
# THE MOCK/REAL SWITCH
#
#   SATOSWARM_MOCK=1     -> MOCK mode. No subprocess calls are made. Every
#                           function below returns simulated GPU-shaped
#                           data instead.
#   SATOSWARM_MOCK=0     -> REAL mode. hipify-clang / hipcc / amd-smi are
#                           actually invoked via subprocess on PATH.
#   unset                -> REAL mode. An unset/lost env var on the real
#                           pod must never silently produce fake "success"
#                           data — it tries real tools and fails loudly
#                           (a clean, logged failure, not a crash — see
#                           _run()'s exception handling below) if something
#                           is actually wrong. Mock requires explicit opt-in.
#
# To run for real: leave SATOSWARM_MOCK unset (or set it to 0) on a machine
# that has hipify-clang, hipcc, and amd-smi on PATH.
# To develop locally without AMD hardware: set SATOSWARM_MOCK=1.
# ---------------------------------------------------------------------------
MOCK = os.environ.get("SATOSWARM_MOCK", "0") == "1"


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 300) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr).

    In REAL mode, a missing binary (e.g. hipcc not on PATH) is reported as
    an ordinary failed-command result rather than raising — this keeps both
    local runs without ROCm installed, and a misconfigured pod, failing
    cleanly and visibly (surfaced in job.error / the report) instead of
    crashing with an unhandled Python exception.
    """
    if MOCK:
        # Never actually shell out on mock
        return 0, f"[MOCK] would run: {' '.join(cmd)}", ""

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Timeout after {timeout}s: {e}"
    except (FileNotFoundError, OSError) as e:
        return 127, "", f"Command not found or failed to launch: {cmd[0]!r}: {e}"


def is_rocm_available() -> bool:
    if MOCK:
        return True
    rc, _, _ = _run(["hipcc", "--version"])
    return rc == 0


def get_gpu_info() -> dict:
    """Best-effort GPU name + ROCm version from amd-smi or hipcc."""
    if MOCK:
        return {"name": "AMD GPU (mock)", "rocm": "6.x-mock", "arch": "gfx942"}

    rc, out, _ = _run(["amd-smi", "--version"])
    if rc == 0:
        rc2, out2, _ = _run(["amd-smi"])
        return {"raw": out.strip() + "\n" + out2.strip()}

    rc, out, _ = _run(["hipcc", "--version"])
    return {"raw": out.strip() if rc == 0 else "unknown"}


def run_hipify(source_dir: Path, out_dir: Path, job_id: str) -> tuple[bool, str, str]:
    """Run hipify-clang on the .cu files in source_dir.

    In MOCK mode, _run() never actually shells out, so no .hip.cpp files
    would otherwise exist afterward — but run_hipcc() downstream needs real
    files to discover and "compile". Writing placeholder .hip.cpp files
    here (mock mode only) keeps the mock pipeline internally consistent
    without run_hipcc() silently receiving an empty source list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cu_files = list(source_dir.glob("*.cu")) + list(source_dir.glob("*.cuh"))
    if not cu_files:
        return False, "", "No .cu files found to hipify"

    hipify_bin = "hipify-clang"
    cmd = [hipify_bin, "--cuda-path=/usr/local/cuda", "-o", str(out_dir)]
    cmd += [str(f) for f in cu_files]

    rc, stdout, stderr = _run(cmd, cwd=source_dir, timeout=120)
    success = rc == 0

    if success and MOCK:
        for f in cu_files:
            placeholder = out_dir / f"{f.stem}.hip.cpp"
            if not placeholder.exists():
                placeholder.write_text(f"// [MOCK] simulated hipify output for {f.name}\n", encoding="utf-8")

    log = f"$ {' '.join(cmd)}\n{stdout}\n{stderr}"
    return success, log, stderr


_detected_arch: Optional[str] = None
_arch_detection_attempted = False


def detect_gpu_arch() -> Optional[str]:
    """Auto-detect the real GPU's ROCm architecture target (e.g.
    "gfx1100", "gfx942") — never assume one.

    Tries, in order:
      1. rocm_agent_enumerator — purpose-built for exactly this, prints
         one gfx code per agent (gfx000 is the host/CPU placeholder and
         is skipped).
      2. rocminfo — falls back to scanning its full output for any
         "gfxNNNN"-shaped token (again skipping gfx000).

    Returns None if neither tool is present or neither found a usable gfx
    target — callers should then fall back to --offload-arch=native
    (supported by sufficiently recent ROCm compilers) rather than
    silently compiling for a specific architecture that may not match the
    actual hardware. A mismatch here is not a compile-time error — it
    produces a binary that segfaults at launch (wrong ISA for the GPU
    actually present), which is exactly what motivated this function.

    Result is cached after the first call (the GPU doesn't change
    mid-process). In MOCK mode, no subprocess is attempted at all — returns
    "gfx942" unconditionally (matching the calibration of _mock_seed_output's
    canned achieved-bandwidth numbers below, so mock mode's efficiency-%
    computation is still demoable and its code path still gets exercised
    by routine mock testing). This is purely cosmetic: it never touches
    real compilation, and every report/message already carries an
    unmistakable separate "Mode: MOCK"/"(SIMULATED)" label anywhere this
    value is shown — see _fmt() and generate_minimal_report().
    """
    global _detected_arch, _arch_detection_attempted
    if _arch_detection_attempted:
        return _detected_arch
    _arch_detection_attempted = True

    if MOCK:
        _detected_arch = "gfx942"
        return _detected_arch

    rc, out, _ = _run(["rocm_agent_enumerator"], timeout=15)
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if re.fullmatch(r"gfx[0-9a-fA-F]+", line) and line != "gfx000":
                _detected_arch = line
                return _detected_arch

    rc, out, _ = _run(["rocminfo"], timeout=15)
    if rc == 0:
        for m in re.finditer(r"\bgfx[0-9a-fA-F]+\b", out):
            if m.group(0) != "gfx000":
                _detected_arch = m.group(0)
                return _detected_arch

    _detected_arch = None
    return None


def run_hipcc(hip_sources: list[Path], out_binary: Path, arch: Optional[str] = None) -> tuple[bool, str, str, str]:
    """Compile with hipcc, targeting the GPU actually present.

    If `arch` isn't given explicitly, auto-detects it via
    detect_gpu_arch(). If detection itself finds nothing, falls back to
    --offload-arch=native (ROCm 5.7+/6.x compilers auto-detect the build
    machine's GPU at compile time) rather than assuming any specific
    architecture — compiling for the wrong one doesn't fail the build, it
    produces a binary that segfaults on launch.

    Returns (success, log, stderr, arch_used) — arch_used is whatever was
    actually passed to --offload-arch=, so the caller can log it and
    record it on the job (never silently lost).
    """
    if not hip_sources:
        return False, "", "No HIP sources to compile", "n/a (compilation never attempted)"

    if arch is None:
        arch = detect_gpu_arch()
        if arch is None:
            arch = "native"

    cmd = ["hipcc", "-O3", f"--offload-arch={arch}"]
    cmd += [str(p) for p in hip_sources]
    cmd += ["-o", str(out_binary)]

    rc, stdout, stderr = _run(cmd, timeout=180)
    success = rc == 0
    log = f"$ {' '.join(cmd)}\n{stdout}\n{stderr}"
    return success, log, stderr, arch


def _try_float(value: Any) -> Optional[float]:
    """Pull a float out of a raw number or a unit-suffixed string like
    '45 %' / '300.5 W' / '62 C' (amd-smi often embeds units in strings).
    Returns None (never a guess) if nothing numeric is found.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+", value)
        if m:
            return float(m.group(0))
    return None


def _dig(node: Any, *path: str) -> Any:
    """Walk a nested dict (unwrapping a leading list, since amd-smi --json
    often returns a top-level list of per-GPU objects) by key path.
    Returns None on any miss instead of raising.
    """
    for key in path:
        if isinstance(node, list):
            if not node:
                return None
            node = node[0]
        if isinstance(node, dict):
            node = node.get(key)
        else:
            return None
    return node


def _parse_amd_smi_json(raw_text: str) -> RawMetrics:
    """Best-effort parse of `amd-smi metric --json` output into RawMetrics.

    IMPORTANT: amd-smi's JSON schema varies across ROCm releases, and this
    parser was written without a real MI300X to validate the exact key
    names against. It tries several plausible key paths per field; any
    field it can't confidently find is left as None (RawMetrics fields are
    all Optional — missing means "not captured", never a guessed number).

    The raw text is always saved to logs/ regardless of what this parses
    (see pipeline.py), so nothing is lost even if every path below misses.

    On Day 0: run `amd-smi metric --json` on the real pod, compare its
    actual structure to the paths tried below, and add/fix key names here
    if the parsed RawMetrics comes back empty.
    """
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return RawMetrics()

    util = (
        _try_float(_dig(data, "usage", "gfx_activity"))
        or _try_float(_dig(data, "usage", "gpu_activity"))
    )
    power = (
        _try_float(_dig(data, "power", "socket_power"))
        or _try_float(_dig(data, "power", "average_socket_power"))
    )
    temp = (
        _try_float(_dig(data, "temperature", "edge"))
        or _try_float(_dig(data, "temperature", "junction"))
    )
    mem = (
        _try_float(_dig(data, "mem_usage", "used"))
        or _try_float(_dig(data, "vram", "used"))
    )
    sclk = (
        _try_float(_dig(data, "clock", "sclk"))
        or _try_float(_dig(data, "clock", "gfx_clk"))
    )
    mclk = (
        _try_float(_dig(data, "clock", "mclk"))
        or _try_float(_dig(data, "clock", "mem_clk"))
    )

    return RawMetrics(
        gpu_utilization_percent=util,
        power_watts_avg=power,
        # A single snapshot has no real "peak" (that needs continuous
        # sampling during the kernel, not yet implemented) — reusing the
        # one real reading we have for both fields, rather than inventing
        # a second, different, fabricated number.
        power_watts_peak=power,
        temperature_c=temp,
        memory_used_mb=mem,
        clock_sclk_mhz=sclk,
        clock_mclk_mhz=mclk,
    )


def capture_amd_smi_snapshot() -> tuple[RawMetrics, str]:
    """Capture a single amd-smi snapshot. Returns (parsed_metrics, raw_text)
    so the caller can always persist the raw text to logs/, even when
    parsing misses a field (or misses everything) — see _parse_amd_smi_json.
    """
    if MOCK:
        raw = (
            "[MOCK] amd-smi metric --json (simulated — not real hardware output)\n"
            '{"usage": {"gfx_activity": "92 %"}, "power": {"socket_power": "680 W"}, '
            '"temperature": {"edge": "67 C"}, "mem_usage": {"used": "42000"}, '
            '"clock": {"sclk": "2100", "mclk": "5200"}}'
        )
        return (
            RawMetrics(
                gpu_utilization_percent=92.0,
                power_watts_avg=680.0,
                power_watts_peak=710.0,
                temperature_c=67.0,
                memory_used_mb=42000.0,
                clock_sclk_mhz=2100.0,
                clock_mclk_mhz=5200.0,
            ),
            raw,
        )

    rc, out, err = _run(["amd-smi", "metric", "--json"], timeout=10)
    if rc != 0 or not out.strip():
        # Fallback to plain amd-smi (not JSON — parsing will likely miss,
        # but we still keep the raw text for a human to read from logs/)
        rc, out, err = _run(["amd-smi"], timeout=10)

    raw_text = out if out.strip() else (err or f"amd-smi returned rc={rc} with no output")
    metrics = _parse_amd_smi_json(raw_text) if out.strip() else RawMetrics()
    return metrics, raw_text


def run_binary(binary: Path, args: list[str], timeout: int = 120) -> tuple[int, str, str, float]:
    """Execute the compiled HIP binary and return (rc, stdout, stderr, wall_time_s)."""
    start = time.time()

    if MOCK:
        name = binary.name.lower()
        if "vectoradd" in name or "vector_add" in name:
            stdout = _mock_seed_output("vectorAdd")
        elif "matmul" in name or "tiled" in name:
            stdout = _mock_seed_output("tiledMatmul")
        elif "reduction" in name or "reduce" in name:
            stdout = _mock_seed_output("reduction")
        else:
            stdout = "seed completed successfully.\nKernel time: 1.23 ms\n"
        rc, stderr = 0, ""
        wall = 0.0123
    else:
        rc, stdout, stderr = _run([str(binary)] + args, timeout=timeout)
        wall = time.time() - start

    return rc, stdout, stderr, wall


def _mock_seed_output(seed_name: str) -> str:
    """Simulated stdout for mock-mode runs — matches the exact wording the
    real seed binaries print (see seeds/*.cu) so mock and real output are
    only ever distinguishable by the numbers, never the format.
    """
    if seed_name == "vectorAdd":
        return (
            "SATO SWARM vectorAdd seed\n"
            "Elements: 268435456 (1024.00 MB per vector)\n"
            "Total data moved (read+write): 3.00 GB\n"
            "Result check: c[0]=0.1800 (exp 0.1800), c[n-1]=2.5100 (exp 2.5100)\n"
            "\n=== vectorAdd Timing ===\n"
            "Kernel time: 0.652 ms\n"
            "Achieved bandwidth: 4601.23 GB/s\n"
            "Theoretical MI300X HBM3 peak (reference): 5300 GB/s\n"
            "Efficiency (approx): 86.8%\n"
            "vectorAdd seed completed successfully.\n"
        )
    elif seed_name == "tiledMatmul":
        return (
            "SATO SWARM tiledMatmul seed\n"
            "Matrix size: 1024x1024 (4.00 MB per matrix)\n"
            "FLOPs (2*M*N*K): 2.15 GFLOPs\n"
            "Sanity C[0,0] = 17.8500 (expected ~17.8500)\n"
            "\n=== Tiled Matmul Timing ===\n"
            "Kernel time: 1.874 ms\n"
            "Achieved: 1.15 TFLOPS\n"
            "Theoretical FP32 peak reference (MI300X): ~163 TFLOPS\n"
            "Efficiency (FP32 approx): 0.7%\n"
            "tiledMatmul seed completed successfully.\n"
        )
    elif seed_name == "reduction":
        return (
            "SATO SWARM reduction seed\n"
            "Elements: 268435456\n"
            "Reduction result: 268435456 (expected 268435456)\n"
            "Kernel time: 0.418 ms\n"
            "Effective read BW: 2568.76 GB/s\n"
            "reduction seed completed.\n"
        )
    return "seed completed successfully.\n"


def parse_binary_output_for_metrics(stdout: str) -> dict:
    """Extract RAW real numbers the seed binary printed to its own stdout.

    This is a fact extractor ONLY — it does not compute any derived
    efficiency number itself. "kernel_time_ms" is the seed's own
    hipEventElapsedTime() measurement (real GPU-side timing, wrapped
    directly around the kernel launch inside the seed — see seeds/*.cu),
    printed by the actual binary. Every other value here is either present
    because the regex matched real printed text, or simply absent from the
    returned dict — there is no default/fallback value substituted for a
    missing key anywhere in this function.

    src/baseline/pipeline.py's _compute_derived_metrics() turns these raw
    facts into achieved_bw_gbs / achieved_tflops, computed directly from
    kernel_time_ms + a real byte/FLOP count parsed here — never from a
    constant, and never from a pre-summarized "Achieved ..." line taken on
    faith (those are also captured below, suffixed "_selfreported", purely
    as a cross-check against the independently Python-computed value).
    """
    data: dict = {}
    if m := re.search(r"Kernel time:\s*([\d.]+)\s*ms", stdout):
        data["kernel_time_ms"] = float(m.group(1))
    if m := re.search(r"Total data moved[^:]*:\s*([\d.]+)\s*GB", stdout):
        data["total_data_moved_gb"] = float(m.group(1))
    if m := re.search(r"FLOPs[^:]*:\s*([\d.]+)\s*GFLOPs", stdout):
        data["gflops"] = float(m.group(1))
    if m := re.search(r"Elements:\s*(\d+)", stdout):
        data["elements"] = int(m.group(1))
    if m := re.search(r"Reduction result:\s*([\d.]+)", stdout):
        data["reduction_result"] = float(m.group(1))
    # Correctness self-check numbers (actual vs. expected), printed by the
    # seed itself — used to compute a REAL max_abs_diff, never a hardcoded
    # "0.0 means it passed" placeholder.
    if m := re.search(r"c\[0\]=([\d.]+)\s*\(exp\s*([\d.]+)\).*?c\[n-1\]=([\d.]+)\s*\(exp\s*([\d.]+)\)", stdout):
        data["check_pairs"] = [(float(m.group(1)), float(m.group(2))), (float(m.group(3)), float(m.group(4)))]
    elif m := re.search(r"Sanity C\[0,0\]\s*=\s*([\d.]+)\s*\(expected\s*~?([\d.]+)\)", stdout):
        data["check_pairs"] = [(float(m.group(1)), float(m.group(2)))]
    elif m := re.search(r"Reduction result:\s*([\d.]+)\s*\(expected\s*([\d.]+)\)", stdout):
        data["check_pairs"] = [(float(m.group(1)), float(m.group(2)))]
    # Cross-check only — never used as the report's headline number:
    if m := re.search(r"Achieved bandwidth:\s*([\d.]+)\s*GB/s", stdout):
        data["achieved_bw_gbs_selfreported"] = float(m.group(1))
    if m := re.search(r"Achieved:\s*([\d.]+)\s*TFLOPS", stdout):
        data["achieved_tflops_selfreported"] = float(m.group(1))
    if m := re.search(r"Effective read BW:\s*([\d.]+)\s*GB/s", stdout):
        data["effective_read_bw_gbs_selfreported"] = float(m.group(1))
    return data
