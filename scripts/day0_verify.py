#!/usr/bin/env python3
"""Day 0 Hardware Verification helper for SATO SWARM.

Run this on the actual AMD GPU instance (before/instead of the full
pipeline) to confirm the ROCm toolchain is actually present and working.
Works on any ROCm-capable AMD GPU, not just MI300X — see also
scripts/preflight.sh, which additionally auto-detects the real GPU
architecture and compiles/runs a real HIP kernel end to end.
This script always tries the real tools — it has no mock mode, since its
entire purpose is a real-hardware sanity check.

Usage (on the instance):
    python scripts/day0_verify.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def shutil_which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def main() -> None:
    print("=" * 70)
    print("SATO SWARM — Day 0 Hardware Verification")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working dir: {Path.cwd()}")
    print(f"SATOSWARM_MOCK: {os.environ.get('SATOSWARM_MOCK', '(unset -> real mode)')}")
    print()

    # 1. ROCm visibility
    print("1. amd-smi / rocm-smi")
    rc, out, err = run(["amd-smi", "--version"] if shutil_which("amd-smi") else ["rocm-smi", "--version"])
    print("   version:", out or err)
    rc, out, err = run(["amd-smi"] if shutil_which("amd-smi") else ["rocm-smi", "-a"])
    print("   output (first 20 lines):")
    for line in (out or err).splitlines()[:20]:
        print("   ", line)
    print()

    # 2. hipify (hipify-perl preferred -- needs no CUDA SDK) + hipcc
    print("2. hipify & hipcc")
    for tool in (["hipify-perl", "--help"], ["hipify-clang", "--help"], ["hipcc", "--version"]):
        rc, out, err = run(tool)
        print(f"   {' '.join(tool[:1])}: rc={rc}")
        if out:
            print("   ", out.splitlines()[0][:100])
    print("   NOTE: hipify-perl is the preferred tool (text-based, no CUDA SDK required).")
    print("   hipify-clang needs a real CUDA install (cuda_runtime.h) to parse sources --")
    print("   it will fail with 'cannot find CUDA installation' on an AMD-only box.")
    print()

    # 3. Torch + HIP (optional — only relevant if you're also using PyTorch)
    print("3. PyTorch ROCm (optional)")
    code = """
import torch
print('PyTorch:', torch.__version__)
print('HIP:', torch.version.hip)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device 0:', torch.cuda.get_device_name(0))
    print('Count:', torch.cuda.device_count())
"""
    rc, out, err = run([sys.executable, "-c", code])
    print(out or err)
    print()

    # 4. One manual seed (vectorAdd) — best effort
    print("4. Manual vectorAdd seed (quick smoke)")
    seeds = Path("seeds/vectorAdd.cu")
    if not seeds.exists():
        print("   seeds/vectorAdd.cu not found in this checkout. Copy it here first.")
    else:
        print("   Found seed. To run full manual port + metrics capture:")
        print("     mkdir -p /tmp/day0 && cp seeds/vectorAdd.cu /tmp/day0/")
        print("     cd /tmp/day0")
        print("     hipify-perl vectorAdd.cu > vectorAdd.hip.cpp   # preferred: no CUDA SDK needed")
        print("     rocminfo | grep -o 'gfx[0-9a-fA-F]*' | grep -v gfx000   # find your real arch first")
        print("     hipcc -O3 --offload-arch=<arch from above, or 'native'> -o vectorAdd_hip *.hip.cpp")
        print("     amd-smi metric --json > before.json")
        print("     ./vectorAdd_hip")
        print("     amd-smi metric --json > during.json")
        print("   Compare before.json / during.json against src/tools/execution.py's")
        print("   _parse_amd_smi_json() field-path guesses — adjust the key paths there")
        print("   if the parsed RawMetrics in a real pipeline run come back empty.")
    print()

    print("=" * 70)
    print("Next: python scripts/test_baseline.py vectorAdd   (real mode — SATOSWARM_MOCK unset)")
    print("=" * 70)


if __name__ == "__main__":
    main()
