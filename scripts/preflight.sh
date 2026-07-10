#!/usr/bin/env bash
# SATO SWARM — Pod Preflight Check
#
# Run this FIRST, before scripts/test_baseline.py, on the real pod. It does
# not touch the pipeline at all — it only answers one question: "does this
# environment actually have a working ROCm/HIP toolchain?"
#
# Architecture-agnostic: this script does NOT assume MI300X (gfx942). It
# auto-detects whatever GPU architecture is actually present (gfx942,
# gfx1100, whatever) via rocm_agent_enumerator / rocminfo, and uses that
# for both the visibility check and the hello-world compile below.
# Compiling for the wrong architecture doesn't fail the build — it produces
# a binary that SEGFAULTS AT LAUNCH on hardware whose ISA doesn't match
# what was compiled for. If you've hit that, this script is exactly the
# tool that would have caught it before you spent time on a real job run.
#
# Checks:
#   1. hipcc, hipify (hipify-perl preferred, hipify-clang fallback), amd-smi,
#      rocprofv3 are on PATH (prints versions)
#   2. rocminfo runs and reports a GPU; its actual gfx architecture is
#      auto-detected (any AMD GPU, not just gfx942)
#   3. A trivial HIP kernel actually compiles (for the detected
#      architecture) AND runs AND prints real device-side output — the
#      only check here that proves the whole chain (compiler -> GPU ->
#      back to host) actually works end to end, as opposed to just "the
#      binary exists on PATH".
#
# Usage:
#   bash scripts/preflight.sh
#
# Exit code: 0 if every check passed, 1 if any failed. Logs (rocminfo
# output, hello-world compile/run logs) are written to preflight_logs/
# for inspection or for saving as deployment evidence.

set -uo pipefail
# NOTE: deliberately NOT using `set -e` -- every check should run and be
# reported even if an earlier one fails, so you get the full picture in
# one pass instead of stopping at the first problem.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGDIR="$REPO_ROOT/preflight_logs"
mkdir -p "$LOGDIR"

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

pass() {
    local name="$1" detail="$2"
    RESULTS+=("PASS|$name|$detail")
    PASS_COUNT=$((PASS_COUNT + 1))
    printf "  \033[32m[PASS]\033[0m %-16s %s\n" "$name" "$detail"
}

fail() {
    local name="$1" detail="$2"
    RESULTS+=("FAIL|$name|$detail")
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf "  \033[31m[FAIL]\033[0m %-16s %s\n" "$name" "$detail"
}

get_version() {
    # Prints the first line of `<cmd> --version`, or empty string if the
    # tool isn't on PATH or the flag isn't supported.
    local cmd="$1"
    command -v "$cmd" &>/dev/null || return 1
    "$cmd" --version 2>&1 | head -n 1
    return 0
}

echo "======================================================================"
echo "SATO SWARM -- Pod Preflight Check (architecture auto-detected)"
echo "Host: $(hostname 2>/dev/null || echo unknown)   Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "Logs: $LOGDIR"
echo "======================================================================"
echo ""

# --------------------------------------------------------------------
# 1. Toolchain presence + versions
# --------------------------------------------------------------------
echo "1. Toolchain"

if ver=$(get_version hipcc); then
    pass "hipcc" "$ver"
else
    fail "hipcc" "not found on PATH"
fi

# hipify-perl is preferred: pure text/regex translation, needs no CUDA
# SDK at all -- the correct default on an AMD-only box, which has no
# cuda_runtime.h / libdevice for hipify-clang's clang-based parser to
# find. hipify-clang is only a fallback, and only useful if a real CUDA
# install is actually present alongside it -- flagged separately below if
# it's present without one, since it would fail at actual invocation time
# despite showing up on PATH.
if command -v hipify-perl &>/dev/null; then
    # hipify-perl's --version behavior isn't consistently documented
    # across ROCm releases -- if this prints something unexpected, that's
    # still a PASS (the tool exists); the raw text is in the log.
    ver=$(hipify-perl --version 2>&1 | head -n 1)
    echo "$ver" > "$LOGDIR/hipify_perl_version.log"
    pass "hipify-perl" "${ver:-present (preferred -- no CUDA SDK required)}"
elif command -v hipify-clang &>/dev/null; then
    ver=$(hipify-clang --version 2>&1 | head -n 1)
    echo "$ver" > "$LOGDIR/hipify_clang_version.log"
    if [ -d /usr/local/cuda/include ] || command -v nvcc &>/dev/null; then
        pass "hipify" "hipify-clang present with a CUDA SDK (fallback path -- hipify-perl not found). ${ver:-no version string}"
    else
        fail "hipify" "hipify-clang found, but no CUDA SDK detected (no nvcc, no /usr/local/cuda/include) -- it will fail with 'cannot find CUDA installation' when actually invoked. Install hipify-perl instead (needs no CUDA SDK) or install a CUDA toolkit alongside hipify-clang."
    fi
else
    fail "hipify" "neither hipify-perl nor hipify-clang found on PATH"
fi

if ver=$(get_version amd-smi); then
    pass "amd-smi" "$ver"
else
    fail "amd-smi" "not found on PATH"
fi

if command -v rocprofv3 &>/dev/null; then
    # Same caveat as hipify-perl/hipify-clang above: rocprofv3's --version behavior wasn't
    # verified against real hardware when this script was written -- PASS
    # is based on presence on PATH; check preflight_logs/ for what it
    # actually printed.
    ver=$(rocprofv3 --version 2>&1 | head -n 1)
    echo "$ver" > "$LOGDIR/rocprofv3_version.log"
    pass "rocprofv3" "${ver:-present, no version string}"
else
    fail "rocprofv3" "not found on PATH"
fi

echo ""

# --------------------------------------------------------------------
# 2. rocminfo -- confirm the GPU is actually visible to ROCm, and detect
#    its REAL architecture. Does not assume gfx942 -- any AMD GPU that
#    ROCm recognizes counts as a pass. DETECTED_ARCH is reused below for
#    the hello-world compile.
# --------------------------------------------------------------------
echo "2. GPU visibility + architecture detection"

DETECTED_ARCH=""
ROCMINFO_OUT=""

if command -v rocminfo &>/dev/null; then
    ROCMINFO_OUT="$(rocminfo 2>&1)"
    echo "$ROCMINFO_OUT" > "$LOGDIR/rocminfo.log"
fi

# Prefer rocm_agent_enumerator when available -- purpose-built for exactly
# this, one gfx code per agent (gfx000 = host/CPU placeholder, skip it).
if command -v rocm_agent_enumerator &>/dev/null; then
    DETECTED_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx[0-9a-fA-F]+$' | grep -v '^gfx000$' | head -n 1)
fi

# Fall back to parsing rocminfo's own output if rocm_agent_enumerator
# wasn't available or found nothing.
if [ -z "$DETECTED_ARCH" ] && [ -n "$ROCMINFO_OUT" ]; then
    DETECTED_ARCH=$(echo "$ROCMINFO_OUT" | grep -oE 'gfx[0-9a-fA-F]+' | grep -v '^gfx000$' | head -n 1)
fi

if command -v rocminfo &>/dev/null; then
    if [ -n "$DETECTED_ARCH" ]; then
        gpu_name=$(echo "$ROCMINFO_OUT" | grep -i "Marketing Name" | head -n 1 | sed 's/^[[:space:]]*//')
        pass "rocminfo" "GPU detected, architecture: $DETECTED_ARCH. ${gpu_name:-see preflight_logs/rocminfo.log}"
    else
        fail "rocminfo" "ran, but no gfx architecture found in its output -- see preflight_logs/rocminfo.log"
    fi
else
    fail "rocminfo" "not found on PATH"
fi

if [ -n "$DETECTED_ARCH" ]; then
    echo "  Will compile the hello-world check below for: $DETECTED_ARCH"
else
    echo "  Could not auto-detect a GPU architecture -- hello-world compile below will fall back to --offload-arch=native"
fi

echo ""

# --------------------------------------------------------------------
# 3. Trivial HIP hello-world -- compile AND run AND check real output.
#    This is the only check that proves the full chain works, not just
#    that the binaries exist.
# --------------------------------------------------------------------
echo "3. End-to-end HIP compile + run (hello world)"

HELLO_SRC="$LOGDIR/hip_hello.hip.cpp"
HELLO_BIN="$LOGDIR/hip_hello"

cat > "$HELLO_SRC" <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void hello() { printf("Hello from GPU thread %d\n", threadIdx.x); }
int main() { hello<<<1, 4>>>(); hipDeviceSynchronize(); return 0; }
EOF

# Use whatever was actually detected above -- NEVER a hardcoded arch.
# --offload-arch=native asks the compiler itself to auto-detect the build
# machine's GPU (supported on sufficiently recent ROCm compilers); it's
# the fallback only when rocm_agent_enumerator AND rocminfo both failed
# to identify anything.
HELLO_ARCH="${DETECTED_ARCH:-native}"
echo "  Compiling with --offload-arch=$HELLO_ARCH"

if command -v hipcc &>/dev/null; then
    if hipcc -O2 "--offload-arch=$HELLO_ARCH" -o "$HELLO_BIN" "$HELLO_SRC" > "$LOGDIR/hello_compile.log" 2>&1; then
        HELLO_OUT="$("$HELLO_BIN" 2>&1)"
        HELLO_RC=$?
        echo "$HELLO_OUT" > "$LOGDIR/hello_run.log"
        if [ "$HELLO_RC" -eq 0 ] && echo "$HELLO_OUT" | grep -q "Hello from GPU"; then
            n_lines=$(echo "$HELLO_OUT" | grep -c "Hello from GPU")
            pass "hip_hello_world" "compiled for $HELLO_ARCH, ran, printed from device ($n_lines/4 threads confirmed) -- see preflight_logs/hello_run.log"
        else
            fail "hip_hello_world" "compiled for $HELLO_ARCH but run failed or produced no device output (rc=$HELLO_RC) -- see preflight_logs/hello_run.log. A segfault here (rc=139) usually means an ISA mismatch: the binary was compiled for a different architecture than the GPU actually present. Compare $HELLO_ARCH above against what rocminfo/rocm_agent_enumerator actually reported."
        fi
    else
        fail "hip_hello_world" "compilation for $HELLO_ARCH failed -- see preflight_logs/hello_compile.log"
    fi
else
    fail "hip_hello_world" "skipped -- hipcc not found"
fi

echo ""

# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------
echo "======================================================================"
echo "PREFLIGHT SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "======================================================================"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r status name detail <<< "$r"
    if [ "$status" = "PASS" ]; then
        printf "  \033[32m[PASS]\033[0m %-16s %s\n" "$name" "$detail"
    else
        printf "  \033[31m[FAIL]\033[0m %-16s %s\n" "$name" "$detail"
    fi
done
echo ""
echo "Full logs saved to: $LOGDIR"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "All checks passed. Safe to proceed to the real pipeline run:"
    echo "  python scripts/test_baseline.py vectorAdd"
    exit 0
else
    echo "$FAIL_COUNT check(s) failed. Resolve before running the pipeline in REAL mode."
    echo "(SATOSWARM_MOCK=1 mode will still work regardless -- it never calls these tools.)"
    exit 1
fi
