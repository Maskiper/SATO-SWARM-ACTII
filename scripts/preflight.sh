#!/usr/bin/env bash
# SATO SWARM — MI300X Pod Preflight Check
#
# Run this FIRST, before scripts/test_baseline.py, on the real pod. It does
# not touch the pipeline at all — it only answers one question: "does this
# environment actually have a working ROCm/HIP toolchain?"
#
# Checks:
#   1. hipcc, hipify-clang, amd-smi, rocprofv3 are on PATH (prints versions)
#   2. rocminfo runs and reports a GPU (looks for gfx942 = MI300X)
#   3. A trivial HIP kernel actually compiles AND runs AND prints real
#      device-side output — the only check here that proves the whole
#      chain (compiler -> GPU -> back to host) actually works end to end,
#      as opposed to just "the binary exists on PATH".
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
echo "SATO SWARM -- MI300X Pod Preflight Check"
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

if command -v hipify-clang &>/dev/null; then
    # hipify-clang's --version output format isn't consistently documented
    # across ROCm releases -- if this prints something unexpected, that's
    # still a PASS (the tool exists and ran); the raw text is in the log.
    ver=$(hipify-clang --version 2>&1 | head -n 1)
    echo "$ver" > "$LOGDIR/hipify_clang_version.log"
    pass "hipify-clang" "${ver:-present, no version string}"
else
    fail "hipify-clang" "not found on PATH"
fi

if ver=$(get_version amd-smi); then
    pass "amd-smi" "$ver"
else
    fail "amd-smi" "not found on PATH"
fi

if command -v rocprofv3 &>/dev/null; then
    # Same caveat as hipify-clang: rocprofv3's --version behavior wasn't
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
# 2. rocminfo -- confirm the GPU is actually visible to ROCm
# --------------------------------------------------------------------
echo "2. GPU visibility (rocminfo)"

if command -v rocminfo &>/dev/null; then
    ROCMINFO_OUT="$(rocminfo 2>&1)"
    echo "$ROCMINFO_OUT" > "$LOGDIR/rocminfo.log"
    if echo "$ROCMINFO_OUT" | grep -qi "gfx942"; then
        gpu_name=$(echo "$ROCMINFO_OUT" | grep -i "Marketing Name" | head -n 1 | sed 's/^[[:space:]]*//')
        pass "rocminfo" "gfx942 (MI300X) detected. ${gpu_name:-see preflight_logs/rocminfo.log}"
    else
        fail "rocminfo" "ran, but 'gfx942' not found in output -- see preflight_logs/rocminfo.log (wrong instance type, or driver issue)"
    fi
else
    fail "rocminfo" "not found on PATH"
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

if command -v hipcc &>/dev/null; then
    if hipcc -O2 --offload-arch=gfx942 -o "$HELLO_BIN" "$HELLO_SRC" > "$LOGDIR/hello_compile.log" 2>&1; then
        HELLO_OUT="$("$HELLO_BIN" 2>&1)"
        HELLO_RC=$?
        echo "$HELLO_OUT" > "$LOGDIR/hello_run.log"
        if [ "$HELLO_RC" -eq 0 ] && echo "$HELLO_OUT" | grep -q "Hello from GPU"; then
            n_lines=$(echo "$HELLO_OUT" | grep -c "Hello from GPU")
            pass "hip_hello_world" "compiled, ran, printed from device ($n_lines/4 threads confirmed) -- see preflight_logs/hello_run.log"
        else
            fail "hip_hello_world" "compiled but run failed or produced no device output (rc=$HELLO_RC) -- see preflight_logs/hello_run.log"
        fi
    else
        fail "hip_hello_world" "compilation failed -- see preflight_logs/hello_compile.log"
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
