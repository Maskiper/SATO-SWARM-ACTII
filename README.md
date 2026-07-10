# SATO SWARM

Autonomous CUDA -> AMD ROCm/HIP migration pipeline, built to run and prove
itself on real AMD GPU hardware.

SWARM = Swarm WorkForce Autonomous ReFActoring Migration.

Given a self-contained CUDA seed, the pipeline: copies it into an isolated
workspace, runs `hipify-clang`, compiles with `hipcc` for whichever GPU
architecture is actually detected on the machine (MI300X, RDNA3, whatever
— never hardcoded), runs the resulting binary, captures live `amd-smi`
telemetry, computes achieved bandwidth/TFLOPS vs. that GPU's real
theoretical peaks (when known), and writes a migration report + a
downloadable artifacts tarball. Every number in that report is either a
real measurement or explicitly labeled "Not captured" — nothing is ever a
guessed placeholder presented as real, and no metric is ever computed
against the wrong hardware's spec sheet.

## Quick start

```powershell
pip install -r requirements.txt

# Mock mode — no AMD hardware needed, runs anywhere
$env:SATOSWARM_MOCK = "1"
python scripts/test_baseline.py vectorAdd
```

## Running for real, on the pod

```bash
pip install -r requirements.txt

bash scripts/preflight.sh            # verifies hipcc/hipify-clang/amd-smi/rocprofv3/rocminfo
                                      # + compiles & runs a real HIP kernel end to end
# Real mode is the default — just don't set SATOSWARM_MOCK (or set it to 0)
python scripts/test_baseline.py vectorAdd
```

See **[RUNBOOK.md](RUNBOOK.md)** for the full copy-paste deployment sequence
(clone, install, preflight, mock smoke test, real run, exactly which
output files to save as proof), and **The mock/real switch** below before
you flip it on the pod. `scripts/day0_verify.py` is an older, lighter
Python-based toolchain check that's still there if useful, but
`scripts/preflight.sh` is the one that actually compiles and runs
something — start with that.

## The mock/real switch

One environment variable controls everything: `SATOSWARM_MOCK`, read exactly
once, in `src/tools/execution.py`. Every other module imports that constant
rather than re-reading the environment, so there's a single source of truth.

| `SATOSWARM_MOCK` | Behavior |
|---|---|
| `1` | MOCK — no subprocess calls at all. Every tool function returns simulated GPU-shaped data. |
| `0`, or **unset** | REAL — `hipify-clang`, `hipcc`, and `amd-smi` are actually invoked via subprocess. |

**Real is the default.** If the variable is ever lost or misconfigured on
the pod, the pipeline tries real tools and fails loudly and cleanly
(a normal `FAILED` job with a real error message) rather than silently
producing mock data that could be mistaken for a genuine hardware result.

## How the target GPU architecture is detected

`hipcc` needs `--offload-arch=<gfxNNNN>` to compile — and getting that
wrong doesn't fail the build, it produces a binary that **segfaults at
launch** on hardware whose ISA doesn't match what was compiled for. This
project provisioned a pod expecting an MI300X (`gfx942`) and got an
RDNA3/Radeon 7900-class card (`gfx1100`) instead — real ROCm hardware,
just not the architecture that was hardcoded at the time, and the
hello-world check segfaulted exactly as described above. Nothing in this
codebase hardcodes an architecture anymore.

`src/tools/execution.py`'s `detect_gpu_arch()` is the single source of
truth, tried in order:
1. `rocm_agent_enumerator` — purpose-built for this, one gfx code per
   agent (`gfx000` is the host/CPU placeholder and is skipped).
2. `rocminfo` — falls back to scanning its output for any `gfxNNNN` token.
3. If both come up empty, `run_hipcc()` falls back to
   `--offload-arch=native`, letting the compiler itself auto-detect the
   build machine's GPU (supported on sufficiently recent ROCm compilers).

Whatever was actually used is recorded on `job.gpu_arch` and shown in
every message and in the report's `**Hardware**` line — never assumed,
never silently defaulted to a specific chip. `scripts/preflight.sh` runs
the same detection before compiling its hello-world check, so a
architecture mismatch shows up there, in seconds, instead of partway
through a real pipeline run.

**Efficiency percentages follow the same rule.** `src/baseline/pipeline.py`'s
`GPU_THEORETICAL_PEAKS` dict maps a detected architecture to its real
spec-sheet numbers (HBM bandwidth, FP32 TFLOPS) — currently just
`gfx942` (MI300X). If the detected architecture isn't in that table (e.g.
`gfx1100` right now — its exact SKU wasn't confirmed against the pod when
this was written), `efficiency_percent` / `efficiency_tflops_percent`
stay `None` — "Not applicable" in the report — rather than divide a real
achieved number by another GPU's peak and print a meaningless percentage.
Add a verified entry to that dict for any architecture you want efficiency
computed for; `achieved_bw_gbs` / `achieved_tflops` themselves are
unaffected either way, since those come straight from the binary's own
measured output.

## Seeds

Three self-contained CUDA kernels in `seeds/`:

- `vectorAdd.cu` — memory-bandwidth-bound (e.g. ~5.3 TB/s HBM3 on MI300X — see the architecture-detection section below for how the actual peak used depends on the detected GPU)
- `tiledMatmul.cu` — compute-bound, shared-memory tiling, targets FP32 TFLOPS peak
- `reduction.cu` — control flow, `__syncthreads`, atomics

Each is fully self-contained: CUDA source + host driver + its own
correctness self-check + timing, in one `.cu` file.

## How timing and bandwidth/TFLOPS are actually measured

Every seed in `seeds/*.cu` wraps its kernel launch in CUDA events
(`cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` /
`cudaEventElapsedTime`), which `hipify-clang` translates directly into the
HIP equivalents before compilation — this is genuine GPU-side timing,
measured by the device itself, not a Python-side guess. The binary prints
this as `Kernel time: X ms`.

`src/tools/execution.py`'s `parse_binary_output_for_metrics()` reads that
line (plus, depending on the seed, `Total data moved: X GB`, `FLOPs: X
GFLOPs`, or `Elements: N`) straight from the binary's real stdout.
`src/baseline/pipeline.py`'s `_compute_derived_metrics()` then computes
achieved bandwidth/TFLOPS in Python from those real numbers —
`bytes_moved / kernel_time_seconds`, `flops / kernel_time_seconds` — never
from a constant. If the binary's own self-printed "Achieved ..." line
disagrees with that independently-computed value by more than 1%, a
warning is logged to the job's message trace, and the Python-computed
value is what's reported.

If the binary doesn't print a parseable `Kernel time:` line (e.g. it
crashed first), `kernel_time_ms` is `None` — "Not captured" in the report.
There is no wall-clock fallback standing in for it: the pipeline does
record the whole process's wall-clock time (in `logs/run.log`), but only
as a diagnostic aside, explicitly not labeled "Kernel time," since it
measures something different (process/subprocess overhead included) from
the real GPU-side hipEvent measurement.

The same real max-abs-diff treatment applies to correctness validation:
each seed prints an actual-vs-expected pair (e.g. `c[0]=0.1800 (exp
0.1800)`), which is parsed and diffed for real — `max_abs_diff` is never a
hardcoded `0.0` standing in for "it passed."

In MOCK mode, the exact same parsing + computation code runs — the only
difference is that the "binary's stdout" was fabricated text instead of a
real subprocess result. Every mock number is tagged `(SIMULATED)` inline
in the report table, on top of the page-level "Mode: MOCK" banner.

## Known limitation: amd-smi metric parsing

`amd-smi metric --json`'s schema varies across ROCm releases, and
`src/tools/execution.py`'s `_parse_amd_smi_json()` was written without
real hardware to validate the exact key names against. It tries several
plausible field paths; anything it can't confidently find is left as `None`
and the report shows "Not captured" — never a guessed number. The raw
`amd-smi` text is always saved to `logs/amd_smi_pre.txt` /
`amd_smi_post.txt` regardless, so nothing measured is ever lost.

**On Day 0**: run `amd-smi metric --json` on the real pod, compare its
actual structure against the candidate paths in `_parse_amd_smi_json()`,
and adjust the key names there if the parsed metrics come back empty on a
real run. Kernel time, achieved bandwidth, and achieved TFLOPS do **not**
depend on this — those are parsed directly from the seed binary's own
stdout and are real (or explicitly absent) regardless of the amd-smi
parser's accuracy.

## Project layout

```
src/
  models/job.py        Job state, metrics, and report schema (Pydantic)
  tools/execution.py    hipify / hipcc / amd-smi / binary-run wrappers — mock/real switch + GPU arch auto-detection live here
  workspace/manager.py  Per-job isolated workspace (jobs/<job_id>/...)
  baseline/pipeline.py  The port -> validate -> benchmark -> report flow + GPU_THEORETICAL_PEAKS
seeds/                  Self-contained CUDA test kernels
scripts/
  preflight.sh           Pod toolchain check — versions + rocminfo + a real HIP compile/run (run this first)
  day0_verify.py         Older, lighter Python-based toolchain check (optional, still works)
  test_baseline.py       CLI entry point — runs the full pipeline end-to-end
RUNBOOK.md               Full copy-paste pod deployment sequence + what to save as proof
```

## Requirements

Runtime dependency: `pydantic>=2.9.0`. That's it — everything else is
Python standard library. On the pod (whatever AMD GPU it turns out to
have), `hipify-clang`, `hipcc`, `amd-smi`, `rocminfo`, and ideally
`rocm_agent_enumerator` need to be on `PATH` — `scripts/preflight.sh`
checks all of them.
