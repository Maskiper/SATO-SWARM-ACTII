# SATO SWARM

Autonomous CUDA -> AMD ROCm/HIP migration pipeline, built to run and prove
itself on real AMD Instinct MI300X hardware.

SWARM = Swarm WorkForce Autonomous ReFActoring Migration.

Given a self-contained CUDA seed, the pipeline: copies it into an isolated
workspace, runs `hipify-clang`, compiles with `hipcc` for `gfx942`, runs the
resulting binary, captures live `amd-smi` telemetry, computes achieved
bandwidth/TFLOPS vs. MI300X's theoretical peaks, and writes a migration
report + a downloadable artifacts tarball. Every number in that report is
either a real measurement or explicitly labeled "Not captured" — nothing is
ever a guessed placeholder presented as real.

## Quick start

```powershell
pip install -r requirements.txt

# Mock mode — no AMD hardware needed, runs anywhere
$env:SATOSWARM_MOCK = "1"
python scripts/test_baseline.py vectorAdd
```

## Running for real, on the MI300X pod

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
| `1` | MOCK — no subprocess calls at all. Every tool function returns simulated MI300X-shaped data. |
| `0`, or **unset** | REAL — `hipify-clang`, `hipcc`, and `amd-smi` are actually invoked via subprocess. |

**Real is the default.** If the variable is ever lost or misconfigured on
the pod, the pipeline tries real tools and fails loudly and cleanly
(a normal `FAILED` job with a real error message) rather than silently
producing mock data that could be mistaken for a genuine hardware result.

## Seeds

Three self-contained CUDA kernels in `seeds/`:

- `vectorAdd.cu` — memory-bandwidth-bound, targets MI300X's ~5.3 TB/s HBM3 peak
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
`src/tools/execution.py`'s `_parse_amd_smi_json()` was written without a
real MI300X to validate the exact key names against. It tries several
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
  tools/execution.py    hipify / hipcc / amd-smi / binary-run wrappers — the mock/real switch lives here
  workspace/manager.py  Per-job isolated workspace (jobs/<job_id>/...)
  baseline/pipeline.py  The port -> validate -> benchmark -> report flow
seeds/                  Self-contained CUDA test kernels
scripts/
  preflight.sh           Pod toolchain check — versions + rocminfo + a real HIP compile/run (run this first)
  day0_verify.py         Older, lighter Python-based toolchain check (optional, still works)
  test_baseline.py       CLI entry point — runs the full pipeline end-to-end
RUNBOOK.md               Full copy-paste pod deployment sequence + what to save as proof
```

## Requirements

Runtime dependency: `pydantic>=2.9.0`. That's it — everything else is
Python standard library. On the MI300X pod, `hipify-clang`, `hipcc`, and
`amd-smi` need to be on `PATH`.
