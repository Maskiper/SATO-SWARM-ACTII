# SATO SWARM

Autonomous CUDA -> AMD ROCm/HIP migration pipeline, built to run and prove
itself on real AMD GPU hardware.

SWARM = Swarm WorkForce Autonomous ReFActoring Migration.

Given a self-contained CUDA seed, the pipeline: copies it into an isolated
workspace, translates it to HIP (`hipify-perl` preferred — no CUDA SDK
required; `hipify-clang` only as a fallback), compiles with `hipcc` for
whichever GPU architecture is actually detected on the machine (MI300X,
RDNA3, whatever — never hardcoded), runs the resulting binary, captures live `amd-smi`
telemetry, computes achieved bandwidth/TFLOPS vs. that GPU's real
theoretical peaks (when known), and writes a migration report + a
downloadable artifacts tarball. Every number in that report is either a
real measurement or explicitly labeled "Not captured" — nothing is ever a
guessed placeholder presented as real, and no metric is ever computed
against the wrong hardware's spec sheet.

Beyond that core baseline, the project now also includes a bounded
**Repair Loop** that can mechanically fix and recompile past one verified
real CUDA -> HIP gap (see **Memory System & Repair Loop** below), a
sandboxed **Tool Registry** the repair loop calls through, a **FastAPI**
backend (`src/main.py`) that's a thin wrapper around this exact same
pipeline code — never a reimplementation — and a **Next.js frontend** for
watching a job run live. See the dedicated sections below for each.

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

bash scripts/preflight.sh            # verifies hipcc/hipify(-perl or -clang)/amd-smi/rocprofv3/rocminfo
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
| `0`, or **unset** | REAL — `hipify-perl` (preferred) or `hipify-clang`, `hipcc`, and `amd-smi` are actually invoked via subprocess. |

**Real is the default.** If the variable is ever lost or misconfigured on
the pod, the pipeline tries real tools and fails loudly and cleanly
(a normal `FAILED` job with a real error message) rather than silently
producing mock data that could be mistaken for a genuine hardware result.

## Which hipify tool is used, and why

There are two CUDA-to-HIP translators in the ROCm toolchain, and they
have fundamentally different requirements:

- **`hipify-clang`** parses source with an actual clang front end, which
  means it needs a real CUDA SDK installed (`cuda_runtime.h`, libdevice)
  to resolve against — even though it's only *translating* the code, not
  compiling it for NVIDIA. On an AMD-only box (no CUDA installed at all,
  by definition), it fails immediately with "cannot find CUDA
  installation" and never translates anything.
- **`hipify-perl`** is pure text/regex substitution — no parsing, no CUDA
  SDK required. This is the tool that actually works on an AMD-only pod,
  confirmed by hand at `/opt/rocm/bin/hipify-perl` converting `vectorAdd.cu`
  cleanly with nothing else installed.

`src/tools/execution.py`'s `run_hipify()` prefers hipify-perl
unconditionally when it's present. hipify-clang is only attempted as a
fallback, and only if hipify-perl is missing **and** a real CUDA SDK is
actually detected (`nvcc` on PATH, or `/usr/local/cuda/include/cuda_runtime.h`)
— otherwise it would just fail the same way, for no benefit.

**The interfaces are also different**, not just the requirements:
hipify-clang takes a `--cuda-path` and can batch multiple files into a
`-o <dir>`; hipify-perl takes exactly one `.cu` file and prints the
translated HIP source to stdout (diagnostics go to stderr, so the stdout
stream stays clean and redirectable). `run_hipify()` hipifies each source
file individually and writes hipify-perl's captured stdout to
`<stem>.hip.cpp` in the job's `hip_out/` directory itself — exactly where
`run_hipcc()` looks for sources afterward. Whichever tool actually ran is
recorded on `job.hipify_command` and shown in the report — never assumed.

`scripts/preflight.sh` checks the same way: hipify-perl first, then
hipify-clang only if it's paired with a detected CUDA SDK (flagged as a
likely-to-fail FAIL otherwise, since it would pass the presence check but
still fail at actual invocation time).

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

**Efficiency percentages follow the same rule.** `src/tools/execution.py`'s
`detect_gpu_theoretical_peaks()` computes the theoretical peak (memory
bandwidth, FP32 TFLOPS) live, every run, from whatever GPU is actually
attached — no hardcoded per-SKU table to keep updating as new cards show
up:
- Bandwidth: `mem_clock_mhz * bus_width_bits * ddr_factor / 8 / 1000`, with
  `mem_clock_mhz` (max), `bus_width_bits`, and vram type all queried from
  `amd-smi`. `ddr_factor` is **not** a flat constant — it's looked up by
  memory *technology* (`_MEM_TECH_DDR_FACTOR` in `src/tools/execution.py`):
  HBM uses a textbook DDR ×2 (confirmed against MI300X's published 5300
  GB/s); GDDR6 uses an empirically-calibrated ×17.8 (confirmed against a
  real RX 7900 XTX/gfx1100 pod — a flat ×2, or even the naive JEDEC-
  clock-ratio guess of ×8, both undershoot real GDDR6 bandwidth
  substantially; see that constant's comment for the full derivation).
  GDDR6X/GDDR5 have no confirmed factor yet and are left unhandled rather
  than guessed.
- FP32 TFLOPS: `compute_units * flops_per_clock_per_cu * engine_clock_mhz / 1e6`,
  with `compute_units` and `engine_clock_mhz` (max) queried from
  `rocminfo`, and `flops_per_clock_per_cu` a small per-architecture-
  *family* constant (not per-SKU — every card in one microarchitecture
  generation shares the same per-CU datapath width).

If either live query can't produce a usable value on this ROCm version
(schema varies by release — same caveat that already applies to amd-smi
telemetry parsing — or the vram technology isn't recognized), it falls
back to a small verified-spec-sheet table (currently `gfx942`/MI300X and
`gfx1100`/RX 7900 XTX), and only if that has no entry either do
`efficiency_percent` / `efficiency_tflops_percent` stay `None` — "Not
applicable" in the report — rather than divide a real achieved number by
another GPU's peak. Every report shows a **Theoretical peak calculation**
line with the exact inputs and formula used (or which fallback path fired
instead), and the full raw rocminfo/amd-smi query is saved to
`logs/gpu_specs.log`, so the number can be checked, not just trusted.
`achieved_bw_gbs` / `achieved_tflops` themselves are unaffected either
way, since those come straight from the binary's own measured output.
`scripts/verify_gpu_specs.py` dumps the raw query + computed result
standalone, without needing a full pipeline run.

## Seeds

Five self-contained CUDA kernels in `seeds/`:

- `vectorAdd.cu` — memory-bandwidth-bound (e.g. ~5.3 TB/s HBM3 on MI300X — see the architecture-detection section below for how the actual peak used depends on the detected GPU)
- `tiledMatmul.cu` — compute-bound, shared-memory tiling, targets FP32 TFLOPS peak
- `reduction.cu` — control flow, `__syncthreads`, atomics
- `repairDemo.cu` — deliberately exercises one verified-real CUDA -> HIP
  porting gap (`cudaCtxResetPersistingL2Cache`, see **Memory System &
  Repair Loop** below), to give the Repair Loop a real, reproducible
  compile failure to fix. Not a hidden trick — the gap and why it's real
  are documented in the file's own header comment.
- `multiFileDemo/` — a DIRECTORY, not a single `.cu` file: `main.cu`
  calls a kernel + host wrapper defined in `helper.cu` and declared in
  `helper.cuh`, exercising multi-file hipify + hipcc compile-and-link
  (see **Tool Registry** below for the general mechanism). Not a
  benchmark seed — no achieved-bandwidth/TFLOPS number is computed for
  it, only real hipEvent kernel timing and a real correctness check.

Each of the 4 single-file seeds is fully self-contained: CUDA source +
host driver + its own correctness self-check + timing, in one `.cu`
file — except `repairDemo.cu`, a partial exception (a single no-op
kernel with no bandwidth/TFLOPS self-check, since its entire purpose is
the one intentional compile gap, not a performance measurement).
`multiFileDemo/` is the one seed that isn't single-file at all by
design — see its own bullet above.

## How timing and bandwidth/TFLOPS are actually measured

Every seed in `seeds/*.cu` wraps its kernel launch in CUDA events
(`cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` /
`cudaEventElapsedTime`), which hipify translates directly into the HIP
equivalents before compilation — this is genuine GPU-side timing,
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

## amd-smi metric parsing — fixed and empirically verified

`amd-smi metric --json`'s schema varies across ROCm releases, and
`src/tools/execution.py`'s `_parse_amd_smi_json()` was originally written
without real hardware to validate the exact key names against (candidate
paths, best guess). **This is no longer the case.** The parser has since
been corrected and empirically verified against real captured amd-smi
output from all 4 real job directories (`job_1684fdb652d5` /
`job_374d6e8c51d1` / `job_7eeb1f8358f8` / `job_09ef95c5f62b` —
`logs/gpu_specs.log` and `logs/amd_smi_{pre,post}.txt` in each), not
guessed a second time. Three real, distinct bugs were found and fixed by
diffing the parser's assumptions against that real captured JSON:

1. **Missing top-level unwrap** — real amd-smi wraps everything in
   `{"gpu_data": [{...}]}`; the parser assumed a bare list or bare dict at
   the top level and never unwrapped this, so every field lookup failed
   silently before it even started. Fixed by a new
   `_unwrap_amd_smi_gpu_node()` helper, applied at every call site
   (`_parse_amd_smi_json()` and both live-spec queries in
   `_query_amd_smi_mem_bus_specs()`).
2. **Un-unwrapped `{"value": N, "unit": "..."}` readings** — every real
   numeric field (`clock.gfx_0.clk`, `clock.mem_0.max_clk`, etc.) is
   wrapped this way; `_try_float()` returned `None` for a dict input it
   didn't know how to unwrap. Fixed by adding a dict-with-`"value"`-key
   branch to `_try_float()`.
3. **Wrong key names** — `mem_usage.used` is actually `mem_usage.used_vram`;
   `clock.sclk`/`clock.gfx_clk` are actually `clock.gfx_0.clk`;
   `clock.mclk`/`clock.mem_clk` are actually `clock.mem_0.clk`. Fixed by
   trying the confirmed-real paths first, with the original guesses kept
   as fallbacks (harmless if some other ROCm release really does use
   them).

A fourth bug, unrelated to key paths, surfaced during verification itself:
`_try_float(a) or _try_float(b)`-style fallback chains silently discarded
a genuine `0.0` reading (GPU idle before kernel launch is real, valid
data) because `0.0` is falsy in Python — the chain would keep falling
through to the next candidate as if the first one were missing. Fixed with
a `_first_non_none()` helper that checks `is not None` instead of
truthiness, applied to all 6 parsed telemetry fields.

All four fixes were verified by feeding the exact real captured bytes from
the 4 real job directories above through the actual parser functions —
not synthetic test JSON — confirming all 6 telemetry fields
(utilization, power, temperature, memory, sclk, mclk) now parse as
non-`None` across all captures, and that the independently-computed GDDR6
bandwidth (see below) still lands within 0.03% of the RX 7900 XTX's
published spec after the fix. `_first_non_none()` and
`_unwrap_amd_smi_gpu_node()` live in `src/tools/execution.py`, right next
to `_parse_amd_smi_json()`.

Kernel time, achieved bandwidth, and achieved TFLOPS were never affected
by any of this — those are parsed directly from the seed binary's own
stdout, independent of amd-smi, in both the buggy and fixed versions of
the parser. The raw `amd-smi` text is still always saved to
`logs/amd_smi_pre.txt` / `amd_smi_post.txt` regardless of parser
correctness, so nothing measured is ever lost even if a future ROCm
release changes the schema again.

## Tool Registry — sandboxed action surface

`src/agents/tools.py`'s `ToolRegistry` gives a bounded, sandboxed surface
for taking real actions inside one job's own workspace:
`ToolRegistry(job_id, workspace_dir).execute(tool_name, **kwargs)` always
returns `{"success": bool, "result": ..., "error": Optional[str]}` and
never raises, no matter what went wrong. 8 tools:

| Tool | What it does |
|---|---|
| `run_hipify` | calls the real `execution.run_hipify()` |
| `run_hipcc` | calls the real `execution.run_hipcc()` |
| `capture_amd_smi` | calls the real `execution.capture_amd_smi_snapshot()` |
| `run_benchmark` | calls the real `execution.run_binary()` |
| `read_file` | reads one file from this job's workspace |
| `apply_search_replace` | patches a file — succeeds **only** on an exactly-one-match `old_text`; refuses (file left untouched) on 0 or 2+ matches, so an ambiguous edit can never silently land on the wrong occurrence |
| `list_workspace_files` | lists files under this job's workspace |
| `write_agent_note` | appends a timestamped line to a shared `notes/blackboard.md` |

The first four call directly into the same `src/tools/execution.py`
functions the baseline pipeline itself uses — never reimplemented here,
and this module never re-checks `SATOSWARM_MOCK` itself (`execution.py`
remains the single source of truth for that switch). The last four are
workspace-native and never touch `execution.py` at all; every one of them
is sandboxed to the job's own `workspace_dir` by resolving the path and
checking real filesystem containment (`Path.relative_to()` on the
resolved path, not a naive string-prefix check) — refuses `../`
traversal, an absolute-path override, and a symlink escape alike.
`scripts/test_tools.py` (28/28 passing) tests every tool plus two
dedicated sandbox-escape attempts.

**Wiring status, stated plainly**: the Tool Registry is wired into the
pipeline for exactly one purpose today — the Repair Loop's
`apply_search_replace` + `run_hipcc` calls (next section). It is **not**
wired in for the 3 original seeds' normal hipify/hipcc/benchmark steps,
which still call `execution.py` directly, unchanged. Broader agent-driven
tool use is future work, not a current claim.

## Memory System & Repair Loop

`src/memory/loader.py`'s `PortingMemory` is a small, file-backed
(`memory/porting_patterns.jsonl` — JSONL, one JSON object per line, hand-
curatable) knowledge base of known CUDA -> HIP porting gaps, retrieved by
keyword overlap against a real compiler/hipify error message —
`get_relevant_patterns(error_text)` scores each pattern by what fraction
of **its own** identifying keywords appear in the error text (plain
token overlap, not semantic reasoning). 7 curated patterns ship today,
each with a real, cited source (HIPIFY's own unsupported-function table,
or the CUDA/HIP header/doc trail directly, not invented): the original 4
(`cudaFuncGetName`, `cudaGraphConditionalHandleCreate`,
`cudaDeviceFlushGPUDirectRDMAWrites`, `cudaCtxResetPersistingL2Cache`)
plus 3 more added later with the same rigor
(`cudaInitDevice`, `cudaEventElapsedTime_v2`, `cudaOccupancyMaxActiveClusters`
— each independently confirmed absent from HIP's real current header by
direct search, not just from the HIPIFY compatibility table alone).
`scripts/test_memory.py` (11/11 passing) exercises retrieval,
`add_pattern()`, and the full agent-context dump.

Only one of the 7 has an `auto_fix` — a literal `old_text`/`new_text`
pair a repair loop can apply *mechanically*, as opposed to `hip_fix`'s
human-readable prose, which is never applied automatically:
**`gap_cudaCtxResetPersistingL2Cache`**.

### The real gap: `cudaCtxResetPersistingL2Cache`

`cudaCtxResetPersistingL2Cache()` is a real CUDA Runtime API function
(`__host__ cudaError_t cudaCtxResetPersistingL2Cache(void)`, CUDA 11.3+)
that resets persisting L2 cache lines. Verified directly against the real
HIPIFY and HIP source, not inferred:

- **hipify-perl** (`github.com/ROCm/HIPIFY`, amd-develop branch) lists it
  in `%hash_HipOnlyUnsupportedFunctions` (`bin/hipify-perl:5584`) and it's
  confirmed absent from `%map_core`, the real substitution table
  (`bin/hipify-perl:10286-16314` — grepped directly, zero matches).
  hipify-perl prints a stderr warning but **exits 0** (no `exit()` call
  anywhere in the ~18,300-line script is tied to a warning) and leaves
  the identifier byte-for-byte untranslated.
- **HIP's real public header** (`github.com/ROCm/HIP`, develop branch,
  `include/hip/hip_runtime_api.h`, 10,524 lines) was searched directly —
  no `hipCtxResetPersistingL2Cache` or equivalent exists under any name.
  HIP does model the *setup* side of the same feature
  (`hipAccessPolicyWindow`, `hipAccessPropertyPersisting`,
  `hipDeviceAttributePersistingL2CacheMaxSize` all exist), which is
  exactly what makes the gap easy to miss — the matching reset call
  simply isn't there.

`seeds/repairDemo.cu` is a minimal seed (one no-op kernel) that calls
this function once, deliberately, giving the repair loop a real,
reproducible compile failure to fix rather than a synthetic one.

### How the repair loop uses it

`src/baseline/pipeline.py`'s `_attempt_hipcc_repair()` runs only when
`job.seed_id == SeedId.REPAIR_DEMO` and the first `hipcc` attempt fails —
for the 3 original seeds this entire block (including constructing
`ToolRegistry`/`PortingMemory`) never executes, and their control flow is
byte-for-byte what it was before the repair loop existed
(`scripts/test_repair_loop.py`'s regression section checks this
directly). Per attempt, budget 3 (`scripts/test_repair_loop.py`, 27/27
passing):

1. Query `PortingMemory.get_relevant_patterns()` with the real hipcc
   error text.
2. If the top untried match has an `auto_fix`, apply it via
   `ToolRegistry.execute("apply_search_replace", ...)` against whichever
   HIP source file actually contains the pattern's `old_text`.
3. Recompile via `ToolRegistry.execute("run_hipcc", ...)`.
4. On success: record a new `confirmed_repair` pattern back into memory
   (distinct from the original research-sourced one, `confidence: 0.99`,
   citing the specific job ID that confirmed it) and continue exactly
   like an ordinary first-try hipcc success. On failure: try the next
   candidate pattern, reacting to the *new* hipcc error, until the budget
   is exhausted, then fall through to the ordinary FAILED path unchanged.

Every repair-loop message states only the mechanical action taken (which
pattern ID matched, its stored confidence, what was patched) — never
language implying reasoning beyond the keyword-overlap match that was
actually performed.

**Confirmed on real hardware, not just in mock-mode tests**:
`jobs/job_1684fdb652d5/` is a real, completed repairDemo run captured
from the pod (`gpu_arch: gfx1100`, `mode: REAL`). Its real message trace
shows hipcc genuinely failing on `cudaCtxResetPersistingL2Cache`, the
repair loop matching the pattern, `apply_search_replace` patching the
real file, and the recompile genuinely succeeding — the same sequence
`scripts/test_repair_loop.py` proves via a monkeypatched first-call
failure in mock mode, but this run proves the *underlying compile
failure itself* is real hardware behavior, not just a predicted one.

## FastAPI backend

`src/main.py` is a thin routing layer only — every endpoint calls
directly into the same already-proven functions the CLI
(`scripts/test_baseline.py`) already uses (`run_baseline()`,
`WorkspaceManager`, `PortingMemory`, `execution.py`'s
`MOCK`/`detect_gpu_arch()`). No pipeline logic is reimplemented in this
file. `scripts/test_main.py`'s CRITICAL test proves this by running the
same seed via a direct `run_baseline()` call and via a full HTTP
`TestClient` request, then diffing the resulting `JobState` and
`migration_report.md` (modulo job ID/timestamps/paths) — 40/40 checks
passing, including this one.

Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /health` | system name, current mode, detected GPU arch, memory pattern count, tool count, jobs run this session |
| `POST /jobs?seed_id=...` | starts a new job in the background, returns immediately with `job_id` |
| `GET /jobs/{id}/status` | full `JobState` — re-reads `state.json` fresh every call, never a cached/duplicated copy |
| `GET /jobs/{id}/report` | `migration_report.md` as plain text, once the job reaches Reporting |
| `GET /jobs/{id}/artifacts` | the artifacts tarball |
| `GET /jobs/{id}/stream` | Server-Sent Events — polls `state.json` (~750ms) and pushes any new `AgentMessage`s, then a final `done` event |
| `POST /demo/replay?seed_id=...` | returns the report + `JobState` from one of the 4 **actual, already-completed real-hardware job dirs** (see `REPLAY_JOB_IDS` in `src/main.py`) — never runs anything new; 404s plainly if the job dir isn't present in this checkout rather than substituting mock data |

Local-only by design: binds `127.0.0.1`, CORS restricted to
`localhost:3000`/`127.0.0.1:3000`. Job state is never duplicated in
memory — every read re-reads the same `state.json`
`WorkspaceManager`/`run_baseline()` already write.

## Frontend

`frontend/` is a Next.js 16 / React 19 / TypeScript app (Tailwind 4,
Vitest + React Testing Library for tests — 23/23 passing) that talks to
the FastAPI backend over plain `fetch`/`EventSource`
(`frontend/lib/api.ts`), nothing more exotic. Five components:

| Component | Role |
|---|---|
| `LiveJobView` | top-level view for one job — status banner, mode badge, GPU arch, polls `GET .../status` |
| `AgentFeed` | live message trace — SSE primary path (`GET .../stream`), with a redundant polling fallback reading the exact same `state.json` source of truth for browsers/proxies where `EventSource` misbehaves |
| `PhaseTimeline` | the 5 real pipeline phases as a stepper, driven by `job.completed_phases`/`job.phase` — shows a "repair loop engaged" badge derived from whether a real `Repair Loop` message exists in the job's own trace, not hardcoded to `repairDemo` |
| `MetricsDashboard` | kernel time / achieved bandwidth-or-TFLOPS / efficiency / GPU telemetry, `"Not captured"`/`"Not applicable"` rendered exactly like the markdown report does — plus the mode footer, see below |
| `ReportViewer` | renders `migration_report.md` via `react-markdown`; reads FAILED/COMPLETED from the report's own `**Final Status**:` line via regex, not a separately-passed prop — so it tells the truth even given only raw markdown text with no live `JobState` alongside it (e.g. a replayed historical report) |

### `job.mode` — real vs. simulated, independent of the server's current mode

`JobState.mode` (`"MOCK"` or `"REAL"`) is set exactly once, in
`run_baseline()`, from the same `SATOSWARM_MOCK`-derived `MOCK` constant
that governs everything else in that run — and never touched again
afterward. This is deliberately **independent** of whatever the server
happens to be running as later: a real job replayed from a server
currently running in MOCK mode still correctly shows as real, and a mock
job still correctly shows as simulated even if the server is later
switched to real mode.

Two places in the UI read `job.mode` directly (never a server-level
prop):
- `LiveJobView`'s mode badge, next to the status banner — "REAL DATA"
  (green) or "SIMULATED" (amber).
- `MetricsDashboard`'s footer — "This job ran on real hardware — no
  simulation" or "This job's data is simulated (MOCK mode) — not
  measured".

The page header's health summary (`Server: {mode} mode...`) is a
**separate, clearly distinct** statement about the server's *current*
mode — what a **new** run would use — not about whichever job happens to
be displayed below it. The two can legitimately disagree (a MOCK-mode
server replaying a REAL historical job is the normal case for local
development), and the UI is designed so that disagreement reads as
informative, not contradictory.

Older `state.json` files written before `job.mode` existed default to
`"MOCK"` on load (a Pydantic field default, for backward compatibility)
— the 4 real job directories pulled from the pod predate this field and
were backfilled to `mode: "REAL"` once, by hand, after confirming each
one's own report text independently already said `**Mode**: REAL
hardware`.

## Running the full stack locally (backend + frontend)

This is a separate, **local-only** stack from the pod CLI flow above —
see **[RUNBOOK.md](RUNBOOK.md)** for the full walkthrough, including how
this differs from (and doesn't replace) the pod deployment sequence.
Quick version:

```powershell
# Terminal 1 — backend
$env:SATOSWARM_MOCK = "1"   # or unset/0 on a real ROCm box
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:3000`. Each seed card has **Run** (starts a
new job against whatever mode the backend is running as) and **Replay
real run** (loads one of the 4 actual pod-captured job dirs via
`/demo/replay` — works even when the backend itself is in MOCK mode,
since it never runs anything new).

## Architecture

Same diagram as **JUDGING.md** — kept identical in both places on purpose;
see the sections above (Tool Registry, Memory System & Repair Loop,
FastAPI backend) for the full detail behind each box.

```
seed .cu(s) -> WorkspaceManager -> hipify (hipify-perl) -> hipcc -> run binary
                                                              |
                                                   [compile failed AND
                                                    seed == repairDemo]
                                                              v
                                          Repair Loop <- Memory (porting_patterns.jsonl)
                                                |             via Tool Registry
                                                v
                                      recompile -> validate -> amd-smi -> report + tar
```

## Project layout

```
src/
  models/job.py          Job state, metrics, and report schema (Pydantic) — incl. job.mode
  tools/execution.py     hipify (perl/clang auto-select) / hipcc / amd-smi / binary-run wrappers — mock/real switch + GPU arch AND theoretical-peak auto-detection live here
  workspace/manager.py   Per-job isolated workspace (jobs/<job_id>/...)
  baseline/pipeline.py   The port -> validate -> benchmark -> report flow, incl. the Repair Loop
  agents/tools.py        ToolRegistry — sandboxed action surface (8 tools)
  memory/loader.py       PortingMemory — the porting-pattern knowledge base
  main.py                FastAPI backend — thin wrapper around the above, nothing reimplemented
seeds/                   Self-contained CUDA test kernels (vectorAdd, tiledMatmul, reduction, repairDemo, multiFileDemo)
memory/porting_patterns.jsonl   The 7 curated porting patterns (JSONL)
frontend/                Next.js/React/TypeScript UI — see Frontend section above
scripts/
  preflight.sh            Pod toolchain check — versions + rocminfo + a real HIP compile/run (run this first)
  day0_verify.py          Older, lighter Python-based toolchain check (optional, still works)
  test_baseline.py        CLI entry point — runs the full pipeline end-to-end
  test_tools.py           Tool Registry regression (28 checks)
  test_memory.py          Memory System regression (11 checks)
  test_repair_loop.py     Repair Loop regression (27 checks)
  test_main.py            FastAPI backend regression, incl. the CLI/API equivalence test (40 checks)
track/BUILD_LOG.md       Factual, git-log-sourced engineering timeline
RUNBOOK.md               Full copy-paste pod deployment sequence + local frontend/backend section
```

## Requirements

**Backend runtime**: `pydantic>=2.9.0`, `fastapi>=0.137.0`, `uvicorn>=0.49.0`
(`requirements.txt`; `httpx` is test-only, for `scripts/test_main.py`'s
`TestClient`). On the pod (whatever AMD GPU it turns out to have),
`hipify-perl` (preferred — see below), `hipcc`, `amd-smi`, `rocminfo`,
and ideally `rocm_agent_enumerator` need to be on `PATH` —
`scripts/preflight.sh` checks all of them.

**Frontend** (only needed for the local UI, not the pod CLI flow):
Node.js + npm, see `frontend/package.json` (Next.js 16, React 19,
TypeScript, Tailwind 4; Vitest + React Testing Library for tests).
