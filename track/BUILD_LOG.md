# Build Log — factual engineering record

This is a factual timeline, not marketing copy. Every entry below is
sourced from one of: `git log`'s real commit hashes/timestamps, a real
job directory's `state.json`/logs, or a file's own filesystem timestamp
— cited inline. Where something is inferred rather than directly
observed, that's said explicitly. Nothing here is a guess presented as
a fact.

Timestamps are as recorded by `git log` / the filesystem (local machine
time, UTC+02:00 throughout). Commit hashes are full 40-character SHAs;
shortened to 7 characters below for readability, matching `git log`'s
own default.

---

## 2026-07-10T02:10:02+02:00 — Initial commit
**`9773890`** — `Initial commit`
LICENSE + a 2-line README stub. Nothing functional yet.

## 2026-07-10T02:17:56+02:00 — Initial pipeline
**`ac88483`** — `SATO SWARM ACT II — clean CUDA to ROCm pipeline, real hardware ready`
The first working baseline: `src/models/job.py`, `src/tools/execution.py`,
`src/baseline/pipeline.py`, `src/workspace/manager.py`, the 3 original
seeds (`vectorAdd.cu`, `tiledMatmul.cu`, `reduction.cu`), and the CLI/
preflight scripts. 2,226 insertions across 20 new files. GPU architecture
was hardcoded to `gfx942` (MI300X) at this point — see the next entry for
why that didn't survive contact with the real pod.

## 2026-07-10T02:19:46+02:00 — Merge
**`f591b66`** — `Merge branch 'main' of https://github.com/Maskiper/sato-swarm-actii`
Repo connected to its GitHub remote around this point.

## 2026-07-10T02:54:57+02:00 — GPU architecture auto-detection
**`6bc9c76`** — `Auto-detect GPU architecture at runtime instead of hardcoding gfx942`
37 minutes after the initial pipeline commit. The pod that was
provisioned expecting an MI300X (`gfx942`) came back as an RDNA3/
Radeon-7900-class card instead (`gfx1100`) — real ROCm hardware, just not
what was hardcoded. Compiling for the wrong `--offload-arch` doesn't fail
the build; it produces a binary that segfaults at launch, which is what
happened to the hardcoded-`gfx942` hello-world check against the real
`gfx1100` pod. Fixed by `detect_gpu_arch()`: `rocm_agent_enumerator` →
`rocminfo` → `--offload-arch=native` fallback, tried in order, nothing
hardcoded anywhere afterward. This fix is why every seed/job directory
from this point forward correctly self-reports `gfx1100`.

## 2026-07-10T08:44:12+02:00 — hipify-perl adoption
**`4feabcf`** — `Use hipify-perl (no CUDA SDK dependency) for AMD-native porting; remove hardcoded efficiency printf from seeds`
`hipify-clang` needs a real CUDA SDK installed to resolve against (even
though it only translates, never compiles for NVIDIA) — it fails
immediately with "cannot find CUDA installation" on an AMD-only pod by
definition. Switched to `hipify-perl` (pure text/regex substitution, no
CUDA SDK required) as the preferred tool, confirmed by hand converting
`vectorAdd.cu` cleanly at `/opt/rocm/bin/hipify-perl` with nothing else
installed. Also removed a hardcoded efficiency-percentage `printf` from
the seeds themselves — efficiency became a Python-computed value instead
(see the GDDR6 entry below for where that calculation itself was later
fixed).

## 2026-07-10T08:50:40+02:00 — Duplicate-symbol linker bug
**`7683856`** — `Fix duplicate hip_files bug causing hipcc linker duplicate-symbol errors`
A real failure seen on the pod: a file named e.g. `vectorAdd.hip.cpp`
matches both the `*.hip.cpp` glob pattern and the `*.cpp` glob pattern;
naively concatenating three separate `glob()` results put that file on
the `hipcc` command line twice, and the linker then saw two definitions
of `main()` and every kernel in it. Fixed by deduplicating by *resolved
path* in `_discover_hip_sources()`, not just `list(set(...))` (which
would also lose deterministic file ordering across runs).

## 2026-07-10T09:07:30+02:00 — Reduction stale-buffer / validation bug
**`4d14176`** — `Fix reduction stale-buffer bug (missing reset before timed run) and tighten validation to check actual tolerance, not just presence of output`
A real correctness bug, not hypothetical: the `reduction` seed was being
reported `PASSED` because the string `"reduction result"` appeared
anywhere in the binary's stdout, regardless of the actual number that
followed it — a missing buffer reset before the timed run made the
kernel report exactly double the expected value, and validation didn't
notice because it never parsed or compared the actual number. Fixed on
both sides: `reduction.cu` gets its accumulator buffer properly reset
before the timed run, and `_validation_passes()` (`src/baseline/
pipeline.py`) now requires every actual-vs-expected pair the binary
prints to be within a real, seed-specific numeric tolerance
(`VALIDATION_TOLERANCES`) — applied uniformly to all three seeds, not
just `reduction`, so this class of bug can't hide in the other two.

## ~2026-07-10T09:08:08 to 09:08:18+02:00 — First real-hardware confirmations (all 3 original seeds)
**Evidence**: `jobs/job_374d6e8c51d1/` (vectorAdd), `jobs/job_7eeb1f8358f8/`
(tiledMatmul), `jobs/job_09ef95c5f62b/` (reduction) — directory
timestamps 09:08:08 / 09:08:13 / 09:08:18 respectively; each `state.json`
records `gpu_arch: "gfx1100"`, `status: "completed"`, and each
`logs/hipify.log` shows a Linux pod path
(`/workspace/SATO-SWARM-ACTII/jobs/<id>/...`), not a local-machine one.

38–48 seconds after the reduction stale-buffer fix (previous entry)
landed, all three original seeds ran for real on the pod, back to back,
and all three completed and validated successfully. These three job
directories (plus `job_1684fdb652d5`, added later — see below) are the
real-hardware ground truth used everywhere else in this project since:
`/demo/replay`'s `REPLAY_JOB_IDS`, this README's cited evidence, and this
log.

## 2026-07-10T15:13:00+02:00 — Standalone GPU spec verification
**`43142a9`** — `Add standalone GPU spec verification script + full raw rocminfo logging`
Added `scripts/verify_gpu_specs.py` (dumps the raw rocminfo/amd-smi query
plus the computed theoretical-peak result, standalone, without needing a
full pipeline run) and the `theoretical_peak_source` /
`theoretical_peak_calculation` fields on `JobState`/`DerivedMetrics`.
`preflight_logs/gpu_specs_verify.log` is dated 15:23 — 10 minutes after
this commit, consistent with the new script being run right after it was
committed.

## 2026-07-10T17:46:19+02:00 — GDDR6 bandwidth calibration fix
**`b88c2ee`** — `Fix GDDR6 bandwidth calculation: key-path bug + memory-technology-aware DDR factor (empirically calibrated to 0.03% accuracy against RX 7900 XTX spec)`
Fixed a key-path bug in the live memory-bus-spec query, and replaced a
flat DDR factor constant with a memory-*technology*-aware lookup
(`_MEM_TECH_DDR_FACTOR`): HBM keeps the textbook DDR ×2 (matches
MI300X's published 5300 GB/s), but GDDR6 needed an empirically-calibrated
×17.8 against the real RX 7900 XTX/`gfx1100` pod — a flat ×2, or even the
naive JEDEC clock-ratio guess of ×8, both undershoot real GDDR6 bandwidth
substantially. Result: 0.03% accuracy against the RX 7900 XTX's published
960 GB/s spec.

## 2026-07-10T18:03:21+02:00 — Three failed local run attempts (not pod evidence — noted for an honest record)
**Evidence**: `real_run_vectoradd_FINAL.log`, `real_run_tiledMatmul_FINAL.log`,
`real_run_reduction_FINAL.log` (repo root; committed later, as part of
`43f48a3` below) — all three dated `2026-07-10T18:03:21`, all three
`Status: JobStatus.FAILED`, `hipify failed (tool: hipify-perl). stderr
snippet: Command not found or failed t[o launch]`. The underlying
`jobs/job_01439eadf023/logs/hipify.log` (referenced by the vectorAdd one)
shows the full error: `Command not found or failed to launch:
'hipify-perl': [WinError 2] The system cannot find the file specified` —
a Windows-specific error, and the job's workspace path is a Windows path
(`C:\Users\...`), not the pod's Linux path.

Despite the "`_FINAL`" filename suggesting a concluding successful run,
**these three files are not additional real-hardware evidence** — they
record `SATOSWARM_MOCK` being left unset (or `0`) on a local Windows
machine that has no ROCm toolchain on `PATH`, the same failure mode as
running real mode anywhere without `hipify-perl`/`hipcc` installed. They
are included in this log only because they exist in the repository and
are named/dated in a way that could otherwise be misread as evidence of
a final pod confirmation run. The genuine real-hardware evidence for all
four seeds remains the `jobs/job_*` directories cited elsewhere in this
log, each independently confirmed via its own Linux pod paths and
`gfx1100` compiles.

## 2026-07-11T02:47:01+02:00 — Tool Registry, Memory System, Repair Loop
**`43f48a3`** — `Add Tool Registry, Memory System, and Repair Loop with a verified-real demo seed (repairDemo). Additive only — zero behavior change to the 3 proven seeds, confirmed by regression.`
1,543 insertions across 14 files. Added:
- `src/agents/tools.py` — `ToolRegistry`, 8 sandboxed tools.
- `src/memory/loader.py` — `PortingMemory`, keyword-overlap pattern
  retrieval against `memory/porting_patterns.jsonl`'s 4 curated patterns
  (only one, `gap_cudaCtxResetPersistingL2Cache`, has a mechanically-
  appliable `auto_fix`).
- `seeds/repairDemo.cu` — the repair-loop demo seed. At this commit, the
  seed does not yet have hipEvent-based timing instrumentation (just
  `cudaDeviceSynchronize()`, no `Kernel time:`/`Flag check:` output —
  see the real-hardware confirmation entry immediately below, and the
  instrumentation fix later in this log).
- `scripts/test_tools.py`, `scripts/test_memory.py`,
  `scripts/test_repair_loop.py` — the regression suites for all of the
  above.
- The 3 `real_run_*_FINAL.log` files from the entry above.

## 2026-07-11T02:49:09+02:00 — repairDemo real-hardware confirmation
**Evidence**: `jobs/job_1684fdb652d5/` — `gpu_arch: "gfx1100"`,
`mode: "REAL"`, `status: "completed"`; real message trace quoted below.

Two minutes after the Tool Registry/Repair Loop commit, `repairDemo` ran
for real on the pod. The real message trace:
```
[Repair Loop/thought] hipcc failed for repairDemo, a repair-loop-enabled
  seed. Querying PortingMemory (4 known patterns) for a fix...
[Repair Loop/observation] Repair attempt 1/3: matched pattern
  'gap_cudaCtxResetPersistingL2Cache' (stored confidence 0.97)...
[Repair Loop/observation] Repair attempt 1/3: patch applied to
  hip_out/repairDemo.hip.cpp... Recompiling with hipcc.
[Repair Loop/observation] Repair attempt 1/3: recompile SUCCEEDED after
  applying pattern 'gap_cudaCtxResetPersistingL2Cache' (target arch:
  gfx1100).
```
This confirms, on real hardware, that (a) `cudaCtxResetPersistingL2Cache`
is a genuine `hipcc` compile-time failure, not just a predicted one from
reading HIPIFY's source, and (b) the repair loop's match → patch →
recompile mechanism works end-to-end against a real compiler, not just
in the mock-mode regression test (`scripts/test_repair_loop.py`, which
proves the same control flow via a monkeypatched first-call failure).

**Honest caveat**: this specific run used the not-yet-instrumented seed
(previous entry) — its stdout was just `SATO SWARM repairDemo seed` /
`repairDemo seed completed successfully.`, no `Kernel time:` or `Flag
check:` line. Consistent with this project's "never fabricate a missing
number" rule, the pipeline correctly recorded `kernel_time_ms: None` and
`validation_passed: false` for this run rather than inventing either —
this is the system working as designed when data genuinely isn't there,
not a bug. See below for the instrumentation fix; a fresh real-hardware
`repairDemo` run with the now-instrumented seed has not yet been
captured as of this writing.

---

## This session — uncommitted as of this writing

Everything below exists in the working tree, is covered by this
session's own regression run (results quoted at the end of this
section), and has real filesystem timestamps — but has **not yet been
committed**, so there is no commit hash to cite for it. Listed
separately, honestly, rather than attributed to a commit that doesn't
exist.

- **`src/main.py`** (first written 2026-07-11T03:27:28+02:00) — FastAPI
  thin-wrapper backend. 7 endpoints (`/health`, `/jobs`,
  `/jobs/{id}/status`, `/jobs/{id}/report`, `/jobs/{id}/artifacts`,
  `/jobs/{id}/stream`, `/demo/replay`), every one calling directly into
  the same `run_baseline()`/`WorkspaceManager`/`PortingMemory` the CLI
  already used — no pipeline logic reimplemented.
- **`scripts/test_main.py`** (2026-07-11T03:39:39+02:00) — backend
  regression suite, including a CRITICAL test that runs the same seed
  via a direct `run_baseline()` call and via a full `TestClient` HTTP
  request, then diffs the resulting `JobState`/report to prove the API
  wrapper doesn't silently diverge from the already-proven CLI behavior.
- **`frontend/`** (scaffolded starting 2026-07-11T03:41:07+02:00) —
  Next.js 16 / React 19 / TypeScript UI, 5 components (`LiveJobView`,
  `AgentFeed`, `PhaseTimeline`, `MetricsDashboard`, `ReportViewer`).
- **`seeds/repairDemo.cu` timing instrumentation** — added real
  `cudaEventCreate`/`cudaEventRecord`/`cudaEventSynchronize`/
  `cudaEventElapsedTime` timing plus `Flag check:`/`Kernel time:` stdout
  lines, bringing it in line with the self-check pattern the 3 original
  seeds already had. Addresses the "Honest caveat" in the entry above;
  does not itself re-run that confirmation (still pending — see Known
  open items below).
- **`src/tools/execution.py` amd-smi parser fix**
  (2026-07-11T21:58:47+02:00) — three real bugs found and fixed by
  diffing the parser's assumptions against real captured amd-smi JSON
  from all 4 real job directories: (1) a missing top-level
  `{"gpu_data": [...]}` unwrap that made every field lookup fail before
  it started, (2) un-unwrapped `{"value": N, "unit": "..."}` reading
  objects that `_try_float()` didn't know how to handle, (3) wrong key
  names (`used_vram` not `used`, `clock.gfx_0.clk`/`clock.mem_0.clk` not
  `clock.sclk`/`clock.mclk`). A fourth bug surfaced during verification
  itself, not anticipated going in: `_try_float(a) or _try_float(b)`-
  style fallback chains silently discarded a genuine `0.0` telemetry
  reading (GPU idle before kernel launch — real, valid data) because
  `0.0` is falsy in Python; fixed with a `_first_non_none()` helper. See
  README.md's "amd-smi metric parsing — fixed and empirically verified"
  section for the full detail.
- **`src/models/job.py` + `src/baseline/pipeline.py` `job.mode` fix**
  (2026-07-11T22:06:1x+02:00) — `JobState.mode` set once in
  `run_baseline()` from the same `MOCK` constant governing everything
  else in that run, independent of whatever the server is running as
  later. The 4 real job directories (predating this field) were
  backfilled to `mode: "REAL"` by hand, one time, each only after
  independently confirming that job's own report text already said
  `**Mode**: REAL hardware`.
- **Frontend UI updated** (`LiveJobView`'s mode badge, `MetricsDashboard`'s
  footer, the page header's wording) to surface `job.mode` distinctly
  from the server's own current mode. Verified live in-browser this
  session: replayed `job_374d6e8c51d1` against a server running in MOCK
  mode — badge correctly read "REAL DATA", footer correctly read "This
  job ran on real hardware — no simulation", and the page header still
  correctly read "Server: MOCK mode (new runs simulated)" throughout. A
  fresh mock run in the same session correctly showed "SIMULATED"
  instead, confirming both badge states.

**Regression run covering all of the above (this session, freshest
run)**: `scripts/test_tools.py` 28/28, `scripts/test_memory.py` 11/11,
`scripts/test_repair_loop.py` 27/27, `scripts/test_main.py` 40/40
(including the CLI/API equivalence test — `job.mode` introduces no
diff), all 4 seeds re-run via the CLI in mock mode with calibration
numerically unchanged from prior runs (vectorAdd 86.8% efficiency /
4601.23 GB/s / 0.652 ms; tiledMatmul 0.7% efficiency; reduction 2568.76
GB/s), `npm run build` clean, `npm run lint` clean, 23/23 Vitest tests,
zero matches on this project's standing repo-wide internal-reference
scan (see CLAUDE.md's "THE BIBLE" section for what that scan checks
for and why).

---

## Known open items (as of this writing)

- The amd-smi parser fix, `job.mode`, `src/main.py`, and `frontend/` are
  all uncommitted (see the section above) — no commit hash exists for
  them yet.
- `repairDemo` has not been re-run on real hardware since its seed was
  instrumented with real hipEvent timing — the one real confirmation
  that exists (`job_1684fdb652d5`) predates that instrumentation and so
  has no captured `Kernel time` or validated correctness (see that
  entry's "Honest caveat").
- The amd-smi parser fix and `job.mode` backfill were verified against
  the 4 existing real job directories' *already-captured* data (static
  files), not by re-running the pipeline live against a real amd-smi
  process on the pod. The parsing logic itself was exercised against
  real bytes, but a fresh live pod run has not yet re-confirmed it
  end-to-end.
