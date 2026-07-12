# SATO SWARM ports CUDA to AMD ROCm, runs it on real GPU hardware, and fixes its own compile failures — autonomously.

*SWARM = Swarm WorkForce Autonomous ReFActoring Migration*

## Elevator Pitch (15 seconds)

> "SATO SWARM ports CUDA kernels to AMD ROCm, compiles and runs them on real GPU hardware, and measures real GPU-side timing — never a simulated number presented as real. Its Repair Loop is the one genuinely autonomous step: when hipcc fails on a known porting gap, it patches the translated source and recompiles itself, and we have a real hardware run where exactly that happened, unattended."

## 90-Second Verification Path

**Fastest — replay real captured hardware data (no compile, no wait):**
```bash
$env:SATOSWARM_MOCK = "1"
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
# in another terminal:
curl.exe -X POST "http://127.0.0.1:8000/demo/replay?seed_id=vectorAdd"
```
Returns the actual `JobState` + report from a real, already-completed pod run (`job_374d6e8c51d1`) — never runs anything new, works even though the server above is in MOCK mode. `"mode": "REAL"` and `"gpu_arch": "gfx1100"` are in the response body.

**Local mock path (no AMD hardware needed, full pipeline logic exercised):**
```bash
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST "http://127.0.0.1:8000/jobs?seed_id=vectorAdd"
# copy the returned job_id
curl.exe http://127.0.0.1:8000/jobs/<job_id>/status
curl.exe http://127.0.0.1:8000/jobs/<job_id>/report
```
Completes in under a second. Every number in the report is tagged `(SIMULATED)` — this path proves the pipeline logic, not hardware.

**Real hardware path (needs an actual ROCm-capable AMD GPU):**
```bash
bash scripts/preflight.sh              # toolchain check + a real trivial HIP compile/run
python scripts/test_baseline.py vectorAdd
```
`preflight.sh` fails fast and specifically if `hipify-perl`/`hipcc`/`amd-smi` aren't on `PATH`, before spending time on a full run.

## Claims Table

| Claim | Verify in 90s | Evidence |
|---|---|---|
| Ran on real AMD GPU hardware, not simulated | `POST /demo/replay?seed_id=vectorAdd`, check `"mode": "REAL"` | `jobs/job_374d6e8c51d1/state.json` (`mode: "REAL"`, `gpu_arch: "gfx1100"`); `jobs/job_374d6e8c51d1/logs/hipify.log` shows a real pod path (`/workspace/SATO-SWARM-ACTII/jobs/...`), not a local one |
| Kernel timing is a real GPU-side hipEvent measurement, not a wall-clock guess | Read the replayed report's `Kernel time (hipEvent)` row | `jobs/job_374d6e8c51d1/reports/migration_report.md` — `4.315 ms`, sourced from the binary's own `hipEventElapsedTime()`; every seed in `seeds/*.cu` wraps its kernel launch in real `cudaEventCreate`/`Record`/`Synchronize`/`ElapsedTime` calls |
| The Repair Loop is a genuine autonomous step, not scripted output | `POST /demo/replay?seed_id=repairDemo`, read the message trace | `jobs/job_1684fdb652d5/state.json` — `hipcc` genuinely fails on `cudaCtxResetPersistingL2Cache`, `Repair Loop` matches pattern `gap_cudaCtxResetPersistingL2Cache`, patches the real file via `apply_search_replace`, recompiles, **succeeds** — `gpu_arch: gfx1100`, real pod, real hipcc, `Final Status: COMPLETED` |
| Metrics are never fabricated — genuinely absent data says so, even when that's less impressive | Read any replayed report for `Not captured` / `Not applicable` | `jobs/job_374d6e8c51d1/reports/migration_report.md`: `"efficiency % of theoretical unknown peak for gfx1100 not applicable for this seed"` — captured before the GDDR6 calibration fix existed; the report says so honestly rather than a number being backfilled after the fact |
| The FastAPI wrapper doesn't silently diverge from the CLI's already-proven behavior | `python scripts/test_main.py` | `scripts/test_main.py`'s CRITICAL section (line 7) — runs the same seed via a direct `run_baseline()` call and via a full `TestClient` HTTP request, diffs the resulting `JobState` and generated report byte-for-byte (modulo job_id/timestamps/paths) |
| GPU architecture is auto-detected every run, never hardcoded | Compare `gpu_arch` across all 4 real job dirs | All 4 real captures show `gpu_arch: "gfx1100"` — the pod was provisioned expecting an MI300X (`gfx942`) and came back RDNA3 instead; `src/tools/execution.py`'s `detect_gpu_arch()` is why every real report still shows the correct architecture instead of a segfaulting binary |

## Demo/replay endpoints

`POST /demo/replay?seed_id=<id>` returns `{"job": <JobState>, "report_md": <string>}` from an **actual, already-completed real-hardware job** — it never runs anything new, and 404s with the exact expected path if that job directory isn't present in the checkout, rather than substituting mock data. Four seeds have real captured data (`REPLAY_JOB_IDS` in `src/main.py`):

| seed_id | Real job ID | What it proves |
|---|---|---|
| `vectorAdd` | `job_374d6e8c51d1` | Memory-bandwidth-bound kernel, real hipEvent timing + achieved bandwidth |
| `tiledMatmul` | `job_7eeb1f8358f8` | Compute-bound kernel, real achieved TFLOPS |
| `reduction` | `job_09ef95c5f62b` | Control-flow/atomics kernel |
| `repairDemo` | `job_1684fdb652d5` | The Repair Loop firing for real (see Claims Table above) |

## Architecture (as it exists in this repo right now)

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
- **Baseline pipeline** (`src/baseline/pipeline.py`): the core port→validate→benchmark→report flow, real for all seeds.
- **Tool Registry** (`src/agents/tools.py`): 8 sandboxed tools; wired into the pipeline today for exactly the Repair Loop's `apply_search_replace`/`run_hipcc` calls.
- **Memory** (`src/memory/loader.py` + `memory/porting_patterns.jsonl`): 7 curated CUDA→HIP porting gaps, keyword-matched against real hipcc errors.
- **Repair Loop** (`_attempt_hipcc_repair()` in `pipeline.py`): only active for `repairDemo`; the 4 other seeds are byte-for-byte untouched by its existence.
- **FastAPI** (`src/main.py`): thin wrapper, proven non-divergent from the CLI (Claims Table above).
- **Frontend** (`frontend/`): Next.js/React UI over the same API.
- *Recently added, mock-verified only as of this writing, real-hardware confirmation still pending*: multi-file project compilation (`seeds/multiFileDemo/`), CUDA library detection + ROCm link-flag auto-selection (`detect_cuda_library_includes()`), and an unrecognized-output-format report fallback.

## If you hit a failure on your own hardware

A compile failure is still a **complete, inspectable run** — `run_baseline()` never early-returns on a FAILED compile; it always finishes the pipeline and produces a diagnostic report + full logs + artifacts tar. For any job directory `jobs/<job_id>/`:
- `logs/hipify.log` — exact hipify-perl (or hipify-clang) command + stdout/stderr. Start here if the report says `hipify failed`.
- `logs/hipcc.log` — the exact `hipcc` command line and full compiler diagnostic output. Start here if `hipcc failed` — the report's `Migration Notes` section also quotes the reproducible command.
- `logs/run.log` — the binary's raw stdout/stderr + process wall-clock, if it got that far.
- `reports/migration_report.md` — always generated, `**Final Status**: FAILED` clearly marked, with the exact `Executive Summary` explaining which step failed.
- `reports/<job_id>_artifacts.tar.gz` — everything above, bundled, for offline inspection.

## Reproducibility

```bash
git clone https://github.com/Maskiper/SATO-SWARM-ACTII.git
cd SATO-SWARM-ACTII
pip install -r requirements.txt
bash scripts/preflight.sh
python scripts/test_baseline.py vectorAdd
python scripts/test_baseline.py tiledMatmul
python scripts/test_baseline.py reduction
python scripts/test_baseline.py repairDemo
```
Full copy-paste sequence with exactly what to save as proof: **RUNBOOK.md**.

## Honest limitations

- **amd-smi's JSON schema varies by ROCm version.** It's already been wrong once and been fixed (three real structural bugs, see README.md's "amd-smi metric parsing" section) against real captured output from all 4 real job dirs — a different ROCm release could expose a field under a different key again. The raw `amd-smi` text is always saved regardless, so nothing is ever lost even when the parser misses a field.
- **The GDDR6 bandwidth factor (`×17.8`) is empirically calibrated against exactly one confirmed card** (RX 7900 XTX / gfx1100, 0.03% accuracy against its published spec) — not a textbook constant, and not yet cross-validated against a second GDDR6 card with a different bus width/clock combination.
- **The 3 real benchmark job reports shipped in this repo predate that calibration fix** — they show real, measured achieved bandwidth/TFLOPS, but "Not applicable" for efficiency-vs-theoretical-peak, because the live theoretical-peak query wasn't returning a usable value yet when they were captured. A fresh real run today would compute it live; these specific historical captures correctly don't claim a number they didn't have.
- **The Repair Loop currently knows one real, mechanically-fixable gap** (`cudaCtxResetPersistingL2Cache`). The other 6 patterns in `memory/porting_patterns.jsonl` are researched, cited, real CUDA→HIP gaps — but prose-only (`hip_fix` text for a human), not wired to `apply_search_replace` yet.
- **The real `repairDemo` capture (`job_1684fdb652d5`) predates that seed's own timing instrumentation** — it proves the compile-failure-and-repair sequence is real (see Claims Table), but that specific run has no captured kernel time or correctness validation, honestly labeled `Not captured` rather than backfilled. The seed has since been instrumented; a fresh real run hasn't been re-captured against it yet.
