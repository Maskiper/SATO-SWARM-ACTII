# SATO SWARM — Pod Deployment Runbook

Exact copy-paste sequence for taking this project from a clean checkout to
a verified real-hardware run, in order. Every command below is meant to be
run as-is (adjust only the two things explicitly marked `<...>`).

---

## Before you start: getting the code onto the pod

**This repo has not been pushed anywhere yet.** `git` was not available on
the machine this project was built on (checked, not found on PATH or in
common install locations), so no `git init` / commit / remote has happened
for this specific clean project. Pick one:

**Option A — push to GitHub first (recommended, matches the clone flow below)**
```bash
# On a machine that has git:
cd sato-swarm-actii
git init
git add .
git commit -m "Initial commit: clean SATO SWARM core pipeline"
git branch -M main
git remote add origin <YOUR_NEW_GITHUB_REPO_URL>
git push -u origin main
```
Double-check `.gitignore` is doing its job before this push — run
`git status` and confirm `.env` / `agent_config.yaml` / `jobs/` /
`preflight_logs/` do **not** appear in the file list about to be committed.

**Option B — skip git, copy the folder directly to the pod**
```bash
# From your local machine, replace <POD_HOST> with the pod's address:
rsync -avz --exclude 'jobs/' --exclude 'preflight_logs/' --exclude '__pycache__' \
  ./sato-swarm-actii/ <POD_HOST>:~/sato-swarm-actii/
```
If you use Option B, skip the `git clone` step below and just `cd` into
the directory you copied.

---

## 1. Clone the repo (on the pod)

```bash
git clone <YOUR_NEW_GITHUB_REPO_URL> sato-swarm-actii
cd sato-swarm-actii
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```
This installs `pydantic` — the only runtime dependency. Nothing else to
set up on the Python side.

## 3. Run preflight — confirm the toolchain before trusting it with anything

```bash
bash scripts/preflight.sh
```

Read the summary at the bottom. It checks `hipcc`, `hipify` (hipify-perl
preferred, hipify-clang fallback), `amd-smi`, `rocprofv3` (presence +
version), `rocminfo` (GPU visible — whatever architecture it actually
reports, gfx942/gfx1100/whatever, not assumed in advance), and actually
compiles + runs a trivial HIP kernel *for that detected architecture* to
prove the whole chain works end to end — not just that the binaries exist
on PATH.

- **All PASS** → continue to step 4.
- **Any FAIL** → stop here. Check `preflight_logs/` for the specific
  compile/run/rocminfo output that explains why, fix the environment
  (missing module load, wrong ROCm version, etc.), and re-run this step.
  Do not proceed to a real pipeline run with a failing preflight — you'll
  just get a `FAILED` job with "command not found," which preflight
  already told you more precisely.
- **`hipify` fails specifically with "cannot find CUDA installation"** →
  that's hipify-clang being used without a CUDA SDK present (it needs one
  to parse source even though it's only translating, not compiling for
  NVIDIA). Install `hipify-perl` instead — pure text substitution, no CUDA
  SDK required, and the tool this pipeline actually prefers. It's usually
  already sitting at `/opt/rocm/bin/hipify-perl` alongside the rest of the
  ROCm toolchain; check it's on `PATH`.
- **`hip_hello_world` fails with rc=139 specifically** → that's a segfault,
  and it almost always means an architecture mismatch: the pod provisioned
  a different GPU than expected (this has actually happened — an MI300X
  request came back as an RDNA3/gfx1100 card instead). Check the PASS/FAIL
  line for `rocminfo` above it in the same run — it names the architecture
  that was actually detected and used. The pipeline itself doesn't have
  this problem (architecture is auto-detected the same way at compile
  time, every run), but if you see this it's worth confirming preflight's
  detected architecture matches what you expect before proceeding.

## 4. Run the pipeline in MOCK mode first

Confirms the pipeline logic itself is intact on this checkout, independent
of hardware — a fast sanity check before spending time on the real run.

```bash
export SATOSWARM_MOCK=1
python scripts/test_baseline.py vectorAdd
python scripts/test_baseline.py tiledMatmul
python scripts/test_baseline.py reduction
unset SATOSWARM_MOCK
```

Expect `Status: JobStatus.COMPLETED` for all three, with every metric in
the printed report tagged `(SIMULATED)`. If any of these fail, something
is wrong with the checkout itself (bad clone/transfer, missing
`requirements.txt` install) — resolve before moving to real mode.

## 5. Run the pipeline for real

`SATOSWARM_MOCK` is unset from the step above — real mode is the default,
so nothing else to set.

```bash
python scripts/test_baseline.py vectorAdd
```

This is the one that matters. Watch for `Status: JobStatus.COMPLETED` and
a `Kernel time: X.XXX ms (hipEvent)` line — that confirms hipify (perl or
clang, whichever was auto-selected — check the report's hipify command
line to see which) and hipcc actually ran, the binary actually executed
on the GPU, and the timing is a real `hipEventElapsedTime()` reading, not
a placeholder.

Once vectorAdd succeeds, run the other two for a fuller picture:
```bash
python scripts/test_baseline.py tiledMatmul
python scripts/test_baseline.py reduction
```

---

## What to save as proof of the real run

Each run creates `jobs/<job_id>/` (the job ID is printed at the start of
`test_baseline.py`'s output, and in the report's `**Job ID**:` line). Save
the whole directory, or at minimum these files:

| What you asked for | Exact file |
|---|---|
| Ported HIP source | `jobs/<job_id>/hip_out/vectorAdd.hip.cpp` (hipify's real output) |
| Compiled binary | `jobs/<job_id>/hip_out/vectorAdd_hip` |
| Compile logs | `jobs/<job_id>/logs/hipify.log` and `jobs/<job_id>/logs/hipcc.log` |
| Metrics JSON | `jobs/<job_id>/state.json` (full JobState incl. `metrics.raw` + `metrics.derived` — this *is* the metrics JSON) |
| amd-smi output | `jobs/<job_id>/logs/amd_smi_pre.txt` and `jobs/<job_id>/logs/amd_smi_post.txt` (raw text, saved regardless of whether the parser understood every field) |
| Theoretical peak query trace | `jobs/<job_id>/logs/gpu_specs.log` (exact rocminfo/amd-smi output + the bandwidth/TFLOPS calculation performed on it — see below) |
| Run output | `jobs/<job_id>/logs/run.log` (binary's real stdout/stderr, incl. the `Kernel time:` line) |
| Human-readable report | `jobs/<job_id>/reports/migration_report.md` |
| Everything bundled | `jobs/<job_id>/reports/<job_id>_artifacts.tar.gz` — all of the above in one file |

Also save the preflight evidence, independent of any job:
- `preflight_logs/rocminfo.log` — proves the GPU was visible before you ran anything
- `preflight_logs/hello_compile.log` + `preflight_logs/hello_run.log` — proves the toolchain worked end to end before you trusted it with the real seeds

## Pulling artifacts back to your machine

```bash
# From your local machine:
scp <POD_HOST>:~/sato-swarm-actii/jobs/<job_id>/reports/<job_id>_artifacts.tar.gz .
scp -r <POD_HOST>:~/sato-swarm-actii/preflight_logs .
```

---

## If a metric shows "Not captured" on the real run

Expected for `Power (avg/peak W)` / `Utilization` / `Temperature` if
`amd-smi metric --json`'s exact field names on this ROCm version don't
match the parser's candidate paths in
`src/tools/execution.py`'s `_parse_amd_smi_json()` — this is a known,
documented limitation (see `README.md`), not a bug to chase blindly.
Compare `jobs/<job_id>/logs/amd_smi_post.txt` (the real raw output) against
the field paths that function tries, and adjust them if needed. Kernel
time, achieved bandwidth, and achieved TFLOPS are **not** affected by this
— those come from the binary's own stdout, independent of amd-smi.

If `Kernel time` itself shows "Not captured," that's more serious — it
means the binary ran but didn't print a `Kernel time:` line, which
shouldn't happen if compilation and preflight both succeeded. Check
`jobs/<job_id>/logs/run.log` for the actual raw output to see what the
binary printed instead.

## If the efficiency percentage shows "Not applicable"

`achieved_bw_gbs` and `achieved_tflops` are always real, measured numbers
when present; only the percentage-of-theoretical-peak can be withheld.

Theoretical peak (bandwidth + FP32 TFLOPS) is computed live on every real
run by `src/tools/execution.py`'s `detect_gpu_theoretical_peaks()` —
querying `rocminfo` (compute units, max engine clock) and `amd-smi` (max
memory clock, memory bus width) from whatever GPU is actually attached,
and deriving both numbers from first principles. This works the same on
gfx942, gfx1100, or any future architecture — there's no per-SKU table to
keep updating.

The report's **Theoretical peak calculation** line (and
`jobs/<job_id>/logs/gpu_specs.log`, which has the full raw
rocminfo/amd-smi output the calculation was built from) shows exactly
which path a given run took:
- **`runtime`** — the normal case: computed live from this GPU's own
  queried specs.
- **`fallback_table`** — the live query didn't return a usable value
  (e.g. `amd-smi`'s memory-bus-width field isn't recognized on this ROCm
  version — its exact JSON schema varies by release, same caveat as
  `_parse_amd_smi_json`) but the detected architecture has a small
  verified fallback entry (currently just MI300X/`gfx942`). Compare
  `logs/gpu_specs.log` against the key paths tried in
  `_query_amd_smi_mem_bus_specs()` / the rocminfo parsing in
  `detect_gpu_theoretical_peaks()`, and fix the key names there if the
  live query should have worked.
- **`unavailable`** ("Not applicable" in the report) — neither the live
  query nor the fallback table could produce a number for this
  architecture. `logs/gpu_specs.log` shows exactly what was tried and
  what came back, for debugging.

---

## Running the local stack (backend + frontend) — NOT a pod deployment

Everything above this line is the pod deployment sequence: get onto real
AMD GPU hardware, run the CLI (`scripts/test_baseline.py`), capture proof
files. **This section is a completely separate, local-only workflow** —
a FastAPI backend (`src/main.py`) + Next.js frontend (`frontend/`) that
run on your own machine (Windows, Mac, whatever — no ROCm/AMD GPU
required if run in MOCK mode) and give you a browser UI for watching a
job run live instead of reading CLI output.

**This stack is not deployed to the pod, and deploying it there is
neither currently supported nor needed.** The pod's job is to run the
actual CUDA -> HIP pipeline and produce real measurements; this local
stack is a viewer/demo layer on top of that, and it's equally useful
pointed at a fresh MOCK-mode local run or at the 4 real job directories
already pulled back from the pod (via **Replay real run**, below),
without the pod itself needing to be reachable at all.

### 1. Start the backend

```powershell
cd sato-swarm-actii
pip install -r requirements.txt      # now also installs fastapi + uvicorn, not just pydantic
$env:SATOSWARM_MOCK = "1"            # or unset/0 if THIS machine has ROCm + a real GPU on it
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Confirm it's up: `curl.exe http://127.0.0.1:8000/health` should return
JSON with `"system": "SATO SWARM"` and the mode you just set.

### 2. Start the frontend

In a second terminal:

```powershell
cd sato-swarm-actii/frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The page's health line at the top shows
the **backend's current mode** — this describes what a *new* run would
use, not any specific job you might view below it (see README.md's
`job.mode` section for why those two are deliberately independent, and
can legitimately disagree).

### 3. What you can do from the UI

- **Run** a seed — starts a brand-new job against whatever mode the
  backend is currently running as. Same `run_baseline()` the CLI itself
  uses under the hood — nothing pipeline-side is reimplemented for the
  API path (see README.md's FastAPI section, and `scripts/test_main.py`'s
  CLI/API equivalence test).
- **Replay real run** — loads one of the 4 actual, already-completed
  real-hardware job directories (the ones pulled back from the pod via
  the `scp`/`rsync` steps earlier in this document) through
  `/demo/replay`. Works even when the backend itself is running in MOCK
  mode, and never runs anything new — it only reads the
  `jobs/<job_id>/state.json` + `migration_report.md` that already exist
  in this checkout. If a given seed's real job directory isn't present
  here, the UI surfaces the same plain 404 message `/demo/replay` itself
  returns, rather than silently substituting mock data.

### Ports and CORS (local-only, by design)

The backend binds `127.0.0.1:8000` only (never `0.0.0.0`), and its CORS
allowlist is hardcoded to `localhost:3000`/`127.0.0.1:3000` — the default
Next.js dev server port (the `allow_origins=[...]` list passed to
`CORSMiddleware` in `src/main.py`). If you need a different frontend
port, either free up 3000 first or edit that list directly — left
un-generalized on purpose, since this whole stack is local-only by
design (see `src/main.py`'s own module docstring). It is not intended to
be exposed beyond this machine as-is.
