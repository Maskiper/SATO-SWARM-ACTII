"""Workspace isolation for jobs.

Per job layout:
  /jobs/<job_id>/
    cuda_src/     # original .cu copied from seeds/
    hip_out/      # after hipify + hipcc artifacts
    logs/         # structured logs (hipify, hipcc, run, amd-smi raw text)
    metrics/      # amd-smi json dumps + timeseries
    reports/      # migration_report.md + artifacts tar
    notes/        # blackboard for future agents
    state.json    # authoritative JobState snapshot
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from src.models.job import JobState, SeedId


class WorkspaceManager:
    def __init__(self, base_dir: Path | str = Path("./jobs")):
        self.base = Path(base_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, job: JobState) -> Path:
        ws = self.base / job.job_id
        for sub in ("cuda_src", "hip_out", "logs", "metrics", "reports", "notes"):
            (ws / sub).mkdir(parents=True, exist_ok=True)

        # Persist initial state
        self.write_state(job, ws)
        return ws

    def get_workspace(self, job_id: str) -> Path:
        return self.base / job_id

    def write_state(self, job: JobState, ws: Optional[Path] = None) -> None:
        ws = ws or self.get_workspace(job.job_id)
        state_path = ws / "state.json"
        # Use model_dump for Pydantic v2
        state_path.write_text(json.dumps(job.model_dump(mode="json"), indent=2, default=str))

    def load_state(self, job_id: str) -> Optional[JobState]:
        ws = self.get_workspace(job_id)
        state_path = ws / "state.json"
        if not state_path.exists():
            return None
        data = json.loads(state_path.read_text())
        return JobState.model_validate(data)

    def copy_seed(self, job: JobState, seeds_root: Path) -> Path:
        """Copy the appropriate self-contained seed .cu into cuda_src/."""
        ws = self.get_workspace(job.job_id)
        cuda_src = ws / "cuda_src"
        cuda_src.mkdir(exist_ok=True)

        seed_file = seeds_root / f"{job.seed_id.value}.cu"
        if not seed_file.exists():
            # Fallback for different naming in UI vs disk
            mapping = {
                SeedId.VECTOR_ADD: "vectorAdd.cu",
                SeedId.TILED_MATMUL: "tiledMatmul.cu",
                SeedId.REDUCTION: "reduction.cu",
            }
            seed_file = seeds_root / mapping.get(job.seed_id, f"{job.seed_id.value}.cu")

        if not seed_file.exists():
            raise FileNotFoundError(f"Seed source not found: {seed_file}")

        dest = cuda_src / seed_file.name
        shutil.copy2(seed_file, dest)
        return dest

    def create_tar(self, job: JobState) -> Path:
        """Create a tar.gz of the entire workspace for download."""
        ws = self.get_workspace(job.job_id)
        tar_path = ws / "reports" / f"{job.job_id}_artifacts.tar.gz"
        shutil.make_archive(str(tar_path).replace(".tar.gz", ""), "gztar", ws)
        # make_archive adds .tar.gz automatically
        final = Path(str(tar_path).replace(".tar.gz", "") + ".tar.gz")
        return final

    def cleanup_old_jobs(self, max_age_hours: int = 24, max_jobs: int = 50) -> int:
        """Light cleanup for the instance (keeps disk usage reasonable)."""
        import time
        now = time.time()
        count = 0

        job_dirs = sorted(self.base.glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True)

        for d in job_dirs[max_jobs:]:
            try:
                shutil.rmtree(d)
                count += 1
            except Exception:
                pass

        for d in job_dirs:
            if now - d.stat().st_mtime > max_age_hours * 3600:
                try:
                    shutil.rmtree(d)
                    count += 1
                except Exception:
                    pass
        return count
