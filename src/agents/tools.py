"""Tool Registry — the uniform surface an agent (or a future repair loop)
calls through to actually do things inside one job's workspace.

Foundational infrastructure only: this module is NOT wired into
src/baseline/pipeline.py yet, and nothing here changes any existing
verified pipeline behavior. Wiring it in (so an agent loop can call these
tools instead of the pipeline calling src/tools/execution.py directly) is
future work.

Every tool is reached through a single entry point:

    ToolRegistry(job_id, workspace_dir).execute(tool_name, **kwargs) -> dict

which ALWAYS returns {"success": bool, "result": ..., "error": Optional[str]}
— never raises, never returns a bare tuple/exception, regardless of which
tool was called or what went wrong. This is deliberate: an agent (or
whatever eventually drives one) should never have to wrap every tool call
in its own try/except to stay alive.

Two kinds of tools:
  - run_hipify / run_hipcc / capture_amd_smi / run_benchmark call directly
    into the REAL src/tools/execution.py functions -- this module never
    re-implements hipify/hipcc/amd-smi/binary-run logic, and never
    re-checks the SATOSWARM_MOCK switch itself. execution.py already owns
    that (see its MOCK flag docstring) and is the single source of truth.
  - read_file / apply_search_replace / list_workspace_files /
    write_agent_note are workspace-native: sandboxed strictly to this
    job's own workspace_dir (see _resolve()), never touching
    src/tools/execution.py at all.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.tools import execution


class ToolRegistry:
    """One job's sandboxed tool surface. Construct with the job's own
    workspace_dir (e.g. WorkspaceManager.get_workspace(job_id) /
    .create_workspace(job)) — every workspace-native tool (read_file,
    apply_search_replace, list_workspace_files, write_agent_note) refuses
    any path that resolves outside that directory, including via `..`
    traversal, an absolute-path override, or a symlink.
    """

    TOOL_NAMES: tuple[str, ...] = (
        "run_hipify",
        "run_hipcc",
        "capture_amd_smi",
        "run_benchmark",
        "read_file",
        "apply_search_replace",
        "list_workspace_files",
        "write_agent_note",
    )

    def __init__(self, job_id: str, workspace_dir: Path | str) -> None:
        self.job_id = job_id
        self.workspace_dir = Path(workspace_dir).resolve()
        if not self.workspace_dir.is_dir():
            raise FileNotFoundError(f"workspace_dir does not exist: {self.workspace_dir}")

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    def execute(self, tool_name: str, **kwargs) -> dict:
        """Dispatch to _tool_<tool_name>(**kwargs) and normalize the
        result to {"success", "result", "error"} no matter what happens:
        an unknown tool name, a bad/missing argument, a sandbox violation,
        or any other exception raised anywhere below all come back as a
        clean error dict rather than propagating.

        Individual _tool_* methods handle their OWN well-understood
        failure modes inline (e.g. apply_search_replace's "text not
        found" is an expected outcome, not a bug) and return the full
        dict themselves in those cases — the try/except here is a safety
        net for everything else (sandbox violations from _resolve(),
        wrong argument names/counts, unexpected errors from the
        underlying execution.py calls), not the primary error path.
        """
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None or not callable(handler):
            return {
                "success": False,
                "result": None,
                "error": f"Unknown tool {tool_name!r}. Available tools: {', '.join(self.TOOL_NAMES)}",
            }
        try:
            return handler(**kwargs)
        except TypeError as e:
            return {"success": False, "result": None, "error": f"Bad arguments for tool {tool_name!r}: {e}"}
        except Exception as e:
            return {"success": False, "result": None, "error": f"{type(e).__name__}: {e}"}

    # -------------------------------------------------------------------
    # Sandboxing
    # -------------------------------------------------------------------

    def _resolve(self, relative_path: Optional[str], must_exist: bool = True) -> Path:
        """Resolve relative_path against workspace_dir and refuse it if
        the resolved path is not actually inside workspace_dir.

        Checking the FULLY RESOLVED path (not the raw string) after the
        join is what catches all three escape shapes at once: `../..`
        traversal, an absolute path that pathlib's `/` operator would
        otherwise let silently replace the base entirely (e.g.
        `workspace_dir / "C:\\Windows\\System32"` resolves to
        `C:\\Windows\\System32`, not something under workspace_dir), and a
        symlink that points outside the sandbox. No path ever gets used
        without this check first.
        """
        if not relative_path:
            raise ValueError("relative_path is required and cannot be empty")
        candidate = (self.workspace_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(
                f"{relative_path!r} resolves outside this job's workspace ({self.workspace_dir}) — refused"
            )
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"{relative_path!r} does not exist in workspace")
        return candidate

    # -------------------------------------------------------------------
    # Tools backed by the REAL src/tools/execution.py -- never reimplemented
    # -------------------------------------------------------------------

    def _tool_run_hipify(self, source_dir: str, out_dir: str) -> dict:
        src = self._resolve(source_dir, must_exist=True)
        out = self._resolve(out_dir, must_exist=False)  # execution.run_hipify() creates it
        ok, log, err, tool_used = execution.run_hipify(src, out, self.job_id)
        return {
            "success": ok,
            "result": {"tool_used": tool_used, "log": log, "out_dir": out_dir},
            "error": None if ok else (err.strip() or "hipify failed"),
        }

    def _tool_run_hipcc(self, hip_sources: list[str], out_binary: str, arch: Optional[str] = None) -> dict:
        if not hip_sources:
            return {"success": False, "result": None, "error": "hip_sources is required and cannot be empty"}
        sources = [self._resolve(p, must_exist=True) for p in hip_sources]
        out = self._resolve(out_binary, must_exist=False)  # hipcc creates it
        ok, log, err, arch_used = execution.run_hipcc(sources, out, arch=arch)
        return {
            "success": ok,
            "result": {"arch_used": arch_used, "log": log, "out_binary": out_binary},
            "error": None if ok else (err.strip() or "hipcc failed"),
        }

    def _tool_capture_amd_smi(self) -> dict:
        metrics, raw_text = execution.capture_amd_smi_snapshot()
        return {
            "success": True,
            "result": {"metrics": metrics.model_dump(mode="json"), "raw_text": raw_text},
            "error": None,
        }

    def _tool_run_benchmark(self, binary: str, args: Optional[list[str]] = None, timeout: int = 120) -> dict:
        # must_exist=False deliberately: MOCK-mode run_hipcc() never
        # actually writes the compiled binary to disk (mock _run() only
        # returns a canned success message), the same way
        # src/baseline/pipeline.py already never pre-checks binary
        # existence before calling run_binary() -- it trusts hipcc's own
        # rc as the compile-success signal instead. In REAL mode, a
        # genuinely missing binary is already handled cleanly by
        # execution.py's _run() (FileNotFoundError -> rc=127, not a
        # crash), so no separate existence check is needed here either.
        binary_path = self._resolve(binary, must_exist=False)
        rc, stdout, stderr, wall = execution.run_binary(binary_path, args or [], timeout=timeout)
        return {
            "success": rc == 0,
            "result": {"returncode": rc, "stdout": stdout, "stderr": stderr, "wall_time_s": wall},
            "error": None if rc == 0 else f"binary exited with rc={rc}: {stderr.strip()[:300]}",
        }

    # -------------------------------------------------------------------
    # Workspace-native tools (no execution.py involved)
    # -------------------------------------------------------------------

    def _tool_read_file(self, relative_path: str) -> dict:
        path = self._resolve(relative_path, must_exist=True)
        if path.is_dir():
            return {"success": False, "result": None, "error": f"{relative_path!r} is a directory, not a file"}
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            return {"success": False, "result": None, "error": f"{relative_path!r} is not valid UTF-8 text: {e}"}
        return {"success": True, "result": {"relative_path": relative_path, "content": content}, "error": None}

    def _tool_apply_search_replace(self, relative_path: str, old_text: str, new_text: str) -> dict:
        """Find old_text in the file and replace it with new_text.

        Succeeds ONLY when old_text occurs in the file EXACTLY ONCE:
          - 0 occurrences -> fails cleanly ("not found"), file untouched.
          - 2+ occurrences -> fails cleanly ("not unique"), file untouched
            -- silently patching "the first match" would risk editing the
            wrong one of several near-identical spots (e.g. one of three
            CHECK_CUDA(cudaFree(...)) calls), which is worse than refusing
            and asking the caller to include more surrounding context to
            disambiguate. This is the actual patching primitive a future
            repair loop will depend on, so ambiguous edits must never
            silently "succeed" against the wrong occurrence.
        The file is only ever written on the exactly-one-match path.
        """
        if not old_text:
            return {"success": False, "result": None, "error": "old_text cannot be empty"}
        path = self._resolve(relative_path, must_exist=True)
        content = path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return {
                "success": False,
                "result": None,
                "error": f"old_text not found in {relative_path!r} — no change made",
            }
        if count > 1:
            return {
                "success": False,
                "result": None,
                "error": (
                    f"old_text is not unique in {relative_path!r} ({count} occurrences found) — "
                    f"no change made; include more surrounding context so it matches exactly one location"
                ),
            }
        new_content = content.replace(old_text, new_text, 1)
        path.write_text(new_content, encoding="utf-8")
        return {
            "success": True,
            "result": {
                "relative_path": relative_path,
                "bytes_before": len(content.encode("utf-8")),
                "bytes_after": len(new_content.encode("utf-8")),
            },
            "error": None,
        }

    def _tool_list_workspace_files(self, subdir: Optional[str] = None) -> dict:
        base = self._resolve(subdir, must_exist=True) if subdir else self.workspace_dir
        if not base.is_dir():
            return {"success": False, "result": None, "error": f"{subdir!r} is not a directory"}
        files = sorted(
            str(p.relative_to(self.workspace_dir)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        )
        return {"success": True, "result": {"files": files, "count": len(files)}, "error": None}

    def _tool_write_agent_note(self, note: str) -> dict:
        """Append a timestamped line to notes/blackboard.md -- one shared
        file every agent appends to (blackboard-style), never overwritten.
        WorkspaceManager.create_workspace() already creates notes/, but
        this also mkdir(exist_ok=True)s it so the tool is robust even
        against a workspace_dir that wasn't created that way.
        """
        if not note or not note.strip():
            return {"success": False, "result": None, "error": "note cannot be empty"}
        notes_dir = self.workspace_dir / "notes"
        notes_dir.mkdir(exist_ok=True)
        notes_path = notes_dir / "blackboard.md"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {note.strip()}\n"
        with notes_path.open("a", encoding="utf-8") as f:
            f.write(line)
        return {
            "success": True,
            "result": {
                "notes_path": str(notes_path.relative_to(self.workspace_dir)).replace("\\", "/"),
                "written": line.rstrip("\n"),
            },
            "error": None,
        }
