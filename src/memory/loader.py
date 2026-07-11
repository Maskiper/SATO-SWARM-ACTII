"""Porting pattern memory — a small, file-backed knowledge base of known
CUDA -> HIP porting gaps, keyed for keyword-overlap retrieval against a
real compiler/hipify error message.

Wired into src/baseline/pipeline.py's repair loop
(_attempt_hipcc_repair()), which calls get_relevant_patterns() with a real
hipcc error, and — for a matched pattern that has an 'auto_fix' field —
applies it mechanically via ToolRegistry.apply_search_replace(). Only
active for SeedId.REPAIR_DEMO; the 3 original seeds' pipeline.py flow is
unchanged (see that function's docstring).

File format: memory/porting_patterns.jsonl — one JSON object per line
(JSONL, not a single JSON array), so a new pattern can be appended without
rewriting the whole file (see add_pattern()). Patterns are plain dicts,
not a Pydantic model (unlike src/models/job.py) — the file is meant to be
easy to hand-write/curate directly, not schema-locked. By convention,
get_relevant_patterns() and get_context_for_agent() read these fields:

  id            short unique slug, e.g. "gap_cudaFuncGetName"
  cuda          the CUDA-side identifier(s) this pattern is about — THIS
                is what get_relevant_patterns() keyword-matches against
                error_text. Keep this IDENTIFIER-FOCUSED (just the bare
                CUDA symbol name, e.g. "cudaCtxResetPersistingL2Cache" —
                a second closely related symbol is fine too, space-
                separated) — do NOT pad it with descriptive English
                words. A camelCase/underscore identifier tokenizes as a
                SINGLE keyword (see _keywords()), and a real hipcc/clang
                error quotes that exact identifier verbatim and nothing
                else — extra descriptive words in 'cuda' only inflate the
                keyword denominator get_relevant_patterns() divides by,
                diluting the score for the one match that will actually
                happen. Put the human-readable description in
                'explanation' instead, which isn't used for matching.
  hip_fix       the concrete, actionable fix — HUMAN-READABLE PROSE,
                never applied mechanically (see auto_fix below)
  auto_fix      OPTIONAL. {"old_text": ..., "new_text": ...} — the exact,
                literal apply_search_replace() arguments for a repair
                loop to apply this fix MECHANICALLY, with zero
                interpretation of 'hip_fix' at repair time. Only present
                on patterns someone has explicitly turned into a
                machine-appliable fix (not automatic just because
                'hip_fix' exists) — a pattern without this field can
                still be matched/reported by get_relevant_patterns(), but
                a repair loop must skip it rather than guess a patch from
                the prose.
  category      short tag, e.g. "runtime_api_gap", or "confirmed_repair"
                for a pattern src/baseline/pipeline.py's repair loop
                itself added after empirically confirming a fix worked
  confidence    a STATIC 0-1 rating of how solid the evidence/fix is —
                independent of get_relevant_patterns()'s min_confidence
                argument, which filters a DIFFERENT, dynamically computed
                per-query match score (see that method's docstring)
  explanation   1-3 sentence human-readable "why this gap exists"
  source        human-readable citation (repo/file/line, or doc name)
  source_url    a real, checkable URL where available
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


class PortingMemory:
    """File-backed CUDA->HIP porting pattern memory.

    Patterns are loaded once at construction (self._patterns, in file
    order) and kept in memory; add_pattern() updates both the in-memory
    list and (if persist=True) appends to the backing file — never
    rewrites it. There is no separate reload method: construct a new
    PortingMemory if the file changed on disk since this instance loaded.
    """

    def __init__(self, path: Path | str = Path("memory/porting_patterns.jsonl")) -> None:
        self.path = Path(path)
        self._patterns: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        self._patterns = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._patterns.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.path}:{line_num}: invalid JSON line: {e}") from e

    @staticmethod
    def _keywords(text: str) -> set[str]:
        """Lowercased identifier-like tokens, 3+ chars — drops noise like
        single-/double-letter tokens ('a', 'if', 'ms'). CUDA/HIP API
        names are the actual signal here, and a real compiler error
        quotes them verbatim, so no stemming or fuzzy matching is
        attempted — plain substring/token overlap is the right tool.
        """
        return set(re.findall(r"[A-Za-z_]\w{2,}", text.lower()))

    def get_relevant_patterns(
        self, error_text: str, top_k: int = 5, min_confidence: float = 0.65
    ) -> list[dict[str, Any]]:
        """Return up to top_k patterns whose 'cuda' field overlaps
        error_text, sorted by match score descending.

        match score = |keywords(error_text) & keywords(pattern['cuda'])| / |keywords(pattern['cuda'])|
        — i.e. what FRACTION OF THIS PATTERN's own identifying keywords
        actually appear in error_text. Normalizing by the pattern's own
        (short) keyword count, not error_text's (usually much longer and
        full of irrelevant tokens — file paths, line numbers, surrounding
        code), is what lets a real, verbose hipcc error message still
        score close to 1.0 against a short, focused pattern whose exact
        API name it happens to quote.

        min_confidence filters on THIS computed match score — "how
        confident are we this specific pattern is relevant to this
        specific error" — not on a pattern's own static 'confidence'
        field (see the module docstring), which is a different, unrelated
        number. A pattern with no 'cuda' field (or an empty one) never
        matches — score 0, excluded regardless of min_confidence.
        """
        error_kw = self._keywords(error_text)
        scored: list[tuple[float, dict[str, Any]]] = []
        for pattern in self._patterns:
            pattern_kw = self._keywords(pattern.get("cuda", ""))
            if not pattern_kw:
                continue
            score = len(error_kw & pattern_kw) / len(pattern_kw)
            if score >= min_confidence:
                scored.append((score, pattern))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [pattern for _, pattern in scored[:top_k]]

    def add_pattern(self, pattern: dict[str, Any], persist: bool = True) -> None:
        """Add a pattern to the in-memory list, and (by default) append
        it to the backing JSONL file as one new line — never rewrites the
        file, so this is safe even while other patterns already exist on
        disk. persist=False stages a pattern in memory only, without
        touching disk (e.g. for testing).
        """
        self._patterns.append(pattern)
        if persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(pattern) + "\n")

    def get_context_for_agent(self) -> str:
        """Format every pattern currently in memory as readable text,
        suitable for prepending to an agent's prompt as background
        context. No filtering or ranking here (get_relevant_patterns() is
        the query-specific entry point for that) — this is the general
        "here's everything currently known" dump.
        """
        if not self._patterns:
            return "No known porting patterns in memory."
        blocks = []
        for pattern in self._patterns:
            lines = [f"- {pattern.get('cuda', pattern.get('id', 'unknown pattern'))}"]
            if pattern.get("explanation"):
                lines.append(f"  Why: {pattern['explanation']}")
            if pattern.get("hip_fix"):
                lines.append(f"  Fix: {pattern['hip_fix']}")
            if pattern.get("source"):
                lines.append(f"  Source: {pattern['source']}")
            blocks.append("\n".join(lines))
        return "Known CUDA -> HIP porting gaps:\n\n" + "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self._patterns)
