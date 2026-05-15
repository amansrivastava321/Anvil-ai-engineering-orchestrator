"""
Persistent repository state store.

Remembers every analysis across server restarts. Stores one JSON file per repo
in data/repo_states/. No AI involved — pure git + filesystem operations.

Token cost: 0. Everything here runs before any LLM is called.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Models ─────────────────────────────────────────────────────────────────────


class HealthSnapshot(BaseModel):
    """Point-in-time health measurement for a repository."""
    date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    health_score: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    total_files: int = 0
    total_modules: int = 0
    test_coverage: Optional[str] = None          # e.g. "67%"
    commit_hash: Optional[str] = None
    files_changed_since_previous: List[str] = Field(default_factory=list)
    new_issues_since_previous: int = 0
    fixed_issues_since_previous: int = 0


class RepoState(BaseModel):
    """Persistent state for a single repository."""
    repo_path: str
    repo_name: str
    first_analyzed: datetime
    last_analyzed: datetime
    analysis_count: int = 1
    last_commit_analyzed: Optional[str] = None   # git commit hash
    current_health: HealthSnapshot
    history: List[HealthSnapshot] = Field(default_factory=list)

    @property
    def trend(self) -> str:
        """Return 'improving', 'stable', or 'declining' based on last two snapshots."""
        if len(self.history) < 2:
            return "stable"
        prev = self.history[-2].health_score
        curr = self.history[-1].health_score
        delta = curr - prev
        if delta >= 3:
            return "improving"
        if delta <= -3:
            return "declining"
        return "stable"

    @property
    def trend_arrow(self) -> str:
        t = self.trend
        return "↑" if t == "improving" else "↓" if t == "declining" else "→"

    @property
    def improvement_rate(self) -> str:
        """Human-readable rate like '+2 points per week'."""
        if len(self.history) < 2:
            return "first analysis"
        first = self.history[0]
        last = self.history[-1]
        delta = last.health_score - first.health_score
        days = max(1, (last.date - first.date).days)
        rate_per_week = delta / days * 7
        sign = "+" if rate_per_week >= 0 else ""
        return f"{sign}{rate_per_week:.1f} points/week"

    @property
    def needs_attention(self) -> bool:
        return (
            self.current_health.critical_issues > 0
            or self.trend == "declining"
            or self.current_health.health_score < 60
        )

    def total_issues(self) -> int:
        h = self.current_health
        return h.critical_issues + h.high_issues + h.medium_issues + h.low_issues

    def days_since_analysis(self) -> int:
        return (datetime.now(timezone.utc) - self.last_analyzed).days


# ── Store ──────────────────────────────────────────────────────────────────────


class RepoStateStore:
    """JSON-backed store for repository states. One file per repo."""

    def __init__(self, data_dir: str | Path = "data/repo_states") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, RepoState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def save_state(self, repo_path: str, snapshot: HealthSnapshot) -> RepoState:
        """Create or update state for a repo. Returns the saved RepoState."""
        key = self._key(repo_path)
        existing = self.get_state(repo_path)
        now = datetime.now(timezone.utc)

        if existing is None:
            state = RepoState(
                repo_path=repo_path,
                repo_name=Path(repo_path).name,
                first_analyzed=now,
                last_analyzed=now,
                analysis_count=1,
                last_commit_analyzed=snapshot.commit_hash,
                current_health=snapshot,
                history=[snapshot],
            )
        else:
            # Delta calculation between old and new snapshot
            snapshot.new_issues_since_previous = max(
                0,
                (snapshot.critical_issues + snapshot.high_issues)
                - (existing.current_health.critical_issues + existing.current_health.high_issues),
            )
            snapshot.fixed_issues_since_previous = max(
                0,
                (existing.current_health.critical_issues + existing.current_health.high_issues)
                - (snapshot.critical_issues + snapshot.high_issues),
            )
            history = existing.history + [snapshot]
            state = RepoState(
                repo_path=repo_path,
                repo_name=existing.repo_name,
                first_analyzed=existing.first_analyzed,
                last_analyzed=now,
                analysis_count=existing.analysis_count + 1,
                last_commit_analyzed=snapshot.commit_hash or existing.last_commit_analyzed,
                current_health=snapshot,
                history=history[-20:],  # keep last 20 snapshots
            )

        self._write(key, state)
        self._cache[key] = state
        return state

    def get_state(self, repo_path: str) -> Optional[RepoState]:
        """Return current state or None if this repo has never been analyzed."""
        key = self._key(repo_path)
        if key in self._cache:
            return self._cache[key]
        return self._read(key)

    def get_all_states(self) -> List[RepoState]:
        """Return states for all known repos, sorted by last_analyzed desc."""
        states: List[RepoState] = []
        for p in self._dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                states.append(RepoState.model_validate(data))
            except Exception:
                continue
        return sorted(states, key=lambda s: s.last_analyzed, reverse=True)

    def get_history(self, repo_path: str) -> List[HealthSnapshot]:
        state = self.get_state(repo_path)
        return state.history if state else []

    async def get_changed_files(self, repo_path: str) -> List[str]:
        """Return files changed since the last analyzed commit (or since HEAD~1)."""
        state = self.get_state(repo_path)
        if state and state.last_commit_analyzed:
            files = await self._git_diff(repo_path, state.last_commit_analyzed, "HEAD")
            if files is not None:
                return files
        # Fallback: uncommitted changes
        return await self._git_diff_head(repo_path)

    async def get_new_commit_count(self, repo_path: str) -> int:
        """How many commits landed since last analysis."""
        state = self.get_state(repo_path)
        if not state or not state.last_commit_analyzed:
            return 0
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-list", "--count",
                f"{state.last_commit_analyzed}..HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return int(stdout.decode().strip() or 0)
        except Exception:
            return 0

    async def get_current_commit(self, repo_path: str) -> Optional[str]:
        """Return current HEAD commit hash."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result = stdout.decode().strip()
            return result if result else None
        except Exception:
            return None

    def get_trend(self, repo_path: str) -> str:
        state = self.get_state(repo_path)
        return state.trend if state else "stable"

    def weekly_summary(self) -> Dict[str, Any]:
        """Aggregate stats for the proactive monitoring banner."""
        states = self.get_all_states()
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        total_commits = 0
        total_new_issues = 0
        total_fixed = 0

        for s in states:
            recent = [h for h in s.history if h.date >= one_week_ago]
            for h in recent:
                total_new_issues += h.new_issues_since_previous
                total_fixed += h.fixed_issues_since_previous

        return {
            "repos_monitored": len(states),
            "new_issues": total_new_issues,
            "issues_fixed": total_fixed,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _key(self, repo_path: str) -> str:
        """Stable filename key for a repo path."""
        normalized = str(Path(repo_path).resolve())
        # Short hash to keep filenames sane on any OS
        h = hashlib.sha1(normalized.encode()).hexdigest()[:12]
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(normalized).name)[:40]
        return f"{name}_{h}"

    def _write(self, key: str, state: RepoState) -> None:
        path = self._dir / f"{key}.json"
        path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _read(self, key: str) -> Optional[RepoState]:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = RepoState.model_validate(data)
            self._cache[key] = state
            return state
        except Exception:
            return None

    async def _git_diff(
        self, repo_path: str, base: str, head: str
    ) -> Optional[List[str]]:
        """Files changed between two commits. Returns None on error."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--name-only", base, head,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            lines = stdout.decode().strip().split("\n")
            return [l for l in lines if l]
        except Exception:
            return None

    async def _git_diff_head(self, repo_path: str) -> List[str]:
        """Uncommitted changes (git diff HEAD)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--name-only", "HEAD",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode().strip().split("\n")
            return [l for l in lines if l]
        except Exception:
            return []


# ── Process-wide singleton ─────────────────────────────────────────────────────

_store: Optional[RepoStateStore] = None


def get_repo_state_store() -> RepoStateStore:
    global _store
    if _store is None:
        _store = RepoStateStore()
    return _store
