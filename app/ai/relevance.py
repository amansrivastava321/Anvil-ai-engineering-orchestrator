"""Relevance engine — narrows 500+ files down to the 3-8 that matter for a task.

Four signals, combined multiplicatively:
  1. Semantic search  — bge-m3 embedding similarity
  2. Graphify expand  — pull in direct dependencies of found files
  3. Git recency boost — recently changed files score 1.5×
  4. Memory boost     — files from similar past tasks score 1.3×
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RelevantFile:
    path: str
    score: float
    reason: str
    suspected_function: Optional[str] = None


class RelevanceEngine:
    """Ranks source files by relevance to a task description."""

    async def find_relevant_files(
        self,
        task_description: str,
        repo_path: str,
        graph_query: Any,  # GraphQuery — avoid circular import
        max_files: int = 8,
    ) -> List[RelevantFile]:
        """Return up to max_files source files ranked by relevance to the task.

        Steps:
        1. Semantic search via bge-m3 / keyword fallback
        2. Expand by one dependency hop
        3. Apply git-recency and memory boosts
        4. Rank and clip to max_files
        """
        # Step 1: semantic / keyword seed
        scores: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}

        semantic_files = await self._semantic_search(task_description, repo_path, graph_query)
        for f, sim in semantic_files.items():
            scores[f] = sim
            reasons.setdefault(f, []).append(f"semantic match ({sim:.0%})")

        # Step 2: one-hop dependency expansion (lower base score)
        expanded = await self._graphify_expand(list(scores.keys()), graph_query)
        for f in expanded:
            if f not in scores:
                scores[f] = 0.30
                reasons.setdefault(f, []).append("dependency of matched file")

        if not scores:
            logger.debug("Relevance engine: no candidates found")
            return []

        # Step 3: boosts (run in parallel)
        git_boost, mem_boost = await asyncio.gather(
            self._git_boost(list(scores.keys()), repo_path, graph_query),
            self._memory_boost(list(scores.keys()), task_description),
        )

        # Apply multipliers
        for f in scores:
            scores[f] *= git_boost.get(f, 1.0) * mem_boost.get(f, 1.0)
            if git_boost.get(f, 1.0) > 1.0:
                reasons[f].append("recently changed")
            if mem_boost.get(f, 1.0) > 1.0:
                reasons[f].append("used in similar past task")

        # Step 4: rank and return
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:max_files]
        return [
            RelevantFile(
                path=f,
                score=round(s, 3),
                reason="; ".join(reasons.get(f, ["graph match"])),
            )
            for f, s in ranked
        ]

    # ── Signal helpers ────────────────────────────────────────────────────────

    async def _semantic_search(
        self,
        task: str,
        repo_path: str,
        graph_query: Any,
    ) -> Dict[str, float]:
        """Score files by semantic / keyword similarity to the task."""
        try:
            files = await graph_query.get_related_files(task, repo_path)
            # get_related_files returns a ranked list; assign decaying scores
            return {f: max(0.3, 0.9 - i * 0.08) for i, f in enumerate(files)}
        except Exception as e:
            logger.debug("Semantic search step failed", error=str(e))
            return {}

    async def _graphify_expand(
        self,
        seed_files: List[str],
        graph_query: Any,
    ) -> List[str]:
        """Add direct dependencies of seed files."""
        expanded = set()
        for f in seed_files:
            try:
                deps = graph_query.get_dependencies(f)
                expanded.update(deps)
            except Exception:
                pass
        # Remove files already in seed
        return [f for f in expanded if f not in seed_files]

    async def _git_boost(
        self,
        files: List[str],
        repo_path: str,
        graph_query: Any,
    ) -> Dict[str, float]:
        """Files changed in the last 7 days score 1.5×."""
        try:
            recent = set(graph_query.get_recently_changed(repo_path, days=7))
            return {f: 1.5 if f in recent else 1.0 for f in files}
        except Exception:
            return {f: 1.0 for f in files}

    async def _memory_boost(
        self, files: List[str], task: str
    ) -> Dict[str, float]:
        """Files from similar past CEO decisions score 1.3×."""
        try:
            from app.ai.decision_record import get_decision_store

            store = get_decision_store()
            task_lower = task.lower()
            task_words = {w for w in task_lower.split() if len(w) > 4}
            past_files: set[str] = set()

            for decision in store.all(50):
                desc_words = {w for w in decision.problem_description.lower().split() if len(w) > 4}
                overlap = len(task_words & desc_words)
                if overlap >= 2:
                    # This past decision is similar — note its problem keywords as file hints
                    for kw in decision.problem_keywords:
                        # Very rough: if keyword appears in a file path, boost that file
                        for f in files:
                            if kw.lower() in Path(f).name.lower():
                                past_files.add(f)

            return {f: 1.3 if f in past_files else 1.0 for f in files}
        except Exception:
            return {f: 1.0 for f in files}
