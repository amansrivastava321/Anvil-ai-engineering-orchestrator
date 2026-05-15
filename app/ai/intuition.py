"""Intuition — CEO's discovered patterns and learned rules.

Patterns are NOT programmed. They emerge from the CEO analyzing its own
decision history. Each pattern is discovered by the LLM reflecting on
a batch of similar decisions and extracting what worked and why.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_INTUITION_FILE = Path("data/ai_decisions/intuition.json")

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]


# ── Models ───────────────────────────────────────────────────────────────────


class Pattern(BaseModel):
    """A pattern the CEO discovered from its own decision history."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    keywords: List[str]
    approach_summary: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    times_matched: int = 0
    times_worked: int = 0
    times_failed: int = 0
    risk_level: str = "medium"
    preferred_mode: str = "consult_experts"
    requires_experts: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None, exclude=False)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        total = self.times_worked + self.times_failed
        return self.times_worked / total if total > 0 else 0.0

    def matches(self, keywords: List[str]) -> float:
        """Return match score [0, 1] against a list of problem keywords."""
        if not self.keywords or not keywords:
            return 0.0
        kw_set = set(kw.lower() for kw in keywords)
        pattern_set = set(kw.lower() for kw in self.keywords)
        overlap = len(kw_set & pattern_set)
        return overlap / len(pattern_set)


class Intuition:
    """Stores and manages all discovered patterns. Learns from outcomes."""

    def __init__(
        self,
        path: Path = _INTUITION_FILE,
        llm: Optional[LLMCallable] = None,
    ) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._patterns: Dict[str, Pattern] = {}
        self._load()

    # ── Pattern matching ────────────────────────────────────────────────────

    def find_matches(self, keywords: List[str], threshold: float = 0.3) -> List[Tuple[Pattern, float]]:
        """Return list of (pattern, score) sorted by score descending."""
        results = []
        for pattern in self._patterns.values():
            score = pattern.matches(keywords)
            if score >= threshold:
                results.append((pattern, score * pattern.confidence))
        return sorted(results, key=lambda x: x[1], reverse=True)

    async def find_matches_semantic(
        self, problem_text: str, threshold: float = 0.70
    ) -> List[Tuple[Pattern, float]]:
        """Return patterns matching problem_text via bge-m3 cosine similarity.

        Falls back to empty list if embeddings unavailable — caller should
        then fall back to keyword find_matches().
        """
        try:
            from app.integrations.ollama.embeddings import cosine_similarity, get_embedding

            query_emb = await get_embedding(problem_text)
            results = []
            for pattern in self._patterns.values():
                if pattern.embedding is None:
                    continue
                sim = cosine_similarity(query_emb, pattern.embedding)
                if sim >= threshold:
                    results.append((pattern, sim * pattern.confidence))
            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.debug("Semantic matching unavailable — use keyword fallback", error=str(e))
            return []

    def best_match(self, keywords: List[str], threshold: float = 0.3) -> Optional[Tuple[Pattern, float]]:
        matches = self.find_matches(keywords, threshold)
        return matches[0] if matches else None

    # ── Learning ─────────────────────────────────────────────────────────────

    def record_match(self, pattern_id: str, success: bool) -> None:
        """Update a pattern's statistics after an outcome is known."""
        p = self._patterns.get(pattern_id)
        if not p:
            return
        p.times_matched += 1
        if success:
            p.times_worked += 1
        else:
            p.times_failed += 1
        # Bayesian-style confidence update
        sr = p.success_rate
        p.confidence = min(1.0, sr * 0.7 + p.confidence * 0.3)
        p.last_updated = datetime.now(timezone.utc)
        self._flush()

    async def discover_patterns_from_decisions(
        self, decisions: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Ask the LLM to analyze decision records and discover new patterns."""
        if not self._llm or not decisions:
            return []

        import re

        from app.ai.model_routing import call_with_json_retry

        sample = decisions[:20]
        history_text = "\n---\n".join(
            f"Problem: {d.get('problem_description', '')[:200]}\n"
            f"Mode: {d.get('mode', '')} | Success: {d.get('outcome_success', '?')}\n"
            f"Keywords: {', '.join(d.get('problem_keywords', []))}"
            for d in sample
        )

        system = (
            "Analyze decision history and find recurring patterns. "
            "Output ONLY a JSON array. Each object: "
            "description, keywords (list), approach_summary, confidence (0-1), "
            "risk_level (low/medium/high/critical), "
            "preferred_mode (decide_alone/consult_experts/convene_council), "
            "requires_experts (list). No prose."
        )
        prompt = f"Decisions:\n\n{history_text}"

        def _parse(raw: str):
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group())
            except Exception:
                return None

        data = await call_with_json_retry(
            llm=self._llm,
            system=system,
            user=prompt,
            parse_fn=_parse,
            max_retries=2,
        )

        if not data:
            logger.warning("Pattern discovery produced no parseable output")
            return []

        new_patterns = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                p = Pattern(**{k: v for k, v in item.items() if k in Pattern.model_fields})
            except Exception:
                continue
            if p.id not in self._patterns:
                # Pre-compute bge-m3 embedding for semantic matching
                try:
                    from app.integrations.ollama.embeddings import get_embedding
                    p.embedding = await get_embedding(p.description)
                except Exception:
                    pass
                self._patterns[p.id] = p
                new_patterns.append(p)

        if new_patterns:
            self._flush()
        logger.info("Discovered patterns from history", count=len(new_patterns))
        return new_patterns

    def add_pattern(self, pattern: Pattern) -> None:
        self._patterns[pattern.id] = pattern
        self._flush()

    def all_patterns(self) -> List[Pattern]:
        return sorted(self._patterns.values(), key=lambda p: p.confidence, reverse=True)

    def pattern_count(self) -> int:
        return len(self._patterns)

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for item in raw:
                p = Pattern.model_validate(item)
                self._patterns[p.id] = p
        except Exception:
            logger.warning("Failed to load intuition — starting fresh")

    def _flush(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    [p.model_dump(mode="json") for p in self._patterns.values()],
                    indent=2,
                )
            )
        except Exception as e:
            logger.error("Failed to persist intuition", error=str(e))
