"""Synthesizer — AI combines council proposals into one unified, executable plan.

The Synthesizer gives all proposals, critiques, and votes to an AI model as
narrative input and asks it to synthesize. There is no vote counting, no tallying,
no algorithmic winner selection. The AI reads everything and produces the plan.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.ai.decision_record import (
    Critique,
    ElementVote,
    Problem,
    Proposal,
    SynthesizedPlan,
)
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]

_SYSTEM_PROMPT = """You are the synthesis expert for an AI engineering org.
Six AI advisors proposed solutions to a problem. Combine them into ONE unified plan.

Rules:
- Include ideas where 2+ experts agree; credit all contributors.
- Resolve conflicts: security > business impact > evidence > feasibility.
- Reject ideas that cannot be implemented; explain why.
- Every step must name which expert's thinking it came from.

Output ONLY valid JSON (no prose):
{"plan":"<actionable steps>","included_from":{"expert":["ideas"]},"rejected_elements":[{"element":"","reason":""}],"conflicts_resolved":[{"conflict":"","resolution":""}],"confidence":<0-1>,"needs_revision":<bool>}"""


class Synthesizer:
    """Combines proposals, critiques, and votes into one unified plan via AI synthesis."""

    def __init__(self, llm: LLMCallable) -> None:
        self._llm = llm

    async def synthesize(
        self,
        problem: Problem,
        proposals: List[Proposal],
        critiques: List[Critique],
        votes: List[ElementVote],
    ) -> SynthesizedPlan:
        """Produce a unified plan by asking AI to synthesize all council input."""
        if not proposals:
            return SynthesizedPlan(
                plan="No proposals received from council.",
                confidence=0.0,
                needs_revision=True,
            )

        from app.ai.model_routing import call_with_json_retry

        proposals_text = self._format_proposals(proposals)
        critiques_text = self._format_critiques(critiques)
        # Top 6 votes by weight to cap prompt size
        top_votes = sorted(votes, key=lambda v: v.weight, reverse=True)[:6]
        votes_text = self._format_votes(top_votes)

        user_msg = (
            f"Problem: {problem.description[:300]}\n\n"
            f"=== PROPOSALS ===\n{proposals_text}\n\n"
            f"=== CRITIQUES ===\n{critiques_text}\n\n"
            f"=== VOTES ===\n{votes_text}\n\n"
            "Output ONE unified plan as JSON."
        )

        result = await call_with_json_retry(
            llm=self._llm,
            system=_SYSTEM_PROMPT,
            user=user_msg,
            parse_fn=lambda raw: self._try_parse_plan(raw, proposals),
            max_retries=2,
        )

        if result is None:
            result = self._fallback_plan(proposals)

        logger.info(
            "Plan synthesized",
            confidence=result.confidence,
            needs_revision=result.needs_revision,
            proposals_count=len(proposals),
        )
        return result

    async def revise(
        self,
        problem: Problem,
        previous_plan: SynthesizedPlan,
        additional_critique: str,
    ) -> SynthesizedPlan:
        """Revise a plan that did not meet confidence threshold."""
        from app.ai.model_routing import call_with_json_retry

        user_msg = (
            f"Problem: {problem.description[:300]}\n\n"
            f"Previous plan (confidence {previous_plan.confidence:.0%}, needs revision):\n"
            f"{previous_plan.plan[:500]}\n\n"
            f"Critique:\n{additional_critique[:400]}\n\n"
            "Revise and output JSON with the same schema."
        )

        result = await call_with_json_retry(
            llm=self._llm,
            system=_SYSTEM_PROMPT,
            user=user_msg,
            parse_fn=lambda raw: self._try_parse_plan(raw, []),
            max_retries=2,
        )

        return result or self._fallback_plan([])

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _format_proposals(self, proposals: List[Proposal]) -> str:
        parts = []
        for p in proposals:
            concerns = "; ".join(p.concerns[:2]) if p.concerns else "none"
            parts.append(
                f"[{p.member_name.upper()}] conf={p.confidence:.0%}\n"
                f"Approach: {p.approach[:200]}\n"
                f"Reasoning: {p.reasoning[:150]}\n"
                f"Concerns: {concerns}"
            )
        return "\n\n".join(parts)

    def _format_critiques(self, critiques: List[Critique]) -> str:
        if not critiques:
            return "None."
        parts = []
        for c in critiques[:8]:
            weaknesses = "; ".join(c.weaknesses[:2]) if c.weaknesses else "none"
            missing = "; ".join(c.missing_considerations[:2]) if c.missing_considerations else "none"
            parts.append(
                f"{c.critic_name}→{c.target_member}: weak={weaknesses} | missing={missing}"
            )
        return "\n".join(parts)

    def _format_votes(self, votes: List[ElementVote]) -> str:
        if not votes:
            return "None."
        lines = []
        for v in votes:
            verdict = "YES" if v.include else "NO"
            lines.append(f"{v.voter} {verdict} '{v.element[:60]}': {v.reasoning[:80]}")
        return "\n".join(lines)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _try_parse_plan(self, raw: str, proposals: List[Proposal]) -> Optional[SynthesizedPlan]:
        """Return SynthesizedPlan or None on parse failure."""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            return SynthesizedPlan(
                plan=str(data.get("plan", "")),
                included_from=dict(data.get("included_from", {})),
                rejected_elements=list(data.get("rejected_elements", [])),
                conflicts_resolved=list(data.get("conflicts_resolved", [])),
                confidence=float(data.get("confidence", 0.7)),
                needs_revision=bool(data.get("needs_revision", False)),
            )
        except Exception as e:
            logger.debug("Synthesizer parse failed", error=str(e), raw=raw[:200])
            return None

    def _fallback_plan(self, proposals: List[Proposal]) -> SynthesizedPlan:
        fallback = "\n\n".join(f"[{p.member_name}]: {p.approach}" for p in proposals)
        included = {p.member_name: [p.approach[:200]] for p in proposals}
        return SynthesizedPlan(
            plan=fallback or "Synthesis unavailable.",
            included_from=included,
            confidence=0.5,
            needs_revision=True,
        )
