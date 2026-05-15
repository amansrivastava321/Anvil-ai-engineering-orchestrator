"""Council members — six specialized AI advisors to the CEO.

Each member has a unique perspective, a focused system prompt, and genuine
specialization. They analyze problems independently, then critique each other,
then vote on specific plan elements.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.ai.decision_record import Critique, ElementVote, Problem, Proposal, RiskLevel
from app.ai.tool_registry import ToolRegistry, get_tool_registry
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]


# ── Base member ───────────────────────────────────────────────────────────────


class CouncilMember(ABC):
    """Abstract base for all council members."""

    name: str = "council_member"
    title: str = "Advisor"

    def __init__(
        self,
        llm: LLMCallable,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._llm = llm
        self._tools = tool_registry or get_tool_registry()

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    async def analyze(self, problem: Problem, context: str = "") -> Proposal:
        """Analyze the problem and return a proposal."""
        from app.ai.model_routing import call_with_json_retry

        tools_summary = self._tools.describe_all()
        user_msg = (
            f"Problem: {problem.description[:300]}\n"
            f"Risk: {problem.risk_level.value} | "
            f"Critical: {problem.is_business_critical} | "
            f"Systems: {', '.join(problem.systems_affected[:3]) or 'unknown'}\n"
            f"Context: {context[:200] or 'none'}\n\n"
            f"Tools: {tools_summary[:300]}\n\n"
            "Output JSON: approach (str), reasoning (str), concerns (list[str]), "
            "recommended_tools (list[str]), estimated_risk (low/medium/high/critical), "
            "confidence (0.0-1.0). Output ONLY valid JSON."
        )

        result = await call_with_json_retry(
            llm=self._llm,
            system=self.system_prompt,
            user=user_msg,
            parse_fn=self._try_parse_proposal,
            max_retries=2,
        )
        return result or Proposal(
            member_name=self.name,
            approach="Analysis unavailable.",
            reasoning="LLM did not return parseable JSON.",
            confidence=0.0,
        )

    async def critique(self, proposals: List[Proposal]) -> List[Critique]:
        """Critique all other members' proposals."""
        if not proposals:
            return []

        from app.ai.model_routing import call_with_json_retry

        others = [p for p in proposals if p.member_name != self.name]
        if not others:
            return []

        proposals_text = "\n---\n".join(
            f"From {p.member_name}: {p.approach[:200]}\nReasoning: {p.reasoning[:100]}"
            for p in others
        )
        user_msg = (
            f"Review these proposals:\n\n{proposals_text}\n\n"
            "Output JSON array of critique objects: "
            "target_member (str), strengths (list[str]), weaknesses (list[str]), "
            "missing_considerations (list[str]). ONLY valid JSON array."
        )

        result = await call_with_json_retry(
            llm=self._llm,
            system=self.system_prompt,
            user=user_msg,
            parse_fn=self._try_parse_critiques,
            max_retries=2,
        )
        return result or []

    async def vote(self, proposals: List[Proposal]) -> List[ElementVote]:
        """Vote on specific elements from all proposals."""
        if not proposals:
            return []

        from app.ai.model_routing import call_with_json_retry

        elements = [f"[{p.member_name}] {p.approach[:200]}" for p in proposals]
        elements_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(elements))
        user_msg = (
            f"Vote on these plan elements:\n\n{elements_text}\n\n"
            "Output JSON array: element (str), voter (your name), "
            "include (bool), reasoning (str). ONLY valid JSON array."
        )

        result = await call_with_json_retry(
            llm=self._llm,
            system=self.system_prompt,
            user=user_msg,
            parse_fn=self._try_parse_votes,
            max_retries=2,
        )
        votes = result or []
        for v in votes:
            v.voter = self.name
        return votes

    # ── Try-parse helpers (return None on failure) ────────────────────────────

    def _try_parse_proposal(self, raw: str) -> Optional[Proposal]:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            risk_raw = data.get("estimated_risk", "medium")
            try:
                risk = RiskLevel(risk_raw)
            except ValueError:
                risk = RiskLevel.MEDIUM
            return Proposal(
                member_name=self.name,
                approach=str(data.get("approach", ""))[:500],
                reasoning=str(data.get("reasoning", ""))[:300],
                concerns=list(data.get("concerns", []))[:5],
                recommended_tools=list(data.get("recommended_tools", [])),
                estimated_risk=risk,
                confidence=float(data.get("confidence", 0.7)),
            )
        except Exception as e:
            logger.debug("Proposal parse failed", member=self.name, error=str(e))
            return None

    def _try_parse_critiques(self, raw: str) -> Optional[List[Critique]]:
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return None
            items = json.loads(match.group())
            return [
                Critique(
                    critic_name=self.name,
                    target_member=str(item.get("target_member", "")),
                    strengths=list(item.get("strengths", []))[:3],
                    weaknesses=list(item.get("weaknesses", []))[:3],
                    missing_considerations=list(item.get("missing_considerations", []))[:3],
                )
                for item in items
                if isinstance(item, dict)
            ]
        except Exception as e:
            logger.debug("Critique parse failed", member=self.name, error=str(e))
            return None

    def _try_parse_votes(self, raw: str) -> Optional[List[ElementVote]]:
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return None
            items = json.loads(match.group())
            return [
                ElementVote(
                    element=str(item.get("element", ""))[:200],
                    voter=self.name,
                    include=bool(item.get("include", True)),
                    reasoning=str(item.get("reasoning", ""))[:150],
                    weight=1.0,
                )
                for item in items
                if isinstance(item, dict)
            ]
        except Exception as e:
            logger.debug("Vote parse failed", member=self.name, error=str(e))
            return None

    # ── Legacy parse helpers kept for test compatibility ──────────────────────

    def _parse_proposal(self, raw: str) -> Proposal:
        result = self._try_parse_proposal(raw)
        return result or Proposal(member_name=self.name, approach=raw[:1000], reasoning="Raw LLM output")

    def _parse_critiques(self, raw: str) -> List[Critique]:
        return self._try_parse_critiques(raw) or []

    def _parse_votes(self, raw: str) -> List[ElementVote]:
        return self._try_parse_votes(raw) or []


# ── Six specialized members ───────────────────────────────────────────────────


class ArchitectAI(CouncilMember):
    name = "architect"
    title = "VP of Architecture"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the VP of Architecture in an AI engineering org. "
            "Focus on module dependencies, interface contracts, coupling, and architectural patterns. "
            "Ask: What modules are affected? What is the dependency chain? Will this create debt? "
            "Propose solutions that preserve clean boundaries and long-term maintainability."
        )


class SecurityAI(CouncilMember):
    name = "security"
    title = "VP of Security"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the VP of Security in an AI engineering org. "
            "Focus on vulnerabilities, injection vectors, auth gaps, and data exposure. "
            "Ask: What can an attacker exploit? Is sensitive data protected? Are secrets safe? "
            "Assume hostile input. Never approve code that trades security for convenience."
        )


class PerformanceAI(CouncilMember):
    name = "performance"
    title = "VP of Performance"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the VP of Performance in an AI engineering org. "
            "Focus on bottlenecks, N+1 queries, blocking I/O, and scaling constraints. "
            "Ask: Will this scale 10x? What is the memory impact? Can this be parallelized? "
            "Be skeptical of solutions that work at small scale but fail under load."
        )


class TestingAI(CouncilMember):
    name = "testing"
    title = "VP of Quality"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the VP of Quality in an AI engineering org. "
            "Focus on edge cases, coverage gaps, and testability issues. "
            "Ask: What edge cases are missing? Is this testable? What inputs will break this? "
            "Propose specific test cases — not vague 'add more tests' suggestions."
        )


class MemoryAI(CouncilMember):
    name = "memory"
    title = "Organizational Historian"

    def __init__(
        self,
        llm: LLMCallable,
        tool_registry: Optional[ToolRegistry] = None,
        decision_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(llm, tool_registry)
        self._history = decision_history or []

    @property
    def system_prompt(self) -> str:
        history_snippet = ""
        if self._history:
            recent = self._history[:5]
            history_snippet = "\n".join(
                f"- {d.get('problem_description', 'unknown')[:100]}: "
                f"{'✓' if d.get('outcome_success') else '✗'}"
                for d in recent
            )
        return (
            "You are the Organizational Historian in an AI engineering org. "
            "You remember past fixes, failures, and lessons learned. "
            "Ask: Have we seen this before? What worked? Are we repeating a past mistake? "
            f"Recent memory:\n{history_snippet or 'No history yet.'}"
        )


class DomainAI(CouncilMember):
    name = "domain"
    title = "Business Analyst"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Business Analyst in an AI engineering org. "
            "Focus on business impact, user flows, revenue implications, and SLAs. "
            "Ask: How many users are affected? What is the revenue impact? What is the downtime cost? "
            "Frame technical problems in terms of business outcomes and stakeholder priorities."
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_all_members(
    llm: LLMCallable,
    tool_registry: Optional[ToolRegistry] = None,
    decision_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, CouncilMember]:
    """Instantiate all six council members."""
    registry = tool_registry or get_tool_registry()
    history = decision_history or []
    return {
        "architect": ArchitectAI(llm=llm, tool_registry=registry),
        "security": SecurityAI(llm=llm, tool_registry=registry),
        "performance": PerformanceAI(llm=llm, tool_registry=registry),
        "testing": TestingAI(llm=llm, tool_registry=registry),
        "memory": MemoryAI(llm=llm, tool_registry=registry, decision_history=history),
        "domain": DomainAI(llm=llm, tool_registry=registry),
    }
