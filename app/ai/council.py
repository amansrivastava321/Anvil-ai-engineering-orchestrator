"""Council — manages the debate process when the CEO convenes advisors."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.ai.council_members import CouncilMember, create_all_members
from app.ai.decision_record import (
    Critique,
    ElementVote,
    Problem,
    Proposal,
    SynthesizedPlan,
)
from app.ai.synthesizer import Synthesizer
from app.ai.tool_registry import ToolRegistry, get_tool_registry
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]

_CONFIDENCE_THRESHOLD = 0.60  # Plans below this trigger revision
_MAX_REVISION_ROUNDS = 2


class AICouncil:
    """Orchestrates multi-member debate: propose → critique → vote → synthesize.

    The CEO convenes the council for Mode 3 decisions. The council:
    1. Sends the problem to selected members simultaneously
    2. Each member proposes independently
    3. Each member critiques the other proposals
    4. Each member votes on specific elements
    5. The Synthesizer combines everything into one plan
    6. If confidence is low, the council does a revision round
    """

    def __init__(
        self,
        llm: LLMCallable,
        tool_registry: Optional[ToolRegistry] = None,
        decision_history: Optional[List[Dict[str, Any]]] = None,
        synthesis_llm: Optional[LLMCallable] = None,
    ) -> None:
        self._llm = llm
        self._all_members = create_all_members(
            llm=llm,
            tool_registry=tool_registry or get_tool_registry(),
            decision_history=decision_history or [],
        )
        self._synthesizer = Synthesizer(llm=synthesis_llm or llm)

    # ── Public API ────────────────────────────────────────────────────────────

    async def convene(
        self,
        problem: Problem,
        member_names: Optional[List[str]] = None,
        context: str = "",
    ) -> SynthesizedPlan:
        """Run a full council debate and return the synthesized plan.

        Args:
            problem: The problem to deliberate on.
            member_names: Names of members to include. None → all six.
            context: Additional context string for member prompts.
        """
        members = self._select_members(member_names)
        logger.info(
            "Council convened",
            problem=problem.description[:80],
            members=[m.name for m in members],
        )

        # Round 1: parallel proposals
        proposals = await self._gather_proposals(members, problem, context)

        # Round 2: parallel critiques
        critiques = await self._gather_critiques(members, proposals)

        # Round 3: parallel votes
        votes = await self._gather_votes(members, proposals)

        # Synthesize
        plan = await self._synthesizer.synthesize(problem, proposals, critiques, votes)
        logger.info("Initial synthesis complete", confidence=plan.confidence)

        # Revision loop
        for round_num in range(_MAX_REVISION_ROUNDS):
            if not plan.needs_revision and plan.confidence >= _CONFIDENCE_THRESHOLD:
                break
            logger.info("Plan needs revision", round=round_num + 1, confidence=plan.confidence)
            critique_summary = self._summarize_critiques(critiques)
            plan = await self._synthesizer.revise(problem, plan, critique_summary)

        logger.info(
            "Council complete",
            final_confidence=plan.confidence,
            needs_revision=plan.needs_revision,
        )
        return plan

    async def consult(
        self,
        problem: Problem,
        expert_names: List[str],
        context: str = "",
    ) -> SynthesizedPlan:
        """Consult a specific subset of experts without full debate.

        Used for Mode 2 decisions where the CEO knows which experts to ask.
        No critique / vote round — just proposals directly synthesized.
        """
        members = [
            self._all_members[name]
            for name in expert_names
            if name in self._all_members
        ]
        if not members:
            members = [self._all_members["architect"]]

        logger.info("Expert consultation", experts=[m.name for m in members])
        proposals = await self._gather_proposals(members, problem, context)
        # Lightweight synthesis — no votes or critiques for speed
        plan = await self._synthesizer.synthesize(problem, proposals, [], [])
        return plan

    def member_names(self) -> List[str]:
        return list(self._all_members.keys())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_members(self, names: Optional[List[str]]) -> List[CouncilMember]:
        if not names:
            return list(self._all_members.values())
        return [
            self._all_members[n] for n in names if n in self._all_members
        ] or list(self._all_members.values())

    async def _gather_proposals(
        self,
        members: List[CouncilMember],
        problem: Problem,
        context: str,
    ) -> List[Proposal]:
        tasks = [m.analyze(problem, context) for m in members]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        proposals = []
        for m, result in zip(members, results):
            if isinstance(result, Exception):
                logger.warning("Member proposal failed", member=m.name, error=str(result))
                proposals.append(
                    Proposal(
                        member_name=m.name,
                        approach="[Analysis failed — see logs]",
                        reasoning=str(result),
                        confidence=0.0,
                    )
                )
            else:
                proposals.append(result)
        return proposals

    async def _gather_critiques(
        self,
        members: List[CouncilMember],
        proposals: List[Proposal],
    ) -> List[Critique]:
        tasks = [m.critique(proposals) for m in members]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_critiques: List[Critique] = []
        for m, result in zip(members, results):
            if isinstance(result, list):
                all_critiques.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Member critique failed", member=m.name, error=str(result))
        return all_critiques

    async def _gather_votes(
        self,
        members: List[CouncilMember],
        proposals: List[Proposal],
    ) -> List[ElementVote]:
        tasks = [m.vote(proposals) for m in members]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_votes: List[ElementVote] = []
        for m, result in zip(members, results):
            if isinstance(result, list):
                all_votes.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Member vote failed", member=m.name, error=str(result))
        return all_votes

    def _summarize_critiques(self, critiques: List[Critique]) -> str:
        if not critiques:
            return "No specific critiques — revise for clarity and completeness."
        parts = []
        for c in critiques[:8]:
            if c.weaknesses:
                parts.append(f"{c.critic_name} on {c.target_member}: {'; '.join(c.weaknesses[:2])}")
            if c.missing_considerations:
                parts.append(f"  Missing: {'; '.join(c.missing_considerations[:2])}")
        return "\n".join(parts)
