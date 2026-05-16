"""CEO AI — Chief Engineering Officer.

The most important component. The CEO:
- Receives every problem first
- Analyzes against learned intuition
- Selects the operating mode
- Executes: alone, consults experts, or convenes the full council
- Records and learns from every decision
- Initiates strategic work without being asked

The CEO ALWAYS makes the final decision. The council advises. The CEO decides.
The CEO MUST explain its reasoning at every step.
The CEO MUST get smarter over time.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.ai.council import AICouncil
from app.ai.decision_record import (
    CEODecision,
    DecisionMode,
    DecisionStore,
    Problem,
    RiskLevel,
    get_decision_store,
)
from app.ai.intuition import Intuition
from app.ai.mode_selector import ModeSelector
from app.ai.tool_registry import ToolRegistry, get_tool_registry
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]

_CEO_SYSTEM_PROMPT = """You are the CEO (Chief Engineering Officer) of an autonomous AI engineering organization.
You have final authority on all decisions. You lead a council of specialized expert AIs.

Your operating philosophy:
- Transparency: Always explain your reasoning. Which mode? Why? What patterns matched?
- Speed: Use your intuition for known patterns. Reserve the full council for genuinely novel problems.
- Learning: Every decision makes you smarter. You track patterns, confidence levels, and outcomes.
- Prevention: You don't just fix problems — you prevent them by anticipating failure before it happens.

Your specialization is strategic engineering judgment:
- You know when a problem is a symptom and when it is the disease
- You know which expert to ask and when the full council is needed
- You understand business impact, technical risk, and organizational priorities
- You have deep intuition built from analyzing thousands of similar problems

When solving a problem, structure your response as:
1. Pattern recognition: What have I seen like this before?
2. Mode selection: Which mode am I using and exactly why?
3. Assessment: What is the core issue?
4. Plan: What specific steps should be taken?
5. Verification: How will I know this worked?"""


class CEO:
    """The Chief Engineering Officer AI."""

    def __init__(
        self,
        llm: LLMCallable,
        store: Optional[DecisionStore] = None,
        intuition: Optional[Intuition] = None,
        tool_registry: Optional[ToolRegistry] = None,
        # Per-component model overrides (used in production; ignored in tests)
        selector_llm: Optional[LLMCallable] = None,
        synthesis_llm: Optional[LLMCallable] = None,
        pattern_llm: Optional[LLMCallable] = None,
    ) -> None:
        self._llm = llm
        self._store = store or get_decision_store()
        self._intuition = intuition or Intuition(llm=pattern_llm or llm)
        self._tools = tool_registry or get_tool_registry()
        self._mode_selector = ModeSelector(
            intuition=self._intuition, llm=selector_llm or llm
        )
        self._council = AICouncil(
            llm=llm,
            tool_registry=self._tools,
            decision_history=self._recent_history(20),
            synthesis_llm=synthesis_llm,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def receive_problem(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        risk_level: str = "medium",
        systems_affected: Optional[List[str]] = None,
        is_business_critical: bool = False,
    ) -> CEODecision:
        """Main entry point. Receive a problem, decide and act, return decision record."""
        start = time.monotonic()

        problem = Problem(
            description=description,
            context=context or {},
            risk_level=RiskLevel(risk_level) if risk_level in RiskLevel._value2member_map_ else RiskLevel.MEDIUM,
            systems_affected=systems_affected or [],
            is_business_critical=is_business_critical,
        )

        logger.info("CEO received problem", problem=description[:100], risk=risk_level)

        # Step 1: Pattern match
        keywords = problem.keywords()
        selection = await self._mode_selector.select(problem)

        logger.info(
            "CEO mode selected",
            mode=selection.mode.value,
            confidence=f"{selection.confidence:.0%}",
            reasoning=selection.reasoning[:100],
        )

        # Step 2: Execute the selected mode
        final_plan = await self._execute_mode(problem, selection, context or {})

        # Step 3: Record
        duration_ms = (time.monotonic() - start) * 1000
        decision = CEODecision(
            problem_id=problem.id,
            problem_description=description,
            mode=selection.mode,
            mode_reasoning=selection.reasoning,
            patterns_matched=selection.patterns_matched,
            confidence=selection.confidence,
            experts_consulted=selection.experts_suggested,
            final_plan=final_plan,
            decided_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            problem_keywords=keywords,
        )
        self._store.save(decision)

        logger.info(
            "CEO decision complete",
            mode=selection.mode.value,
            duration_ms=f"{duration_ms:.0f}",
        )
        return decision

    async def record_outcome(
        self,
        decision_id: str,
        success: bool,
        notes: str = "",
    ) -> Optional[CEODecision]:
        """Record the outcome of a decision and update learned patterns."""
        decision = self._store.get(decision_id)
        if not decision:
            logger.warning("Decision not found for outcome recording", id=decision_id)
            return None

        decision.outcome_success = success
        decision.outcome_notes = notes

        # Generate learning notes via LLM
        learning = await self._reflect_on_outcome(decision, success)
        decision.learning_notes = learning

        # Update intuition
        for pattern_id in decision.patterns_matched:
            self._intuition.record_match(pattern_id, success)

        self._store.update(decision)

        # Trigger pattern discovery periodically
        if self._store.count() % 10 == 0:
            asyncio.create_task(self._discover_patterns())

        logger.info("Outcome recorded", decision_id=decision_id, success=success)
        return decision

    async def strategic_review(self) -> List[CEODecision]:
        """Mode 4: CEO-initiated strategic work without being asked.

        Analyzes system-wide patterns, identifies proactive improvements,
        and generates strategic initiatives.
        """
        history = self._recent_history(50)
        logger.info("CEO strategic review initiated", decisions_analyzed=len(history))

        if not history:
            logger.info("No history for strategic review")
            return []

        history_summary = "\n".join(
            f"- {d.get('problem_description', '')[:80]}: "
            f"{'✓' if d.get('outcome_success') else '?'}"
            for d in history[:20]
        )

        system = (
            "You are the CEO doing a strategic engineering review. "
            "Analyze the decision history and identify: "
            "1) Recurring problems that should be fixed permanently, "
            "2) Systems showing warning signs before they break, "
            "3) Opportunities to improve engineering quality proactively. "
            "Output a JSON array of strategic initiatives, each with: "
            "title (str), description (str), rationale (str), urgency (low/medium/high). "
            "Output ONLY valid JSON array."
        )
        prompt = f"Recent decision history:\n{history_summary}"

        initiatives: List[CEODecision] = []
        try:
            import json
            import re
            raw = await self._llm(system, prompt)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                items = json.loads(match.group())
                for item in items[:5]:
                    desc = str(item.get("description", item.get("title", "")))
                    if desc:
                        d = await self.receive_problem(
                            description=desc,
                            context={"strategic": True, "rationale": item.get("rationale", "")},
                            risk_level="low",
                        )
                        initiatives.append(d)
        except Exception as e:
            logger.warning("Strategic review parsing failed", error=str(e))

        return initiatives

    async def handle(self, problem: "Problem") -> "CEODecision":
        """Smart entry point: auto-detects onboarding requests vs normal problems.

        If the description is empty or one of the onboarding trigger words, runs
        the onboarding pipeline instead of a standard council deliberation.
        """
        import os

        model_tier = "cloud" if os.environ.get("OPENROUTER_API_KEY") else "local"
        problem.context["model_tier"] = model_tier
        logger.info("CEO handling problem", tier=model_tier)

        _ONBOARD_TRIGGERS = {"", "analyze", "audit", "explore", "onboard", "scan"}
        desc = (problem.description or "").strip().lower()

        if desc in _ONBOARD_TRIGGERS:
            return await self._handle_onboarding(problem)

        repo_path = problem.context.get("repo_path", "")
        if repo_path:
            # Layer 1: repo state memory + graph summary (~300 tokens)
            self._inject_repo_context(repo_path, problem.context)
            # Layer 2: focused code context via relevance engine + file chunker
            await self._inject_code_context(problem, repo_path)

        return await self.receive_problem(
            description=problem.description,
            context=problem.context,
            risk_level=problem.risk_level.value,
            systems_affected=problem.systems_affected,
            is_business_critical=problem.is_business_critical,
        )

    async def _handle_onboarding(self, problem: "Problem") -> "CEODecision":
        """Delegate to the onboarding service and wrap the result as a CEODecision."""
        from app.services.onboarding_service import get_onboarding_service

        repo_path = problem.context.get("repo_path", "")
        if not repo_path:
            raise ValueError("repo_path is required for onboarding")

        svc = get_onboarding_service()
        report = await svc.onboard_repository(repo_path)

        description = (
            f"Onboarding complete for {report.project_name}. "
            f"Health score: {report.health_score}/100. "
            f"Critical issues: {len(report.critical)}, High: {len(report.high)}."
        )
        return await self.receive_problem(
            description=description,
            context={
                "repo_path": repo_path,
                "onboarding_report": report.model_dump(mode="json"),
                "graph_summary": report.executive_summary,
            },
            risk_level="medium",
        )

    def _inject_repo_context(self, repo_path: str, context: dict) -> None:
        """Populate context with repo state memory and graph summary (zero AI tokens)."""
        # 1. Repo state (what we already know)
        if "repo_state" not in context:
            try:
                from app.core.repo_state import get_repo_state_store
                store = get_repo_state_store()
                state = store.get_state(repo_path)
                if state:
                    context["repo_state"] = {
                        "health_score": state.current_health.health_score,
                        "health_trend": state.trend,
                        "analysis_count": state.analysis_count,
                        "critical_issues": state.current_health.critical_issues,
                        "last_analyzed": state.last_analyzed.isoformat(),
                    }
                    context["health_trend"] = state.trend
                    context["previous_health"] = state.current_health.health_score
            except Exception as e:
                logger.debug("Could not load repo state", error=str(e))

        # 2. Graph summary (structural map, ~300 tokens)
        if "graph_summary" not in context:
            try:
                summary = ""
                from pathlib import Path as _Path
                from app.integrations.graphify.query import get_graph_query
                graph_json = _Path(repo_path) / "graphify-out" / "graph.json"
                if graph_json.exists():
                    gq = get_graph_query(graph_json)
                    summary = gq.get_repo_summary()
                if summary:
                    context["graph_summary"] = summary
            except Exception as e:
                logger.debug("Graph summary skipped", error=str(e))

    async def _inject_code_context(self, problem: "Problem", repo_path: str) -> None:
        """Layered code retrieval: function-level search first, file-level fallback.

        Path A (code index exists): CodeContextBuilder → semantic function search →
            Graphify dependency expansion → AST compression → ~1,200 tokens.
        Path B (no index): RelevanceEngine → file-level search → skeleton compression.
        """
        # ── Path A: function-level retrieval via CodeContextBuilder ───────────
        try:
            from app.ai.code_context_builder import CodeContextBuilder

            builder = CodeContextBuilder()
            focused = await builder.build_context(
                problem=problem.description,
                repo_path=repo_path,
                max_tokens=1200,
            )
            if focused.code_snippets:
                problem.context["code_snippets"] = focused.code_snippets
                problem.context["dependency_graph"] = focused.dependency_graph
                if focused.graph_summary and "graph_summary" not in problem.context:
                    problem.context["graph_summary"] = focused.graph_summary
                problem.context["context_tokens"] = focused.total_tokens

                logger.info(
                    "Context built",
                    tokens=focused.total_tokens,
                    snippets=len(focused.code_snippets),
                    dep_graph_lines=focused.dependency_graph.count("\n"),
                )
                return
        except Exception as e:
            logger.debug("CodeContextBuilder failed", error=str(e))

        # ── Path B: file-level fallback via RelevanceEngine ───────────────────
        from pathlib import Path as _Path
        from app.integrations.graphify.query import get_graph_query
        from app.ai.relevance import RelevanceEngine
        from app.ai.context_budget import ContextBudget
        from app.tools.file_system.file_chunker import FileChunker

        graph_json = _Path(repo_path) / "graphify-out" / "graph.json"
        if not graph_json.exists():
            return

        try:
            gq = get_graph_query(graph_json)
            gq.load()
        except Exception as e:
            logger.debug("Graph load skipped", error=str(e))
            return

        total_files = gq.total_files
        engine = RelevanceEngine()
        try:
            relevant = await engine.find_relevant_files(
                task_description=problem.description,
                repo_path=repo_path,
                graph_query=gq,
                max_files=8,
            )
        except Exception as e:
            logger.debug("Relevance engine failed", error=str(e))
            return

        if not relevant:
            return

        budget = ContextBudget()
        chunker = FileChunker()
        snippets: list = []

        for rf in relevant:
            try:
                p = _Path(rf.path)
                if not p.exists():
                    p = _Path(repo_path) / rf.path
                if not p.exists():
                    continue

                content = p.read_text(encoding="utf-8", errors="replace")
                code = chunker.extract_skeleton(rf.path, content) if len(content) > 8_000 else content
                code = budget.truncate_to_fit("file_contents", code)
                if not budget.allocate_text("file_contents", code):
                    break

                snippets.append(f"# {rf.path}  [relevance {rf.score:.2f} — {rf.reason}]\n{code}")
            except Exception as e:
                logger.debug("Could not read relevant file", path=rf.path, error=str(e))

        if snippets:
            problem.context["code_snippets"] = snippets
            problem.context["files_analyzed"] = len(snippets)
            problem.context["files_skipped"] = max(0, total_files - len(snippets))

        logger.info(
            "CEO context assembled (file-level fallback)",
            analyzed=len(snippets),
            total_files=total_files,
            skipped=max(0, total_files - len(snippets)),
            budget=budget.get_budget_report(),
        )

    async def _get_graph_summary(self, repo_path: str) -> str:
        """Return a ~300-token graph summary for the given repo, or empty string."""
        from pathlib import Path
        from app.integrations.graphify.query import get_graph_query

        graph_json = Path(repo_path) / "graphify-out" / "graph.json"
        if not graph_json.exists():
            return ""
        try:
            gq = get_graph_query(graph_json)
            return gq.get_repo_summary()
        except Exception as e:
            logger.warning("Graph summary failed", repo=repo_path, error=str(e))
            return ""

    async def discover_patterns(self) -> int:
        """Ask the CEO to analyze its own history and discover new patterns."""
        history = self._recent_history(100)
        new = await self._intuition.discover_patterns_from_decisions(history)
        return len(new)

    def get_intuition_summary(self) -> Dict[str, Any]:
        patterns = self._intuition.all_patterns()
        return {
            "total_patterns": len(patterns),
            "high_confidence": [
                {"id": p.id, "description": p.description, "confidence": p.confidence}
                for p in patterns
                if p.confidence >= 0.8
            ],
            "total_decisions": self._store.count(),
            "success_rate": self._calc_success_rate(),
        }

    # ── Mode execution ────────────────────────────────────────────────────────

    async def _execute_mode(
        self,
        problem: Problem,
        selection: Any,  # ModeSelection
        context: Dict[str, Any],
    ) -> str:
        mode = selection.mode

        if mode == DecisionMode.DECIDE_ALONE:
            return await self._mode1_decide_alone(problem, selection, context)
        elif mode == DecisionMode.CONSULT_EXPERTS:
            return await self._mode2_consult_experts(problem, selection, context)
        elif mode == DecisionMode.CONVENE_COUNCIL:
            return await self._mode3_convene_council(problem, context)
        else:
            return await self._mode4_strategic(problem, context)

    async def _mode1_decide_alone(
        self,
        problem: Problem,
        selection: Any,
        context: Dict[str, Any],
    ) -> str:
        """Mode 1: CEO acts immediately on learned intuition."""
        tools_summary = self._tools.describe_all()
        user_msg = (
            f"Problem: {problem.description}\n\n"
            f"You are acting in Mode 1 (Decide Alone) because: {selection.reasoning}\n"
            f"Pattern confidence: {selection.confidence:.0%}\n"
            f"Risk: {problem.risk_level.value}\n\n"
            f"Available tools:\n{tools_summary}\n\n"
            "Provide your direct decision: specific steps, tools to use, and how to verify success. "
            "Be concrete and actionable. This is your direct judgment, no council needed."
        )
        return await self._llm(_CEO_SYSTEM_PROMPT, user_msg)

    async def _mode2_consult_experts(
        self,
        problem: Problem,
        selection: Any,
        context: Dict[str, Any],
    ) -> str:
        """Mode 2: Consult specific experts, then make final decision."""
        experts = selection.experts_suggested or ["architect"]
        context_str = str(context) if context else ""

        plan = await self._council.consult(problem, experts, context_str)

        # CEO makes final call
        user_msg = (
            f"Problem: {problem.description}\n\n"
            f"You consulted {', '.join(experts)} and received this plan:\n{plan.plan}\n\n"
            f"Plan confidence: {plan.confidence:.0%}\n\n"
            "As CEO, review this expert advice and state your FINAL DECISION. "
            "You may accept, modify, or override the expert plan. Explain any changes."
        )
        ceo_final = await self._llm(_CEO_SYSTEM_PROMPT, user_msg)
        return f"[CEO Mode 2 — consulted {', '.join(experts)}]\n\n{ceo_final}"

    async def _mode3_convene_council(
        self,
        problem: Problem,
        context: Dict[str, Any],
    ) -> str:
        """Mode 3: Full council debate, then CEO makes final decision."""
        context_str = str(context) if context else ""
        plan = await self._council.convene(problem, context=context_str)

        # CEO has final say
        user_msg = (
            f"Problem: {problem.description}\n\n"
            f"The full council deliberated and produced this plan:\n{plan.plan}\n\n"
            f"Council confidence: {plan.confidence:.0%}\n"
            f"Conflicts resolved: {len(plan.conflicts_resolved)}\n"
            f"Rejected elements: {len(plan.rejected_elements)}\n\n"
            "As CEO, state your FINAL DECISION. You have final authority. "
            "Accept the council plan, modify it, or override it — but always explain why."
        )
        ceo_final = await self._llm(_CEO_SYSTEM_PROMPT, user_msg)
        return (
            f"[CEO Mode 3 — Full Council]\n"
            f"Council plan (confidence {plan.confidence:.0%}):\n{plan.plan}\n\n"
            f"CEO Final Decision:\n{ceo_final}"
        )

    async def _mode4_strategic(
        self,
        problem: Problem,
        context: Dict[str, Any],
    ) -> str:
        """Mode 4: Strategic initiative — CEO-initiated, no user request."""
        user_msg = (
            f"Strategic Initiative: {problem.description}\n"
            f"Rationale: {context.get('rationale', 'Proactive improvement')}\n\n"
            "This is a CEO-initiated strategic action. Describe the specific steps, "
            "expected impact, and success criteria. This will happen autonomously."
        )
        return f"[CEO Mode 4 — Strategic Initiative]\n{await self._llm(_CEO_SYSTEM_PROMPT, user_msg)}"

    # ── Learning ──────────────────────────────────────────────────────────────

    async def _reflect_on_outcome(self, decision: CEODecision, success: bool) -> str:
        outcome_word = "succeeded" if success else "failed"
        user_msg = (
            f"I just made a decision that {outcome_word}.\n\n"
            f"Problem: {decision.problem_description}\n"
            f"Mode I used: {decision.mode.value} — {decision.mode_reasoning}\n"
            f"Confidence: {decision.confidence:.0%}\n"
            f"Experts consulted: {', '.join(decision.experts_consulted) or 'none'}\n"
            f"Plan: {decision.final_plan[:500]}\n\n"
            "As CEO, reflect: Was my mode selection correct? Was my confidence calibrated? "
            "What would I do differently? What did I learn? "
            "In 2-3 sentences, state the specific learning for next time."
        )
        try:
            return await self._llm(_CEO_SYSTEM_PROMPT, user_msg)
        except Exception:
            return f"Decision {outcome_word}. Mode {decision.mode.value} used."

    async def _discover_patterns(self) -> None:
        try:
            count = await self.discover_patterns()
            if count > 0:
                logger.info("CEO auto-discovered patterns", count=count)
        except Exception as e:
            logger.warning("Auto pattern discovery failed", error=str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _recent_history(self, n: int) -> List[Dict[str, Any]]:
        return [d.model_dump(mode="json") for d in self._store.all(n)]

    def _calc_success_rate(self) -> Optional[float]:
        decisions = self._store.all(500)
        rated = [d for d in decisions if d.outcome_success is not None]
        if not rated:
            return None
        return sum(1 for d in rated if d.outcome_success) / len(rated)


# ── Default LLM callable (wraps OllamaClient) ────────────────────────────────


async def _ollama_llm(system: str, prompt: str) -> str:
    """Default production LLM callable (dolphin-mistral — general reasoning)."""
    from app.ai.model_routing import make_ollama_llm, MODEL_ROUTING
    llm = make_ollama_llm(MODEL_ROUTING["default"])
    return await llm(system, prompt)


# ── Singleton ─────────────────────────────────────────────────────────────────

_ceo: Optional[CEO] = None


def get_ceo(llm: Optional[LLMCallable] = None) -> CEO:
    """Get or create the singleton CEO instance.

    In production (llm=None), each component gets a cloud-first routing LLM via
    make_routing_llm — free OpenRouter models are tried first, local Ollama is
    the guaranteed fallback.  In tests, the provided mock_llm is used for everything.
    """
    global _ceo
    if _ceo is None:
        if llm is not None:
            # Test / custom mode — single LLM for everything
            _ceo = CEO(llm=llm)
        else:
            from app.ai.model_routing import make_routing_llm
            _ceo = CEO(
                llm=make_routing_llm("ceo_reasoning"),
                selector_llm=make_routing_llm("mode_selection"),
                synthesis_llm=make_routing_llm("synthesis"),
                pattern_llm=make_routing_llm("pattern_discovery"),
            )
    return _ceo
