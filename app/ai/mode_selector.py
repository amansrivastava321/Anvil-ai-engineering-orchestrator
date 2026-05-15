"""Mode selector — AI-driven decision engine for choosing the CEO's operating mode.

phi4-mini:latest is used in production (fast classification task).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Tuple

from app.ai.decision_record import DecisionMode, Problem
from app.ai.intuition import Intuition, Pattern
from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

LLMCallable = Callable[[str, str], Coroutine[Any, Any, str]]

# ── Tight system prompt (~240 tokens) ─────────────────────────────────────────

_SYSTEM_PROMPT = """You are the CEO mode advisor for an AI engineering org.
Choose ONE operating mode for each incoming problem.

Modes:
- decide_alone: known pattern, high confidence, low risk. CEO acts immediately.
- consult_experts: need 1-2 specialist opinions. Moderate complexity or risk.
- convene_council: novel, high-risk, or business-critical. Full team needed.
- strategic_initiative: proactive CEO-initiated work, not a reactive fix.

Respond ONLY with valid JSON (no extra text):
{"mode":"<mode>","reasoning":"<why, max 100 chars>","experts":["<name>"],"confidence":<0-100>}

Expert names: architect, security, performance, testing, memory, domain."""


@dataclass
class ModeSelection:
    mode: DecisionMode
    reasoning: str
    confidence: float
    patterns_matched: List[str]
    experts_suggested: List[str]


class ModeSelector:
    """Asks phi4-mini to classify the CEO's operating mode. No hardcoded rules."""

    def __init__(self, intuition: Intuition, llm: LLMCallable) -> None:
        self._intuition = intuition
        self._llm = llm

    async def select(self, problem: Problem) -> ModeSelection:
        keywords = problem.keywords()

        # Prefer semantic matching; fall back to keyword overlap
        matches = await self._intuition.find_matches_semantic(
            problem.description, threshold=0.70
        )
        if not matches:
            matches = self._intuition.find_matches(keywords, threshold=0.25)

        pattern_ids = [p.id for p, _ in matches[:3]]
        patterns_text = self._format_patterns(matches[:3])
        user_prompt = self._build_prompt(problem, patterns_text)

        from app.ai.model_routing import call_with_json_retry

        result = await call_with_json_retry(
            llm=self._llm,
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            parse_fn=lambda raw: self._try_parse(raw, pattern_ids),
            max_retries=2,
        )

        if result is not None:
            logger.info(
                "AI mode selected",
                mode=result.mode.value,
                confidence=f"{result.confidence:.0%}",
            )
            return result

        logger.warning("Mode selection failed — defaulting to council")
        return ModeSelection(
            mode=DecisionMode.CONVENE_COUNCIL,
            reasoning="Mode selection unavailable. Defaulting to full council.",
            confidence=0.3,
            patterns_matched=pattern_ids,
            experts_suggested=[],
        )

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _format_patterns(self, matches: List[Tuple[Pattern, float]]) -> str:
        if not matches:
            return "No similar patterns. Novel problem."
        parts = []
        for pattern, score in matches:
            parts.append(
                f"- {pattern.description} | confidence {pattern.confidence:.0%} "
                f"| match {score:.0%} | mode: {pattern.preferred_mode}"
            )
        return "\n".join(parts)

    def _build_prompt(self, problem: Problem, patterns_text: str) -> str:
        systems = ", ".join(problem.systems_affected[:3]) if problem.systems_affected else "unknown"
        return (
            f"Problem: {problem.description[:300]}\n"
            f"Risk: {problem.risk_level.value} | "
            f"Critical: {'YES' if problem.is_business_critical else 'no'} | "
            f"Systems: {systems}\n\n"
            f"Past patterns:\n{patterns_text}\n\n"
            "Choose the operating mode. Output JSON only."
        )

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _try_parse(self, raw: str, pattern_ids: List[str]):
        """Return ModeSelection or None on failure."""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            mode = self._map_mode(str(data.get("mode", "consult_experts")).lower())
            experts = [str(e) for e in data.get("experts", [])]
            raw_conf = float(data.get("confidence", 50))
            confidence = max(0.0, min(1.0, raw_conf / 100.0 if raw_conf > 1.0 else raw_conf))
            reasoning = str(data.get("reasoning", "AI-selected mode."))
            return ModeSelection(
                mode=mode,
                reasoning=reasoning,
                confidence=confidence,
                patterns_matched=pattern_ids,
                experts_suggested=experts,
            )
        except Exception as e:
            logger.debug("Mode selector parse failed", error=str(e), raw=raw[:200])
            return None

    # Keep old name for test compatibility
    def _parse_response(self, raw: str, pattern_ids: List[str]) -> ModeSelection:
        result = self._try_parse(raw, pattern_ids)
        if result:
            return result
        return ModeSelection(
            mode=DecisionMode.CONSULT_EXPERTS,
            reasoning="Could not parse AI response.",
            confidence=0.4,
            patterns_matched=pattern_ids,
            experts_suggested=["architect"],
        )

    def _map_mode(self, mode_str: str) -> DecisionMode:
        mapping = {
            "decide_alone": DecisionMode.DECIDE_ALONE,
            "mode_1": DecisionMode.DECIDE_ALONE,
            "mode1": DecisionMode.DECIDE_ALONE,
            "consult_experts": DecisionMode.CONSULT_EXPERTS,
            "mode_2": DecisionMode.CONSULT_EXPERTS,
            "mode2": DecisionMode.CONSULT_EXPERTS,
            "convene_council": DecisionMode.CONVENE_COUNCIL,
            "mode_3": DecisionMode.CONVENE_COUNCIL,
            "mode3": DecisionMode.CONVENE_COUNCIL,
            "strategic_initiative": DecisionMode.STRATEGIC_INITIATIVE,
            "mode_4": DecisionMode.STRATEGIC_INITIATIVE,
            "mode4": DecisionMode.STRATEGIC_INITIATIVE,
        }
        return mapping.get(mode_str, DecisionMode.CONSULT_EXPERTS)
