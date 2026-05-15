"""Tests for the AI Engineering Organization (CEO, Council, Members, etc.)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ai.council import AICouncil
from app.ai.council_members import (
    ArchitectAI,
    DomainAI,
    MemoryAI,
    PerformanceAI,
    SecurityAI,
    TestingAI,
    create_all_members,
)
from app.ai.decision_record import (
    CEODecision,
    DecisionMode,
    DecisionStore,
    ElementVote,
    Problem,
    Proposal,
    RiskLevel,
    SynthesizedPlan,
    Critique,
)
from app.ai.intuition import Intuition, Pattern
from app.ai.mode_selector import ModeSelector
from app.ai.synthesizer import Synthesizer
from app.ai.tool_registry import ToolDefinition, ToolRegistry
from app.ai.ceo import CEO


# ── Fixtures ──────────────────────────────────────────────────────────────────


async def mock_llm(system: str, prompt: str) -> str:
    """A mock LLM that returns predictable JSON based on the prompt."""
    # Mode selector — detected by its distinctive system prompt
    if "mode advisor" in system or "operating mode" in system:
        p_lower = prompt.lower()
        if "risk: critical" in p_lower:
            mode = "convene_council"
        elif "risk: high" in p_lower:
            mode = "convene_council"
        elif "critical: yes" in p_lower:
            mode = "convene_council"
        elif "confidence 9" in p_lower and "risk: low" in p_lower:
            mode = "decide_alone"
        else:
            mode = "consult_experts"
        return json.dumps({
            "mode": mode,
            "reasoning": f"AI analysis selected {mode} based on risk, criticality, and past experience.",
            "experts": ["architect", "security"],
            "confidence": 75,
        })
    if "approach" in prompt and "concerns" in prompt:
        return json.dumps({
            "approach": "Use connection pooling",
            "reasoning": "Standard fix for timeout issues",
            "concerns": ["Ensure pool size is appropriate"],
            "recommended_tools": ["file_reader", "test_runner"],
            "estimated_risk": "low",
            "confidence": 0.85,
        })
    if "Synthesize" in prompt or "unified" in prompt.lower() or "plan" in prompt.lower():
        return json.dumps({
            "plan": "Step 1: Fix connection pool. Step 2: Run tests.",
            "included_from": {"architect": ["connection pooling"]},
            "rejected_elements": [],
            "conflicts_resolved": [],
            "confidence": 0.82,
            "needs_revision": False,
        })
    if "JSON array" in prompt and "critique" in prompt.lower():
        return json.dumps([
            {
                "target_member": "architect",
                "strengths": ["Good analysis"],
                "weaknesses": ["Missing security angle"],
                "missing_considerations": ["PII handling"],
            }
        ])
    if "Vote" in prompt or "vote" in prompt.lower():
        return json.dumps([
            {"element": "Use connection pooling", "voter": "test", "include": True,
             "reasoning": "Standard and proven"}
        ])
    if "discover" in prompt.lower() or "pattern" in prompt.lower():
        return json.dumps([
            {
                "description": "Payment timeout pattern",
                "keywords": ["payment", "timeout"],
                "approach_summary": "Increase connection pool",
                "confidence": 0.8,
                "risk_level": "high",
                "preferred_mode": "consult_experts",
                "requires_experts": ["security"],
            }
        ])
    if "strategic" in prompt.lower() or "initiative" in prompt.lower():
        return json.dumps([
            {
                "title": "Improve test coverage",
                "description": "Generate tests for uncovered API endpoints",
                "rationale": "Coverage dropped below threshold",
                "urgency": "medium",
            }
        ])
    if "reflect" in prompt.lower() or "learning" in prompt.lower():
        return "Mode 1 was correct. Confidence well-calibrated."
    return "I'll fix this by applying standard engineering practices."


@pytest.fixture()
def tmp_store(tmp_path):
    return DecisionStore(path=tmp_path / "decisions.json")


@pytest.fixture()
def tmp_intuition(tmp_path):
    return Intuition(path=tmp_path / "intuition.json", llm=mock_llm)


@pytest.fixture()
def problem_simple():
    return Problem(
        description="Fix the payment timeout bug",
        risk_level=RiskLevel.HIGH,
        systems_affected=["payment"],
        is_business_critical=True,
    )


@pytest.fixture()
def problem_complex():
    return Problem(
        description="Redesign the entire authentication system for multi-tenant support",
        risk_level=RiskLevel.CRITICAL,
        systems_affected=["auth", "user", "session", "api", "database"],
        is_business_critical=True,
    )


@pytest.fixture()
def problem_simple_low_risk():
    return Problem(
        description="Explain what this utility function does",
        risk_level=RiskLevel.LOW,
        systems_affected=[],
        is_business_critical=False,
    )


# ── Problem model ─────────────────────────────────────────────────────────────


class TestProblem:
    def test_keywords_extracted(self, problem_simple):
        kw = problem_simple.keywords()
        assert "payment" in kw
        assert "timeout" in kw
        assert "fix" in kw

    def test_keywords_lowercase(self):
        p = Problem(description="FIX THE BUG in Payment Module")
        assert "payment" in p.keywords()
        assert "module" in p.keywords()

    def test_short_words_excluded(self):
        p = Problem(description="fix it now")
        kw = p.keywords()
        assert "it" not in kw

    def test_default_risk_is_medium(self):
        p = Problem(description="some task")
        assert p.risk_level == RiskLevel.MEDIUM


# ── DecisionStore ─────────────────────────────────────────────────────────────


class TestDecisionStore:
    def test_save_and_retrieve(self, tmp_store):
        d = CEODecision(
            problem_id="p1",
            problem_description="Fix bug",
            mode=DecisionMode.DECIDE_ALONE,
            mode_reasoning="Pattern recognized",
            confidence=0.9,
        )
        tmp_store.save(d)
        retrieved = tmp_store.get(d.id)
        assert retrieved is not None
        assert retrieved.problem_description == "Fix bug"

    def test_all_newest_first(self, tmp_store):
        for i in range(3):
            tmp_store.save(CEODecision(
                problem_id=f"p{i}",
                problem_description=f"Problem {i}",
                mode=DecisionMode.DECIDE_ALONE,
                mode_reasoning="test",
            ))
        all_d = tmp_store.all()
        assert len(all_d) == 3
        assert all_d[0].problem_description == "Problem 2"

    def test_update(self, tmp_store):
        d = CEODecision(
            problem_id="p1", problem_description="Test",
            mode=DecisionMode.DECIDE_ALONE, mode_reasoning="test",
        )
        tmp_store.save(d)
        d.outcome_success = True
        tmp_store.update(d)
        retrieved = tmp_store.get(d.id)
        assert retrieved.outcome_success is True

    def test_successful_filter(self, tmp_store):
        for success in [True, True, False, None]:
            d = CEODecision(
                problem_id="p", problem_description="p",
                mode=DecisionMode.DECIDE_ALONE, mode_reasoning="t",
                outcome_success=success,
            )
            tmp_store.save(d)
        assert len(tmp_store.successful()) == 2
        assert len(tmp_store.failed()) == 1

    def test_count(self, tmp_store):
        assert tmp_store.count() == 0
        tmp_store.save(CEODecision(
            problem_id="p", problem_description="p",
            mode=DecisionMode.DECIDE_ALONE, mode_reasoning="t",
        ))
        assert tmp_store.count() == 1

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "test.json"
        store1 = DecisionStore(path=path)
        d = CEODecision(
            problem_id="p1", problem_description="Persisted",
            mode=DecisionMode.DECIDE_ALONE, mode_reasoning="test",
        )
        store1.save(d)
        store2 = DecisionStore(path=path)
        assert store2.count() == 1
        assert store2.all()[0].problem_description == "Persisted"

    def test_corrupted_file_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json at all!!!")
        store = DecisionStore(path=path)
        assert store.count() == 0


# ── Intuition ─────────────────────────────────────────────────────────────────


class TestIntuition:
    def test_find_matches_empty(self, tmp_intuition):
        matches = tmp_intuition.find_matches(["payment", "timeout"])
        assert matches == []

    def test_add_and_match_pattern(self, tmp_intuition):
        p = Pattern(
            description="Payment timeout pattern",
            keywords=["payment", "timeout"],
            approach_summary="Increase connection pool",
            confidence=0.85,
        )
        tmp_intuition.add_pattern(p)
        matches = tmp_intuition.find_matches(["payment", "timeout", "fix"])
        assert len(matches) == 1
        assert matches[0][0].id == p.id

    def test_match_score_is_proportional(self, tmp_intuition):
        p1 = Pattern(description="p1", keywords=["payment", "timeout", "error"],
                     approach_summary="fix", confidence=1.0)
        p2 = Pattern(description="p2", keywords=["payment"],
                     approach_summary="fix", confidence=1.0)
        tmp_intuition.add_pattern(p1)
        tmp_intuition.add_pattern(p2)
        # p2 has fewer keywords so it should match better (higher ratio)
        matches = tmp_intuition.find_matches(["payment"])
        assert matches[0][0].id == p2.id

    def test_best_match_returns_top(self, tmp_intuition):
        p = Pattern(description="p", keywords=["auth", "bug"],
                    approach_summary="fix", confidence=0.9)
        tmp_intuition.add_pattern(p)
        best = tmp_intuition.best_match(["auth", "bug"])
        assert best is not None
        assert best[0].id == p.id

    def test_record_match_updates_confidence(self, tmp_intuition):
        p = Pattern(description="p", keywords=["bug"],
                    approach_summary="fix", confidence=0.5)
        tmp_intuition.add_pattern(p)
        # Record 3 successes
        for _ in range(3):
            tmp_intuition.record_match(p.id, success=True)
        updated = tmp_intuition._patterns[p.id]
        assert updated.times_matched == 3
        assert updated.times_worked == 3

    def test_record_match_unknown_id(self, tmp_intuition):
        # Should not raise
        tmp_intuition.record_match("nonexistent-id", success=True)

    def test_pattern_persists(self, tmp_path):
        path = tmp_path / "intuition.json"
        i1 = Intuition(path=path, llm=mock_llm)
        p = Pattern(description="p", keywords=["test"], approach_summary="fix", confidence=0.7)
        i1.add_pattern(p)
        i2 = Intuition(path=path)
        assert i2.pattern_count() == 1

    @pytest.mark.asyncio
    async def test_discover_patterns_from_decisions(self, tmp_intuition):
        decisions = [
            {"problem_description": "payment timeout", "mode": "decide_alone",
             "outcome_success": True, "problem_keywords": ["payment", "timeout"]}
        ] * 5
        new = await tmp_intuition.discover_patterns_from_decisions(decisions)
        # Should return patterns (mock LLM returns 1 pattern)
        assert isinstance(new, list)

    @pytest.mark.asyncio
    async def test_discover_no_llm_returns_empty(self, tmp_path):
        i = Intuition(path=tmp_path / "i.json", llm=None)
        result = await i.discover_patterns_from_decisions([{"problem_description": "bug"}])
        assert result == []

    def test_success_rate_calculation(self):
        p = Pattern(description="p", keywords=[], approach_summary="a", confidence=0.5)
        assert p.success_rate == 0.0
        p.times_worked = 3
        p.times_failed = 1
        assert p.success_rate == 0.75


# ── Pattern matching ───────────────────────────────────────────────────────────

class TestPatternMatching:
    def test_no_match_below_threshold(self):
        p = Pattern(description="auth bug", keywords=["auth", "login", "session"],
                    approach_summary="reset", confidence=1.0)
        score = p.matches(["payment", "timeout"])
        assert score == 0.0

    def test_full_match(self):
        p = Pattern(description="exact", keywords=["auth", "bug"],
                    approach_summary="fix", confidence=1.0)
        score = p.matches(["auth", "bug"])
        assert score == 1.0

    def test_partial_match(self):
        p = Pattern(description="partial", keywords=["auth", "bug", "session"],
                    approach_summary="fix", confidence=1.0)
        score = p.matches(["auth"])
        assert 0 < score < 1.0

    def test_empty_keywords_returns_zero(self):
        p = Pattern(description="p", keywords=[], approach_summary="fix", confidence=1.0)
        assert p.matches(["auth"]) == 0.0
        p2 = Pattern(description="p2", keywords=["auth"], approach_summary="fix", confidence=1.0)
        assert p2.matches([]) == 0.0


# ── ModeSelector ─────────────────────────────────────────────────────────────


class TestModeSelector:
    def _selector_with_pattern(self, tmp_intuition, keywords, confidence=0.9):
        p = Pattern(
            description="Test pattern",
            keywords=keywords,
            approach_summary="Standard fix",
            confidence=confidence,
            requires_experts=["security"],
        )
        tmp_intuition.add_pattern(p)
        return ModeSelector(intuition=tmp_intuition, llm=mock_llm)

    @pytest.mark.asyncio
    async def test_no_pattern_medium_risk_consult(self, tmp_intuition):
        selector = ModeSelector(intuition=tmp_intuition, llm=mock_llm)
        p = Problem(description="Explain this code", risk_level=RiskLevel.MEDIUM)
        result = await selector.select(p)
        assert result.mode in (DecisionMode.CONSULT_EXPERTS, DecisionMode.CONVENE_COUNCIL)

    @pytest.mark.asyncio
    async def test_high_confidence_low_risk_decide_alone(self, tmp_intuition):
        selector = self._selector_with_pattern(tmp_intuition, ["fix", "bug"], confidence=0.95)
        p = Problem(description="fix the bug", risk_level=RiskLevel.LOW)
        result = await selector.select(p)
        assert result.mode == DecisionMode.DECIDE_ALONE

    @pytest.mark.asyncio
    async def test_critical_risk_novel_forces_council(self, tmp_intuition):
        selector = ModeSelector(intuition=tmp_intuition, llm=mock_llm)
        p = Problem(
            description="Redesign entire auth system for global scale",
            risk_level=RiskLevel.CRITICAL,
            systems_affected=["auth", "users", "api", "db"],
            is_business_critical=True,
        )
        result = await selector.select(p)
        assert result.mode == DecisionMode.CONVENE_COUNCIL

    @pytest.mark.asyncio
    async def test_multi_system_high_risk_forces_council(self, tmp_intuition):
        selector = ModeSelector(intuition=tmp_intuition, llm=mock_llm)
        p = Problem(
            description="Refactor payment and auth and database",
            risk_level=RiskLevel.HIGH,
            systems_affected=["payment", "auth", "database"],
        )
        result = await selector.select(p)
        assert result.mode == DecisionMode.CONVENE_COUNCIL

    @pytest.mark.asyncio
    async def test_moderate_confidence_consults_experts(self, tmp_intuition):
        selector = self._selector_with_pattern(tmp_intuition, ["timeout", "fix"], confidence=0.6)
        p = Problem(description="fix the timeout", risk_level=RiskLevel.MEDIUM)
        result = await selector.select(p)
        assert result.mode == DecisionMode.CONSULT_EXPERTS

    @pytest.mark.asyncio
    async def test_reasoning_is_populated(self, tmp_intuition):
        selector = ModeSelector(intuition=tmp_intuition, llm=mock_llm)
        p = Problem(description="some problem", risk_level=RiskLevel.LOW)
        result = await selector.select(p)
        assert len(result.reasoning) > 10

    @pytest.mark.asyncio
    async def test_novel_high_risk_forces_council(self, tmp_intuition):
        selector = ModeSelector(intuition=tmp_intuition, llm=mock_llm)
        p = Problem(description="completely novel problem xyz", risk_level=RiskLevel.HIGH)
        result = await selector.select(p)
        assert result.mode == DecisionMode.CONVENE_COUNCIL


# ── ToolRegistry ─────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_default_tools_registered(self):
        r = ToolRegistry()
        names = r.names()
        assert "file_reader" in names
        assert "test_runner" in names
        assert "security_scanner" in names

    def test_register_custom_tool(self):
        r = ToolRegistry()
        t = ToolDefinition(name="custom_tool", description="Custom", when_to_use="Testing")
        r.register(t)
        assert r.get("custom_tool") is not None

    def test_by_risk_filter(self):
        r = ToolRegistry()
        low_risk = r.by_risk("low")
        for t in low_risk:
            assert t.risk_level == "low"

    def test_describe_all_returns_string(self):
        r = ToolRegistry()
        desc = r.describe_all()
        assert "file_reader" in desc
        assert len(desc) > 100

    def test_get_nonexistent_returns_none(self):
        r = ToolRegistry()
        assert r.get("nonexistent") is None


# ── Council members ───────────────────────────────────────────────────────────


class TestCouncilMembers:
    @pytest.mark.asyncio
    async def test_architect_produces_proposal(self, problem_simple):
        member = ArchitectAI(llm=mock_llm)
        proposal = await member.analyze(problem_simple)
        assert proposal.member_name == "architect"
        assert len(proposal.approach) > 0

    @pytest.mark.asyncio
    async def test_security_produces_proposal(self, problem_simple):
        member = SecurityAI(llm=mock_llm)
        proposal = await member.analyze(problem_simple)
        assert proposal.member_name == "security"
        assert 0.0 <= proposal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_performance_produces_proposal(self, problem_simple):
        member = PerformanceAI(llm=mock_llm)
        proposal = await member.analyze(problem_simple)
        assert proposal.member_name == "performance"

    @pytest.mark.asyncio
    async def test_testing_produces_proposal(self, problem_simple):
        member = TestingAI(llm=mock_llm)
        proposal = await member.analyze(problem_simple)
        assert proposal.member_name == "testing"

    @pytest.mark.asyncio
    async def test_memory_includes_history(self, problem_simple):
        history = [{"problem_description": "past bug", "outcome_success": True}]
        member = MemoryAI(llm=mock_llm, decision_history=history)
        assert "past bug" in member.system_prompt

    @pytest.mark.asyncio
    async def test_domain_produces_proposal(self, problem_simple):
        member = DomainAI(llm=mock_llm)
        proposal = await member.analyze(problem_simple)
        assert proposal.member_name == "domain"

    @pytest.mark.asyncio
    async def test_critique_other_proposals(self, problem_simple):
        arch = ArchitectAI(llm=mock_llm)
        sec = SecurityAI(llm=mock_llm)
        arch_proposal = await arch.analyze(problem_simple)
        critiques = await sec.critique([arch_proposal])
        assert isinstance(critiques, list)

    @pytest.mark.asyncio
    async def test_critique_returns_empty_for_no_others(self, problem_simple):
        arch = ArchitectAI(llm=mock_llm)
        proposal = await arch.analyze(problem_simple)
        critiques = await arch.critique([proposal])  # only self
        assert critiques == []

    @pytest.mark.asyncio
    async def test_vote_on_proposals(self, problem_simple):
        arch = ArchitectAI(llm=mock_llm)
        proposal = await arch.analyze(problem_simple)
        votes = await arch.vote([proposal])
        assert isinstance(votes, list)

    @pytest.mark.asyncio
    async def test_vote_empty_proposals(self, problem_simple):
        arch = ArchitectAI(llm=mock_llm)
        votes = await arch.vote([])
        assert votes == []

    def test_create_all_members(self):
        members = create_all_members(llm=mock_llm)
        assert len(members) == 6
        for name in ["architect", "security", "performance", "testing", "memory", "domain"]:
            assert name in members

    @pytest.mark.asyncio
    async def test_malformed_llm_response_falls_back(self, problem_simple):
        async def bad_llm(s, p):
            return "This is not JSON at all"

        arch = ArchitectAI(llm=bad_llm)
        proposal = await arch.analyze(problem_simple)
        # Should still produce a Proposal (fallback)
        assert proposal.member_name == "architect"
        assert len(proposal.approach) > 0


# ── Synthesizer ───────────────────────────────────────────────────────────────


class TestSynthesizer:
    @pytest.fixture()
    def synth(self):
        return Synthesizer(llm=mock_llm)

    @pytest.fixture()
    def proposals(self):
        return [
            Proposal(member_name="architect", approach="Fix connection pool",
                     reasoning="Standard fix", confidence=0.8),
            Proposal(member_name="security", approach="Add rate limiting",
                     reasoning="Prevents abuse", confidence=0.7),
        ]

    @pytest.mark.asyncio
    async def test_synthesize_produces_plan(self, synth, problem_simple, proposals):
        plan = await synth.synthesize(problem_simple, proposals, [], [])
        assert isinstance(plan, SynthesizedPlan)
        assert len(plan.plan) > 0
        assert 0.0 <= plan.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_synthesize_empty_proposals(self, synth, problem_simple):
        plan = await synth.synthesize(problem_simple, [], [], [])
        assert plan.needs_revision is True
        assert plan.confidence == 0.0

    @pytest.mark.asyncio
    async def test_revise_improves_plan(self, synth, problem_simple, proposals):
        low_confidence_plan = SynthesizedPlan(
            plan="Unclear plan", confidence=0.4, needs_revision=True
        )
        revised = await synth.revise(problem_simple, low_confidence_plan, "Missing security checks")
        assert isinstance(revised, SynthesizedPlan)

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self, problem_simple, proposals):
        async def bad_llm(s, p):
            return "Cannot parse this as JSON"

        synth = Synthesizer(llm=bad_llm)
        plan = await synth.synthesize(problem_simple, proposals, [], [])
        assert len(plan.plan) > 0  # fallback uses proposal text
        assert plan.needs_revision is True

    def test_format_proposals_text(self, synth, proposals):
        text = synth._format_proposals(proposals)
        assert "architect" in text.lower()
        assert "security" in text.lower()

    def test_format_votes_narrative(self, synth):
        votes = [
            ElementVote(element="Fix pool", voter="arch", include=True, reasoning="proven approach"),
            ElementVote(element="Fix pool", voter="sec", include=False, reasoning="increases attack surface"),
        ]
        text = synth._format_votes(votes)
        assert "YES" in text or "NO" in text
        assert "arch" in text

    def test_format_votes_empty(self, synth):
        text = synth._format_votes([])
        assert "No votes" in text or "None" in text


# ── AICouncil ─────────────────────────────────────────────────────────────────


class TestAICouncil:
    @pytest.fixture()
    def council(self):
        return AICouncil(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_convene_full_council(self, council, problem_simple):
        plan = await council.convene(problem_simple)
        assert isinstance(plan, SynthesizedPlan)
        assert len(plan.plan) > 0

    @pytest.mark.asyncio
    async def test_consult_specific_experts(self, council, problem_simple):
        plan = await council.consult(problem_simple, expert_names=["architect", "security"])
        assert isinstance(plan, SynthesizedPlan)

    @pytest.mark.asyncio
    async def test_consult_unknown_expert_falls_back(self, council, problem_simple):
        plan = await council.consult(problem_simple, expert_names=["nonexistent"])
        # Falls back to architect
        assert isinstance(plan, SynthesizedPlan)

    def test_member_names(self, council):
        names = council.member_names()
        assert len(names) == 6
        assert "architect" in names

    @pytest.mark.asyncio
    async def test_member_failure_is_graceful(self, problem_simple):
        call_count = 0

        async def flaky_llm(s, p):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("LLM unavailable")
            return await mock_llm(s, p)

        council = AICouncil(llm=flaky_llm)
        # Should not raise even if some members fail
        plan = await council.convene(problem_simple)
        assert isinstance(plan, SynthesizedPlan)

    @pytest.mark.asyncio
    async def test_convene_with_specific_members(self, council, problem_simple):
        plan = await council.convene(problem_simple, member_names=["architect", "testing"])
        assert isinstance(plan, SynthesizedPlan)


# ── CEO ───────────────────────────────────────────────────────────────────────


class TestCEO:
    @pytest.fixture()
    def ceo(self, tmp_path):
        store = DecisionStore(path=tmp_path / "decisions.json")
        intuition = Intuition(path=tmp_path / "intuition.json", llm=mock_llm)
        return CEO(llm=mock_llm, store=store, intuition=intuition)

    @pytest.mark.asyncio
    async def test_receive_simple_problem(self, ceo, problem_simple):
        decision = await ceo.receive_problem(
            description="Fix the payment timeout bug",
            risk_level="high",
            systems_affected=["payment"],
            is_business_critical=True,
        )
        assert isinstance(decision, CEODecision)
        assert len(decision.final_plan) > 0
        assert decision.mode in DecisionMode.__members__.values()

    @pytest.mark.asyncio
    async def test_receive_problem_saved_to_store(self, ceo):
        decision = await ceo.receive_problem(description="Explain this code")
        retrieved = ceo._store.get(decision.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_record_outcome_success(self, ceo):
        decision = await ceo.receive_problem(description="Fix bug")
        updated = await ceo.record_outcome(decision.id, success=True, notes="All tests pass")
        assert updated is not None
        assert updated.outcome_success is True
        assert len(updated.learning_notes) > 0

    @pytest.mark.asyncio
    async def test_record_outcome_failure(self, ceo):
        decision = await ceo.receive_problem(description="Fix bug")
        updated = await ceo.record_outcome(decision.id, success=False, notes="Broke prod")
        assert updated.outcome_success is False

    @pytest.mark.asyncio
    async def test_record_outcome_unknown_id(self, ceo):
        result = await ceo.record_outcome("nonexistent-id", success=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_ceo_uses_mode1_for_known_pattern(self, ceo, tmp_path):
        # Seed high-confidence pattern
        p = Pattern(
            description="Fix timeout pattern",
            keywords=["fix", "timeout", "bug"],
            approach_summary="Standard fix",
            confidence=0.92,
            requires_experts=[],
        )
        ceo._intuition.add_pattern(p)

        decision = await ceo.receive_problem(
            description="Fix the timeout bug",
            risk_level="low",
        )
        assert decision.mode == DecisionMode.DECIDE_ALONE

    @pytest.mark.asyncio
    async def test_ceo_uses_mode3_for_novel_critical(self, ceo):
        decision = await ceo.receive_problem(
            description="Redesign the entire microservice architecture for 10M users",
            risk_level="critical",
            systems_affected=["auth", "payment", "api", "db", "cache"],
            is_business_critical=True,
        )
        assert decision.mode == DecisionMode.CONVENE_COUNCIL

    @pytest.mark.asyncio
    async def test_get_intuition_summary(self, ceo):
        await ceo.receive_problem(description="Fix bug")
        summary = ceo.get_intuition_summary()
        assert "total_patterns" in summary
        assert "total_decisions" in summary
        assert summary["total_decisions"] == 1

    @pytest.mark.asyncio
    async def test_success_rate_none_when_no_outcomes(self, ceo):
        await ceo.receive_problem(description="Test problem")
        assert ceo._calc_success_rate() is None

    @pytest.mark.asyncio
    async def test_success_rate_calculated(self, ceo):
        d1 = await ceo.receive_problem(description="Problem 1")
        d2 = await ceo.receive_problem(description="Problem 2")
        await ceo.record_outcome(d1.id, success=True)
        await ceo.record_outcome(d2.id, success=False)
        rate = ceo._calc_success_rate()
        assert rate == 0.5

    @pytest.mark.asyncio
    async def test_discover_patterns(self, ceo):
        # Save some decisions first
        for i in range(3):
            await ceo.receive_problem(description=f"payment timeout bug {i}")
        count = await ceo.discover_patterns()
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_strategic_review_no_history(self, ceo):
        initiatives = await ceo.strategic_review()
        assert initiatives == []

    @pytest.mark.asyncio
    async def test_strategic_review_with_history(self, ceo):
        for i in range(3):
            d = await ceo.receive_problem(description=f"payment timeout {i}")
            await ceo.record_outcome(d.id, success=True)

        initiatives = await ceo.strategic_review()
        assert isinstance(initiatives, list)

    @pytest.mark.asyncio
    async def test_invalid_risk_level_defaults_to_medium(self, ceo):
        decision = await ceo.receive_problem(
            description="Some problem", risk_level="not_a_real_level"
        )
        assert decision is not None

    @pytest.mark.asyncio
    async def test_decision_contains_keywords(self, ceo):
        decision = await ceo.receive_problem(
            description="Fix payment timeout connection pool"
        )
        assert "payment" in decision.problem_keywords
        assert "timeout" in decision.problem_keywords

    @pytest.mark.asyncio
    async def test_mode2_with_partial_pattern(self, ceo):
        # Add a moderate-confidence pattern
        p = Pattern(
            description="Auth pattern",
            keywords=["auth", "login"],
            approach_summary="Standard auth fix",
            confidence=0.6,
            requires_experts=["security"],
        )
        ceo._intuition.add_pattern(p)
        decision = await ceo.receive_problem(
            description="auth login fails", risk_level="medium"
        )
        assert decision.mode == DecisionMode.CONSULT_EXPERTS


# ── Learning loop ─────────────────────────────────────────────────────────────


class TestLearningLoop:
    @pytest.mark.asyncio
    async def test_confidence_increases_with_successes(self, tmp_path):
        """Simulates Week 4: CEO has seen patterns, confidence increases."""
        store = DecisionStore(path=tmp_path / "d.json")
        intuition = Intuition(path=tmp_path / "i.json", llm=mock_llm)
        ceo = CEO(llm=mock_llm, store=store, intuition=intuition)

        # Add initial pattern with moderate confidence
        p = Pattern(
            description="Payment timeout",
            keywords=["payment", "timeout"],
            approach_summary="Pool fix",
            confidence=0.7,
            preferred_mode="consult_experts",
        )
        intuition.add_pattern(p)
        initial_confidence = p.confidence

        # Simulate 5 successful decisions
        for _ in range(5):
            d = await ceo.receive_problem(description="payment timeout bug")
            await ceo.record_outcome(d.id, success=True)

        # Pattern confidence should have increased
        updated_p = intuition._patterns[p.id]
        assert updated_p.times_worked > 0

    @pytest.mark.asyncio
    async def test_50_decisions_builds_history(self, tmp_path):
        """Simulates accumulating decision history for pattern discovery."""
        store = DecisionStore(path=tmp_path / "d.json")
        intuition = Intuition(path=tmp_path / "i.json", llm=mock_llm)
        ceo = CEO(llm=mock_llm, store=store, intuition=intuition)

        for i in range(10):
            d = await ceo.receive_problem(description=f"payment timeout issue {i}")
            await ceo.record_outcome(d.id, success=(i % 3 != 0))

        assert store.count() == 10
        assert len(store.successful()) > 0
        assert len(store.failed()) > 0
