"""app.ai — Autonomous AI Engineering Organization.

Public API surface:
    CEO            - Chief Engineering Officer (main entry point)
    AICouncil      - Council debate system
    DecisionMode   - Operating modes (1-4)
    Problem        - Problem descriptor
    get_ceo()      - Singleton CEO factory
"""
from app.ai.ceo import CEO, get_ceo
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
    Problem,
    Proposal,
    RiskLevel,
    SynthesizedPlan,
    get_decision_store,
)
from app.ai.intuition import Intuition, Pattern
from app.ai.mode_selector import ModeSelection, ModeSelector
from app.ai.synthesizer import Synthesizer
from app.ai.tool_registry import ToolRegistry, get_tool_registry

__all__ = [
    # CEO
    "CEO",
    "get_ceo",
    # Council
    "AICouncil",
    "ArchitectAI",
    "SecurityAI",
    "PerformanceAI",
    "TestingAI",
    "MemoryAI",
    "DomainAI",
    "create_all_members",
    "Synthesizer",
    # Data models
    "CEODecision",
    "DecisionMode",
    "DecisionStore",
    "Problem",
    "Proposal",
    "RiskLevel",
    "SynthesizedPlan",
    "get_decision_store",
    # Intuition
    "Intuition",
    "Pattern",
    # Mode selection
    "ModeSelection",
    "ModeSelector",
    # Tools
    "ToolRegistry",
    "get_tool_registry",
]
