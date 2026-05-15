"""
Enterprise-grade context assembly service for AI agent interactions.

This service is the intelligence layer that:
- Assembles optimal prompts from multiple sources
- Manages token budgets with intelligent allocation
- Compresses context when exceeding model limits
- Prioritizes relevant Graphify intelligence
- Injects skillfile-managed AI skills for better small-model performance
- Implements retrieval-augmented context building
- Tracks context usage and efficiency metrics
- Handles multi-modal context (code, docs, graphs, skills)
- Provides context caching for repeated queries

This is the core "brain" that determines what the AI model sees.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache

import structlog
from pydantic import BaseModel, Field, field_validator

from app.core.config.settings import settings
from app.core.monitoring.logging import get_logger, log_function_call
from app.core.monitoring.metrics import (
    graphify_context_size,
    graphify_parsing_duration,
    MetricsTracker,
)
from app.integrations.graphify.parser import GraphifyParser, get_default_parser
from app.integrations.skillfile.client import SkillfileClient, get_skillfile_client
from app.utils.validators import InputSanitizer, PathValidator
from app.utils.retry import async_retry

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ContextSource(str, Enum):
    """Sources of context information."""
    GRAPHIFY_SUMMARY = "graphify_summary"
    GRAPHIFY_GRAPH = "graphify_graph"
    GRAPHIFY_APP_MAP = "graphify_app_map"
    SKILLFILE_SKILL = "skillfile_skill"       # 🆕 skillfile-managed skills
    CODE_FILE = "code_file"
    DOCUMENTATION = "documentation"
    TEST_FILE = "test_file"
    CONFIG_FILE = "config_file"
    DEPENDENCY_LIST = "dependency_list"
    USER_PROMPT = "user_prompt"
    SYSTEM_INSTRUCTION = "system_instruction"
    CONVERSATION_HISTORY = "conversation_history"
    AGENT_MEMORY = "agent_memory"


class ContextPriority(str, Enum):
    """Priority levels for context inclusion."""
    CRITICAL = "critical"     # Always included (user prompt, skills for the task)
    HIGH = "high"             # Essential context (Graphify summary, direct deps)
    MEDIUM = "medium"         # Helpful context (app map, related files)
    LOW = "low"               # Supplementary (documentation, examples)
    OPTIONAL = "optional"     # Only if budget allows


class ContextMode(str, Enum):
    """Context assembly modes for different use cases."""
    PRECISE = "precise"       # Minimal context, focused on specific task
    BALANCED = "balanced"     # Moderate context, good for most tasks
    COMPREHENSIVE = "comprehensive"  # Maximum context for complex analysis
    CODE_ONLY = "code_only"   # Only code, no documentation/analysis
    DIAGRAM = "diagram"       # Focus on architecture/relationships


@dataclass
class ContextChunk:
    """A piece of context with metadata."""
    content: str
    source: ContextSource
    priority: ContextPriority
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    relevance_score: float = 1.0
    skill_name: Optional[str] = None       # 🆕 For skillfile skills
    skill_source: Optional[str] = None     # 🆕 github/local/url
    
    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = self._estimate_tokens(self.content)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Quick token estimation (rough: 1 token ≈ 4 chars)."""
        return len(text) // 4


@dataclass
class ContextBudget:
    """Token budget for context assembly."""
    total_limit: int
    system_instructions: int = 0
    conversation_history: int = 0
    user_prompt: int = 0
    graphify_context: int = 0
    skill_context: int = 0          # 🆕 Budget for injected skills
    code_files: int = 0
    documentation: int = 0
    agent_memory: int = 0
    reserved_safety: int = 200  # Safety margin
    
    @property
    def allocated(self) -> int:
        """Total allocated tokens."""
        return (
            self.system_instructions +
            self.conversation_history +
            self.user_prompt +
            self.graphify_context +
            self.skill_context +
            self.code_files +
            self.documentation +
            self.agent_memory
        )
    
    @property
    def available(self) -> int:
        """Available tokens remaining."""
        return max(0, self.total_limit - self.allocated - self.reserved_safety)
    
    def allocate(self, category: str, amount: int) -> bool:
        """Allocate tokens to a category. Returns True if successful."""
        if not hasattr(self, category):
            logger.warning(f"Unknown budget category: {category}")
            return False
        current = getattr(self, category, 0)
        if self.available >= amount:
            setattr(self, category, current + amount)
            return True
        return False


@dataclass
class AssembledContext:
    """Final assembled context ready for model consumption."""
    system_prompt: str
    user_prompt: str
    context_chunks: List[ContextChunk]
    total_tokens: int
    budget_used: Dict[str, int]
    graphify_available: bool
    skills_injected: int           # 🆕 Number of skills injected
    skill_names: List[str]         # 🆕 Names of injected skills
    files_included: List[str]
    assembly_time_ms: float
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Context Assembler
# ============================================================================

class ContextAssembler:
    """
    Intelligent context assembly with token budgeting, prioritization,
    and skillfile skill injection.
    
    Integrates:
    - Graphify knowledge graphs for repository intelligence
    - skillfile for declarative AI skill management
    - File system for direct code context
    - Conversation history for continuity
    """
    
    def __init__(
        self,
        parser: Optional[GraphifyParser] = None,
        skillfile_client: Optional[SkillfileClient] = None,
    ):
        """
        Initialize context assembler.
        
        Args:
            parser: GraphifyParser instance (uses default if not provided)
            skillfile_client: SkillfileClient instance (uses default if not provided)
        """
        self.parser = parser or get_default_parser()
        self.skillfile = skillfile_client or get_skillfile_client()
        
        # Context cache for repeated queries
        self._context_cache: Dict[str, Tuple[AssembledContext, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Skill injection control
        self._skill_injection_enabled = True
        self._max_skills_per_task = 3
        self._max_skill_tokens_per_skill = 1500
        # Evolution-managed context strategies per task/workflow type
        self._context_strategies: Dict[str, Any] = {}
        # Evolution-managed preferred skill combinations
        self._preferred_skill_combinations: list = []
        
        # Statistics
        self._assembly_count = 0
        self._total_tokens_used = 0
        self._average_tokens = 0
        self._skills_injected_total = 0
        
        logger.info(
            "Context assembler initialized",
            skill_injection=self._skill_injection_enabled,
            graphify_available=True,
        )
    
    # ========================================
    # Configuration
    # ========================================
    
    def enable_skill_injection(self, enabled: bool = True) -> None:
        """Enable or disable skill injection."""
        self._skill_injection_enabled = enabled
        logger.info("Skill injection", enabled=enabled)
    
    def set_skill_limits(
        self,
        max_skills: Optional[int] = None,
        max_tokens_per_skill: Optional[int] = None,
    ) -> None:
        """Configure skill injection limits."""
        if max_skills is not None:
            self._max_skills_per_task = max_skills
        if max_tokens_per_skill is not None:
            self._max_skill_tokens_per_skill = max_tokens_per_skill

    def update_context_strategy(
        self, task_type: str, strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store an evolution-derived context assembly strategy for a task type.

        Called by EvolutionService when empirical data suggests a better mode.
        Returns the previous strategy dict for rollback.

        Args:
            task_type: workflow or task type key (e.g., ``"debug"``, ``"audit"``)
            strategy: dict describing the preferred context behaviour,
                      e.g. ``{"preferred_mode": "comprehensive"}``
        """
        previous = self._context_strategies.get(task_type, {})
        self._context_strategies[task_type] = strategy
        logger.info(
            "Context strategy updated by evolution",
            task_type=task_type,
            strategy=strategy,
        )
        return previous

    def get_context_strategy(self, task_type: str) -> Dict[str, Any]:
        """Return the current evolution-managed context strategy for a task type."""
        return self._context_strategies.get(task_type, {})
    
    # ========================================
    # Main Assembly Methods
    # ========================================
    
    @log_function_call(level="INFO", log_time=True)
    async def assemble_context(
        self,
        user_prompt: str,
        repo_path: str,
        task_type: str = "code",
        mode: ContextMode = ContextMode.BALANCED,
        max_tokens: Optional[int] = None,
        include_files: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        force_refresh: bool = False,
        inject_skills: Optional[bool] = None,  # 🆕 Override skill injection
    ) -> AssembledContext:
        """
        Assemble optimized context for an AI agent interaction.
        
        Args:
            user_prompt: The user's question or request
            repo_path: Path to the repository
            task_type: Type of task (code, architecture, debug, etc.)
            mode: Context assembly mode (precise, balanced, comprehensive)
            max_tokens: Override maximum context tokens
            include_files: Specific files to include
            conversation_history: Previous conversation messages
            force_refresh: Bypass cache
            inject_skills: Override skill injection setting
            
        Returns:
            Assembled context ready for model consumption
            
        Raises:
            ValueError: If context cannot be assembled within budget
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []
        
        # Determine if skills should be injected
        use_skills = (
            self._skill_injection_enabled
            if inject_skills is None
            else inject_skills
        )
        
        # Validate inputs
        repo_path_obj = PathValidator.validate_path(
            repo_path,
            must_exist=True,
            must_be_dir=True,
        )
        
        sanitized_prompt = InputSanitizer.sanitize_string(
            user_prompt,
            field_name="user_prompt",
            max_length=10000,
        )
        
        # Determine token budget (allocate more for skills if enabled)
        token_limit = max_tokens or self._get_default_token_limit(mode)
        if use_skills:
            # Increase budget slightly to accommodate skills
            token_limit = min(token_limit + 2000, 32768)
        
        budget = self._create_budget(token_limit, task_type, mode, use_skills)
        
        # Check cache for identical context
        cache_key = self._generate_cache_key(
            sanitized_prompt,
            str(repo_path_obj),
            mode.value,
            tuple(include_files or []),
            use_skills,
        )
        
        if not force_refresh and cache_key in self._context_cache:
            cached_context, timestamp = self._context_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                logger.debug("Context cache hit", cache_key=cache_key[:20])
                return cached_context
        
        # Collect context chunks from various sources
        chunks: List[ContextChunk] = []
        
        # 1. User prompt (CRITICAL - always included)
        user_chunk = ContextChunk(
            content=sanitized_prompt,
            source=ContextSource.USER_PROMPT,
            priority=ContextPriority.CRITICAL,
        )
        chunks.append(user_chunk)
        budget.user_prompt = user_chunk.token_estimate
        
        # 2. Load skillfile skills (CRITICAL - injected early for best effect)
        skill_names: List[str] = []
        if use_skills:
            skill_chunks, skill_names = await self._load_skill_context(
                repo_path_obj,
                task_type,
                budget,
                mode,
            )
            chunks.extend(skill_chunks)
        
        # 3. Load Graphify context
        graphify_chunks = await self._load_graphify_context(
            repo_path_obj,
            task_type,
            budget,
            mode,
        )
        chunks.extend(graphify_chunks)
        
        # 4. Include specific files if requested
        if include_files:
            file_chunks = await self._load_file_context(
                repo_path_obj,
                include_files,
                budget,
            )
            chunks.extend(file_chunks)
        
        # 5. Include conversation history
        if conversation_history:
            history_chunks = self._load_conversation_history(
                conversation_history,
                budget,
            )
            chunks.extend(history_chunks)
        
        # 6. Add agent memory if available
        memory_chunks = await self._load_agent_memory(
            repo_path_obj,
            task_type,
            budget,
        )
        chunks.extend(memory_chunks)
        
        # Sort by priority (critical first, optional last)
        chunks.sort(key=lambda c: self._priority_sort_order(c.priority))
        
        # Assemble final context
        system_prompt = self._build_system_prompt(task_type, mode, use_skills)
        budget.system_instructions = len(system_prompt) // 4
        
        # Build user-facing prompt with context
        context_text = self._merge_chunks(chunks)
        full_user_prompt = self._wrap_user_prompt(
            sanitized_prompt,
            context_text,
            use_skills,
        )
        
        # Calculate final stats
        total_tokens = sum(c.token_estimate for c in chunks) + budget.system_instructions
        assembly_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Check if within budget
        if total_tokens > token_limit:
            warnings.append(
                f"Context exceeds token limit ({total_tokens} > {token_limit}). "
                f"Some content may be truncated."
            )
            # Emergency truncation
            chunks = self._emergency_truncate(chunks, token_limit - budget.system_instructions)
            context_text = self._merge_chunks(chunks)
            full_user_prompt = self._wrap_user_prompt(
                sanitized_prompt,
                context_text,
                use_skills,
            )
            total_tokens = sum(c.token_estimate for c in chunks) + budget.system_instructions
        
        # Create assembled context
        assembled = AssembledContext(
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            context_chunks=chunks,
            total_tokens=total_tokens,
            budget_used={
                "system": budget.system_instructions,
                "user_prompt": budget.user_prompt,
                "graphify": budget.graphify_context,
                "skills": budget.skill_context,
                "code_files": budget.code_files,
                "documentation": budget.documentation,
                "history": budget.conversation_history,
                "memory": budget.agent_memory,
            },
            graphify_available=any(
                c.source in (
                    ContextSource.GRAPHIFY_SUMMARY,
                    ContextSource.GRAPHIFY_GRAPH,
                    ContextSource.GRAPHIFY_APP_MAP,
                )
                for c in chunks
            ),
            skills_injected=len(skill_names),
            skill_names=skill_names,
            files_included=[
                c.file_path for c in chunks
                if c.file_path and c.source == ContextSource.CODE_FILE
            ],
            assembly_time_ms=assembly_time,
            warnings=warnings,
        )
        
        # Update statistics
        self._update_stats(assembled)
        
        # Cache the result
        self._context_cache[cache_key] = (assembled, datetime.utcnow())
        
        logger.info(
            "Context assembled",
            total_tokens=total_tokens,
            chunks=len(chunks),
            graphify_available=assembled.graphify_available,
            skills_injected=assembled.skills_injected,
            files=len(assembled.files_included),
            time_ms=round(assembly_time, 2),
            warnings=len(warnings),
        )
        
        return assembled
    
    # ========================================
    # Skill Context Loader (🆕)
    # ========================================
    
    async def _load_skill_context(
        self,
        repo_path: Path,
        task_type: str,
        budget: ContextBudget,
        mode: ContextMode,
    ) -> Tuple[List[ContextChunk], List[str]]:
        """
        Load relevant skills from skillfile and inject into context.
        
        Args:
            repo_path: Repository path
            task_type: Type of task
            budget: Token budget
            mode: Context mode
            
        Returns:
            Tuple of (skill chunks, skill names)
        """
        chunks: List[ContextChunk] = []
        skill_names: List[str] = []
        
        if mode == ContextMode.CODE_ONLY:
            return chunks, skill_names
        
        if not self.skillfile.is_installed:
            logger.debug("skillfile not installed, skipping skill injection")
            return chunks, skill_names
        
        try:
            # Try to install/update skills first
            await self.skillfile.install_skills(update=False)
            
            # Load skills relevant to this task
            skills = await self.skillfile.load_skills(task_type)
            
            if not skills:
                logger.debug("No skills found for task", task_type=task_type)
                return chunks, skill_names
            
            # Allocate budget for skills
            if mode == ContextMode.PRECISE:
                skill_budget = min(budget.available, 1000)
            elif mode == ContextMode.BALANCED:
                skill_budget = min(budget.available // 4, 2000)
            elif mode == ContextMode.COMPREHENSIVE:
                skill_budget = min(budget.available // 3, 4000)
            else:
                skill_budget = min(budget.available // 4, 1500)
            
            # Inject skills (most relevant first)
            skills_injected = 0
            tokens_used = 0
            
            for skill in skills[:self._max_skills_per_task]:
                # Check budget
                if tokens_used + skill.token_estimate > skill_budget:
                    # Truncate to fit remaining budget
                    remaining = skill_budget - tokens_used
                    if remaining < 200:  # Too small to be useful
                        break
                    truncated_content = self._truncate_to_tokens(
                        skill.content,
                        remaining,
                    )
                    skill_content = truncated_content
                    token_estimate = remaining
                else:
                    skill_content = skill.content
                    token_estimate = skill.token_estimate
                
                # Create skill chunk
                chunk = ContextChunk(
                    content=self._format_skill_content(
                        skill_content,
                        skill.entry.name,
                    ),
                    source=ContextSource.SKILLFILE_SKILL,
                    priority=ContextPriority.CRITICAL,  # Skills are critical
                    token_estimate=token_estimate,
                    metadata={
                        "skill_name": skill.entry.name,
                        "skill_source": skill.entry.source.value,
                        "skill_type": skill.entry.skill_type.value,
                    },
                    skill_name=skill.entry.name,
                    skill_source=skill.entry.source.value,
                )
                
                chunks.append(chunk)
                skill_names.append(skill.entry.name)
                budget.allocate("skill_context", token_estimate)
                
                skills_injected += 1
                tokens_used += token_estimate
            
            if skills_injected > 0:
                self._skills_injected_total += skills_injected
                logger.info(
                    "Skills injected into context",
                    task_type=task_type,
                    skills_count=skills_injected,
                    tokens=tokens_used,
                    skill_names=skill_names,
                )
            
        except Exception as e:
            logger.warning(
                "Failed to load skill context",
                error=str(e),
                task_type=task_type,
            )
        
        return chunks, skill_names
    
    def _format_skill_content(self, content: str, skill_name: str) -> str:
        """Format skill content for context injection."""
        return (
            f"## EXPERT INSTRUCTIONS: {skill_name}\n\n"
            f"The following expert guidelines should be followed for this task:\n\n"
            f"{content}\n\n"
            f"---\n"
        )
    
    # ========================================
    # Source Loaders
    # ========================================
    
    async def _load_graphify_context(
        self,
        repo_path: Path,
        task_type: str,
        budget: ContextBudget,
        mode: ContextMode,
    ) -> List[ContextChunk]:
        """Load and prioritize Graphify context."""
        chunks: List[ContextChunk] = []
        
        if mode == ContextMode.CODE_ONLY:
            return chunks
        
        try:
            # Parse Graphify output
            graphify_output = await self.parser.parse_repository(str(repo_path))
            
            if not graphify_output.is_valid:
                logger.debug("No valid Graphify output found")
                return chunks
            
            # 1. Summary (HIGH priority - always include if available)
            if graphify_output.has_summary and mode != ContextMode.CODE_ONLY:
                summary = graphify_output.summary
                summary_text = self._format_summary(summary, mode)
                
                # Truncate based on budget
                max_summary_tokens = self._get_graphify_budget(budget, mode)
                if max_summary_tokens > 0:
                    summary_text = self._truncate_to_tokens(
                        summary_text,
                        max_summary_tokens,
                    )
                    
                    chunk = ContextChunk(
                        content=summary_text,
                        source=ContextSource.GRAPHIFY_SUMMARY,
                        priority=ContextPriority.HIGH,
                    )
                    chunks.append(chunk)
                    budget.allocate("graphify_context", chunk.token_estimate)
            
            # 2. App map (MEDIUM priority - for architecture tasks)
            if graphify_output.has_app_map and task_type in (
                "architecture", "refactor", "debug"
            ):
                app_map_text = self._format_app_map(
                    graphify_output.app_map,
                    task_type,
                )
                
                available = budget.available
                if available > 500:  # Minimum 500 tokens for app map
                    app_map_text = self._truncate_to_tokens(
                        app_map_text,
                        min(available, 3000),
                    )
                    
                    chunk = ContextChunk(
                        content=app_map_text,
                        source=ContextSource.GRAPHIFY_APP_MAP,
                        priority=ContextPriority.MEDIUM,
                    )
                    chunks.append(chunk)
                    budget.allocate("graphify_context", chunk.token_estimate)
            
            # 3. Graph data (MEDIUM priority - specific queries)
            if graphify_output.has_graph and task_type in (
                "debug", "architecture", "refactor"
            ):
                graph_text = self._format_graph_context(
                    graphify_output.graph,
                    task_type,
                )
                
                available = budget.available
                if available > 1000:
                    graph_text = self._truncate_to_tokens(
                        graph_text,
                        min(available, 5000),
                    )
                    
                    chunk = ContextChunk(
                        content=graph_text,
                        source=ContextSource.GRAPHIFY_GRAPH,
                        priority=ContextPriority.MEDIUM,
                    )
                    chunks.append(chunk)
                    budget.allocate("graphify_context", chunk.token_estimate)
            
        except Exception as e:
            logger.warning(
                "Failed to load Graphify context",
                error=str(e),
                repo_path=str(repo_path),
            )
        
        return chunks
    
    async def _load_file_context(
        self,
        repo_path: Path,
        file_paths: List[str],
        budget: ContextBudget,
    ) -> List[ContextChunk]:
        """Load specific files into context."""
        chunks: List[ContextChunk] = []
        
        # Calculate budget per file
        available_per_file = budget.available // max(len(file_paths), 1)
        
        for file_path in file_paths[:10]:  # Max 10 files
            full_path = repo_path / file_path
            
            if not full_path.exists():
                logger.debug(f"File not found: {full_path}")
                continue
            
            try:
                content = await self._read_file_safe(full_path)
                
                # Truncate to budget
                if len(content) > available_per_file * 4:  # Rough char estimate
                    content = content[:available_per_file * 4]
                    content += f"\n\n... [File truncated. Full size: {full_path.stat().st_size} bytes]"
                
                chunk = ContextChunk(
                    content=f"FILE: {file_path}\n```\n{content}\n```\n",
                    source=ContextSource.CODE_FILE,
                    priority=ContextPriority.HIGH,
                    file_path=file_path,
                )
                
                chunks.append(chunk)
                budget.allocate("code_files", chunk.token_estimate)
                
            except Exception as e:
                logger.warning(
                    "Failed to read file",
                    file_path=str(full_path),
                    error=str(e),
                )
        
        return chunks
    
    def _load_conversation_history(
        self,
        history: List[Dict[str, str]],
        budget: ContextBudget,
    ) -> List[ContextChunk]:
        """Format conversation history."""
        if not history:
            return []
        
        # Keep last N messages
        max_messages = 10
        recent_history = history[-max_messages:]
        
        formatted = []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 2000:
                content = content[:2000] + "..."
            formatted.append(f"{role.upper()}: {content}")
        
        history_text = "CONVERSATION HISTORY:\n" + "\n".join(formatted)
        
        # Truncate to budget
        max_history_tokens = min(budget.available, 2000)
        history_text = self._truncate_to_tokens(history_text, max_history_tokens)
        
        chunk = ContextChunk(
            content=history_text,
            source=ContextSource.CONVERSATION_HISTORY,
            priority=ContextPriority.MEDIUM,
        )
        
        budget.allocate("conversation_history", chunk.token_estimate)
        return [chunk]
    
    async def _load_agent_memory(
        self,
        repo_path: Path,
        task_type: str,
        budget: ContextBudget,
    ) -> List[ContextChunk]:
        """Load agent memory/learnings from previous interactions."""
        # Placeholder for agent memory system
        # Future: Load from vector store, SQLite, or Redis
        return []
    
    # ========================================
    # Formatting Helpers
    # ========================================
    
    def _format_summary(self, summary: Any, mode: ContextMode) -> str:
        """Format Graphify summary for context inclusion."""
        parts = ["REPOSITORY ANALYSIS (Graphify):", ""]
        
        if hasattr(summary, "project_name"):
            parts.append(f"Project: {summary.project_name}")
        
        if hasattr(summary, "primary_language"):
            parts.append(f"Language: {summary.primary_language}")
        
        if hasattr(summary, "total_files"):
            parts.append(f"Files: {summary.total_files}")
        
        if hasattr(summary, "architecture_pattern"):
            parts.append(f"Architecture: {summary.architecture_pattern}")
        
        if hasattr(summary, "key_components"):
            parts.append("\nKey Components:")
            for comp in summary.key_components[:10]:
                parts.append(f"  • {comp}")
        
        if hasattr(summary, "dependencies"):
            parts.append("\nKey Dependencies:")
            for dep in summary.dependencies[:10]:
                parts.append(f"  • {dep}")
        
        if hasattr(summary, "recommendations") and mode == ContextMode.COMPREHENSIVE:
            parts.append("\nRecommendations:")
            for rec in summary.recommendations[:5]:
                parts.append(f"  • {rec}")
        
        return "\n".join(parts)
    
    def _format_app_map(self, app_map: Any, task_type: str) -> str:
        """Format application map for context."""
        parts = ["APPLICATION MODULES:", ""]
        
        if hasattr(app_map, "modules"):
            for module_name, module_data in list(app_map.modules.items())[:15]:
                parts.append(f"  • {module_name}")
                if isinstance(module_data, dict):
                    if "description" in module_data:
                        parts.append(f"    - {module_data['description'][:100]}")
        
        if hasattr(app_map, "entry_points"):
            parts.append(f"\nEntry Points: {', '.join(app_map.entry_points[:5])}")
        
        return "\n".join(parts)
    
    def _format_graph_context(self, graph: Any, task_type: str) -> str:
        """Format graph data for context."""
        parts = ["DEPENDENCY GRAPH INSIGHTS:", ""]
        
        if hasattr(graph, "node_count") and hasattr(graph, "edge_count"):
            parts.append(f"Total nodes: {graph.node_count}")
            parts.append(f"Total edges: {graph.edge_count}")
        
        # Identify high-degree nodes (god nodes in Graphify terms)
        if hasattr(graph, "nodes") and hasattr(graph, "edges"):
            node_degrees = {}
            for edge in graph.edges:
                node_degrees[edge.source] = node_degrees.get(edge.source, 0) + 1
                node_degrees[edge.target] = node_degrees.get(edge.target, 0) + 1
            
            # Top 5 most connected nodes
            top_nodes = sorted(
                node_degrees.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            
            if top_nodes:
                parts.append("\nMost Connected Components (God Nodes):")
                for node_id, degree in top_nodes:
                    node = graph.get_node_by_id(node_id) if hasattr(graph, "get_node_by_id") else None
                    node_name = node.name if node else node_id
                    parts.append(f"  • {node_name} ({degree} connections)")
        
        return "\n".join(parts)
    
    def _build_system_prompt(
        self,
        task_type: str,
        mode: ContextMode,
        skills_injected: bool = False,
    ) -> str:
        """Build system prompt based on task, mode, and skill availability."""
        base_prompt = [
            "You are an expert AI software engineering assistant.",
        ]
        
        if skills_injected:
            base_prompt.append(
                "You have been provided with specialized expert instructions "
                "(see 'EXPERT INSTRUCTIONS' sections above). Follow them carefully."
            )
        
        base_prompt.extend([
            "You have access to repository analysis via Graphify knowledge graphs.",
            "Provide accurate, actionable responses based on the context provided.",
        ])
        
        # Task-specific instructions
        task_instructions = {
            "code": [
                "Generate clean, well-documented code.",
                "Follow existing patterns and conventions in the codebase.",
                "Include error handling and edge cases.",
            ],
            "architecture": [
                "Analyze the system architecture and component relationships.",
                "Identify design patterns and architectural decisions.",
                "Suggest improvements with clear reasoning.",
            ],
            "debug": [
                "Analyze the issue systematically.",
                "Identify root causes, not just symptoms.",
                "Provide step-by-step debugging guidance.",
            ],
            "refactor": [
                "Suggest refactoring with minimal risk.",
                "Preserve existing functionality.",
                "Improve code clarity and maintainability.",
            ],
            "test": [
                "Generate comprehensive test cases.",
                "Cover edge cases and error conditions.",
                "Use existing test patterns from the codebase.",
            ],
            "documentation": [
                "Write clear, concise documentation.",
                "Explain complex concepts simply.",
                "Include usage examples where helpful.",
            ],
        }
        
        instructions = task_instructions.get(task_type, task_instructions["code"])
        base_prompt.extend(instructions)
        
        # Mode-specific additions
        if mode == ContextMode.PRECISE:
            base_prompt.append("Be concise and focused. Prioritize accuracy over breadth.")
        elif mode == ContextMode.COMPREHENSIVE:
            base_prompt.append("Provide thorough analysis. Consider edge cases and alternatives.")
        
        return "\n".join(base_prompt)
    
    def _wrap_user_prompt(
        self,
        user_prompt: str,
        context: str,
        skills_injected: bool = False,
    ) -> str:
        """Wrap user prompt with assembled context."""
        parts = []
        
        if context.strip():
            parts.append("=== REPOSITORY CONTEXT ===")
            parts.append(context.strip())
            parts.append("")
        
        parts.append("=== USER REQUEST ===")
        parts.append(user_prompt)
        
        parts.append("")
        
        if skills_injected:
            parts.append(
                "Please apply the expert instructions provided above when responding."
            )
        
        parts.append("Please respond based on the context provided above.")
        parts.append("Reference specific files and modules when relevant.")
        
        return "\n".join(parts)
    
    def _merge_chunks(self, chunks: List[ContextChunk]) -> str:
        """Merge context chunks into a single string, separated by source."""
        # Group by source
        grouped: Dict[ContextSource, List[ContextChunk]] = {}
        for chunk in chunks:
            if chunk.source not in grouped:
                grouped[chunk.source] = []
            grouped[chunk.source].append(chunk)
        
        # Order: Skills first, then Graphify, then files, then history
        source_order = [
            ContextSource.SKILLFILE_SKILL,
            ContextSource.GRAPHIFY_SUMMARY,
            ContextSource.GRAPHIFY_APP_MAP,
            ContextSource.GRAPHIFY_GRAPH,
            ContextSource.CODE_FILE,
            ContextSource.DOCUMENTATION,
            ContextSource.CONVERSATION_HISTORY,
            ContextSource.AGENT_MEMORY,
        ]
        
        sections = []
        for source in source_order:
            if source == ContextSource.USER_PROMPT:
                continue  # User prompt handled separately
            
            if source in grouped:
                section_parts = []
                for chunk in grouped[source]:
                    section_parts.append(chunk.content)
                
                if section_parts:
                    sections.append("\n".join(section_parts))
        
        return "\n\n---\n\n".join(sections)
    
    # ========================================
    # Budget Management
    # ========================================
    
    def _get_default_token_limit(self, mode: ContextMode) -> int:
        """Get default token limit based on mode."""
        limits = {
            ContextMode.PRECISE: 4096,
            ContextMode.BALANCED: 8192,
            ContextMode.COMPREHENSIVE: 16384,
            ContextMode.CODE_ONLY: 8192,
            ContextMode.DIAGRAM: 8192,
        }
        return limits.get(mode, 8192)
    
    def _create_budget(
        self,
        total_limit: int,
        task_type: str,
        mode: ContextMode,
        include_skills: bool = False,
    ) -> ContextBudget:
        """Create initial budget allocation."""
        budget = ContextBudget(total_limit=total_limit)
        
        # Default allocations by mode
        if mode == ContextMode.PRECISE:
            budget.graphify_context = total_limit // 5
            budget.skill_context = total_limit // 5 if include_skills else 0
            budget.code_files = total_limit // 4
        elif mode == ContextMode.BALANCED:
            budget.graphify_context = total_limit // 4
            budget.skill_context = total_limit // 5 if include_skills else 0
            budget.code_files = total_limit // 4
        elif mode == ContextMode.COMPREHENSIVE:
            budget.graphify_context = total_limit // 4
            budget.skill_context = total_limit // 4 if include_skills else 0
            budget.code_files = total_limit // 3
            budget.documentation = total_limit // 6
        
        return budget
    
    def _get_graphify_budget(
        self,
        budget: ContextBudget,
        mode: ContextMode,
    ) -> int:
        """Get token budget for Graphify context."""
        if mode == ContextMode.CODE_ONLY:
            return 0
        return max(0, budget.graphify_context - budget.allocated)
    
    def _emergency_truncate(
        self,
        chunks: List[ContextChunk],
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Emergency truncation when context exceeds budget."""
        # Sort by priority (keep critical first)
        sorted_chunks = sorted(chunks, key=lambda c: self._priority_sort_order(c.priority))
        
        kept_chunks = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            if chunk.priority == ContextPriority.CRITICAL:
                kept_chunks.append(chunk)
                current_tokens += chunk.token_estimate
            elif current_tokens + chunk.token_estimate <= max_tokens:
                kept_chunks.append(chunk)
                current_tokens += chunk.token_estimate
            elif current_tokens < max_tokens:
                # Truncate this chunk to fit
                available = max_tokens - current_tokens
                truncated_content = self._truncate_to_tokens(chunk.content, available)
                chunk.content = truncated_content
                chunk.token_estimate = available
                kept_chunks.append(chunk)
                break
        
        logger.warning(
            f"Emergency truncation: {len(chunks)} → {len(kept_chunks)} chunks"
        )
        
        return kept_chunks
    
    # ========================================
    # Utility Helpers
    # ========================================
    
    @staticmethod
    def _priority_sort_order(priority: ContextPriority) -> int:
        """Get sort order for priority."""
        order = {
            ContextPriority.CRITICAL: 0,
            ContextPriority.HIGH: 1,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 3,
            ContextPriority.OPTIONAL: 4,
        }
        return order.get(priority, 5)
    
    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token count."""
        if max_tokens <= 0:
            return ""
        
        # Rough: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        
        cut_point = max(last_period, last_newline, max_chars - 100)
        
        return text[:cut_point] + f"\n\n... [Truncated for token budget. Original: {len(text)} chars]"
    
    @staticmethod
    async def _read_file_safe(file_path: Path, max_size: int = 100_000) -> str:
        """Safely read a file with size limits."""
        import aiofiles
        
        file_size = file_path.stat().st_size
        if file_size > max_size:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read(max_size)
            return content + f"\n\n... [File truncated. Total size: {file_size} bytes]"
        
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return await f.read()
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        combined = "|".join(str(arg) for arg in args)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _update_stats(self, assembled: AssembledContext) -> None:
        """Update assembly statistics."""
        self._assembly_count += 1
        self._total_tokens_used += assembled.total_tokens
        self._average_tokens = self._total_tokens_used / self._assembly_count
        if assembled.skills_injected > 0:
            self._skills_injected_total += assembled.skills_injected
    
    # ========================================
    # Skill Management Helpers (🆕)
    # ========================================
    
    async def discover_and_add_skills(
        self,
        task_type: str,
        repo_context: Optional[Dict[str, Any]] = None,
        auto_add: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Discover relevant skills from community and optionally add them.
        
        Args:
            task_type: Type of task
            repo_context: Repository context for better matching
            auto_add: Automatically add discovered skills
            
        Returns:
            List of discovered skills
        """
        if not self.skillfile.is_installed:
            return []
        
        try:
            results = await self.skillfile.discover_skills_for_task(
                task_type=task_type,
                repo_context=repo_context,
            )
            
            discovered = []
            for result in results[:5]:
                skill_info = {
                    "name": result.name,
                    "description": result.description,
                    "source": result.source,
                    "stars": result.stars,
                    "security_score": result.security_score,
                    "tags": result.tags,
                }
                discovered.append(skill_info)
                
                # Auto-add if requested
                if auto_add and result.security_score >= 70:
                    await self.skillfile.add_skill(
                        name=result.name,
                        source=SkillSource.GITHUB,
                        path=result.url,
                    )
            
            return discovered
            
        except Exception as e:
            logger.warning("Failed to discover skills", error=str(e))
            return []
    
    async def customize_skill_for_project(
        self,
        skill_name: str,
        custom_additions: str,
    ) -> bool:
        """
        Customize a skill with project-specific additions.
        
        Args:
            skill_name: Name of skill to customize
            custom_additions: Custom content to add
            
        Returns:
            True if successful
        """
        if not self.skillfile.is_installed:
            return False
        
        try:
            # Load existing skill
            skills = await self.skillfile.load_skills("custom")
            
            existing_content = ""
            for skill in skills:
                if skill.entry.name == skill_name:
                    existing_content = skill.content
                    break
            
            if not existing_content:
                # Try to load the skill directly
                skill_path = self.repo_path / ".claude" / "skills" / skill_name / "SKILL.md"
                if skill_path.exists():
                    import aiofiles
                    async with aiofiles.open(skill_path, "r") as f:
                        existing_content = await f.read()
            
            # Append custom additions
            new_content = existing_content + "\n\n## Project-Specific Additions\n\n" + custom_additions
            
            # Save and pin
            return await self.skillfile.customize_skill(
                name=skill_name,
                custom_content=new_content,
                pin=True,
            )
            
        except Exception as e:
            logger.warning("Failed to customize skill", skill_name=skill_name, error=str(e))
            return False
    
    # ========================================
    # Public Utilities
    # ========================================
    
    async def estimate_tokens(self, repo_path: str, prompt: str) -> Dict[str, int]:
        """Estimate token usage without full assembly."""
        budget = self._create_budget(8192, "code", ContextMode.BALANCED, True)
        
        # Quick estimate
        prompt_tokens = len(prompt) // 4
        
        # Try to get Graphify size
        graphify_tokens = 0
        try:
            output = await self.parser.parse_repository(repo_path)
            if output.has_summary:
                summary_str = str(output.summary.model_dump() if hasattr(output.summary, "model_dump") else output.summary)
                graphify_tokens = len(summary_str) // 4
        except Exception:
            pass
        
        # Estimate skill tokens
        skill_tokens = 0
        if self.skillfile.is_installed and self._skill_injection_enabled:
            skill_tokens = 1500  # Average skill size
        
        return {
            "prompt_tokens": prompt_tokens,
            "graphify_tokens": graphify_tokens,
            "skill_tokens": skill_tokens,
            "system_tokens": 200,
            "estimated_total": prompt_tokens + graphify_tokens + skill_tokens + 200,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context assembler statistics."""
        return {
            "assemblies": self._assembly_count,
            "total_tokens_used": self._total_tokens_used,
            "average_tokens": round(self._average_tokens),
            "skills_injected_total": self._skills_injected_total,
            "skill_injection_enabled": self._skill_injection_enabled,
            "cache_size": len(self._context_cache),
            "cache_ttl_seconds": self._cache_ttl.total_seconds(),
        }
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._context_cache.clear()
        logger.info("Context cache cleared")
    
    async def close(self) -> None:
        """Clean shutdown."""
        self.clear_cache()
        await self.parser.close()
        await self.skillfile.close()
        logger.info("Context assembler shut down")


# ============================================================================
# Quick Helper Functions
# ============================================================================

async def build_context_for_agent(
    user_prompt: str,
    repo_path: str,
    task_type: str = "code",
    mode: str = "balanced",
    include_files: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    inject_skills: bool = True,
) -> AssembledContext:
    """
    Convenience function for building context for an agent.
    
    Args:
        user_prompt: User's request
        repo_path: Repository path
        task_type: Type of task
        mode: Context mode (precise, balanced, comprehensive)
        include_files: Files to include
        max_tokens: Token limit
        inject_skills: Whether to inject skillfile skills
        
    Returns:
        Assembled context
    """
    mode_enum = ContextMode(mode) if mode in [m.value for m in ContextMode] else ContextMode.BALANCED
    
    assembler = ContextAssembler()
    return await assembler.assemble_context(
        user_prompt=user_prompt,
        repo_path=repo_path,
        task_type=task_type,
        mode=mode_enum,
        include_files=include_files,
        max_tokens=max_tokens,
        inject_skills=inject_skills,
    )


# Default service instance
_default_assembler: Optional[ContextAssembler] = None


def get_context_service() -> ContextAssembler:
    """Get or create the default context assembler."""
    global _default_assembler
    if _default_assembler is None:
        _default_assembler = ContextAssembler()
    return _default_assembler


logger.info("Context service module initialized successfully")