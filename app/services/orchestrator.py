"""
Central orchestration service that coordinates all AI engineering workflows.

This is the main "conductor" that:
- Coordinates model selection, context assembly, and agent execution
- Implements complex multi-agent workflows (debug, refactor, test)
- Manages artifact storage and retrieval
- Handles streaming and non-streaming responses
- Provides comprehensive execution tracking
- Implements retry and fallback strategies
- Manages concurrent request processing
- Tracks execution metrics and performance

This is the primary entry point for all AI-assisted engineering tasks.
Production-grade with full observability, error handling, and recovery.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel, Field, field_validator

from app.core.config.settings import settings
from app.core.monitoring.logging import (
    get_logger,
    set_request_context,
    clear_request_context,
    LogContext,
)
from app.core.monitoring.metrics import (
    agent_executions,
    agent_execution_duration,
    agent_tokens_used,
    successful_code_generations,
    MetricsTracker,
)
from app.services.model_service import (
    ModelService,
    TaskCategory,
    get_model_service,
)
from app.services.context_service import (
    ContextAssembler,
    ContextMode,
    AssembledContext,
    get_context_service,
)
from app.integrations.ollama.client import (
    OllamaClient,
    get_default_client,
    OllamaClientError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from app.integrations.graphify.parser import GraphifyParser, get_default_parser
from app.artifacts.store import ArtifactStore, get_artifact_store
from app.utils.validators import PathValidator, InputSanitizer, validate_repo_path
from app.utils.retry import async_retry, CircuitBreakerOpenError
from pathlib import Path
from app.integrations.graphify.parser import GraphifyWrapper
from app.integrations.skillfile.client import SkillfileClient, get_skillfile_client
from app.agents.agent_factory import AgentFactory, get_agent_factory
from app.agents.base_agent import AgentTask
from app.workflows.debug_workflow import DebugWorkflow
from app.workflows.refactor_workflow import RefactorWorkflow
from app.workflows.testing_workflow import TestingWorkflow

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class WorkflowType(str, Enum):
    """Available workflow types."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    DEBUG_ANALYSIS = "debug_analysis"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    IMPACT_ANALYSIS = "impact_analysis"
    GENERAL_QA = "general_qa"


class ExecutionStatus(str, Enum):
    """Execution status for tracking."""
    PENDING = "pending"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    FALLBACK_USED = "fallback_used"


class ExecutionMode(str, Enum):
    """Execution mode for agent interactions."""
    SYNC = "sync"           # Wait for complete response
    STREAMING = "streaming"  # Stream response tokens
    BATCH = "batch"         # Queue for later execution


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""
    execution_id: str
    workflow_type: WorkflowType
    model_used: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    context_tokens: int = 0
    response_tokens: int = 0
    retry_count: int = 0
    fallback_used: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None
    
    def complete(self, tokens: int = 0, status: ExecutionStatus = ExecutionStatus.COMPLETED):
        """Mark execution as complete."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.response_tokens = tokens
        self.status = status
    
    def fail(self, error: str):
        """Mark execution as failed."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = ExecutionStatus.FAILED
        self.error = error


class OrchestratorRequest(BaseModel):
    """Validated request to the orchestrator."""
    repo_path: str = Field(..., description="Path to the repository")
    prompt: str = Field(..., min_length=1, max_length=10000)
    workflow_type: WorkflowType = Field(default=WorkflowType.GENERAL_QA)
    mode: ExecutionMode = Field(default=ExecutionMode.SYNC)
    context_mode: str = Field(default="balanced")
    preferred_model: Optional[str] = Field(default=None)
    include_files: Optional[List[str]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None, ge=256, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    conversation_id: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("repo_path")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        """Validate repository path."""
        path = PathValidator.validate_path(v, must_exist=True, must_be_dir=True)
        return str(path)


class OrchestratorResponse(BaseModel):
    """Standard orchestrator response."""
    execution_id: str
    status: ExecutionStatus
    workflow_type: WorkflowType
    model_used: str
    response: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    graphify_available: bool = False
    files_analyzed: List[str] = Field(default_factory=list)
    tokens_used: int = 0
    duration_ms: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Orchestrator
# ============================================================================

class Orchestrator:
    """
    Central orchestrator for AI engineering workflows.
    
    Coordinates all services to execute complex agent workflows.
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model_service: Optional[ModelService] = None,
        context_service: Optional[ContextAssembler] = None,
        artifact_store: Optional[ArtifactStore] = None,
        graphify_parser: Optional[GraphifyParser] = None,
    ):
        """
        Initialize orchestrator with all required services.
        
        Args:
            ollama_client: Ollama client instance
            model_service: Model selection service
            context_service: Context assembly service
            artifact_store: Artifact storage service
            graphify_parser: Graphify parser instance
        """
        # Core services
        self.ollama = ollama_client or get_default_client()
        self.model_service = model_service or get_model_service()
        self.context_service = context_service or get_context_service()
        self.artifact_store = artifact_store or get_artifact_store()
        self.graphify_parser = graphify_parser or get_default_parser()
        self.skillfile = get_skillfile_client()
        self._agent_factory = AgentFactory(ollama_client=self.ollama)
        self._debug_workflow = DebugWorkflow(factory=self._agent_factory)
        self._refactor_workflow = RefactorWorkflow(factory=self._agent_factory)
        self._testing_workflow = TestingWorkflow(factory=self._agent_factory)
        self._intelligence_auto_refresh = True  # Enable automatic updates

        # Execution tracking
        self._active_executions: Dict[str, ExecutionMetrics] = {}
        self._execution_history: List[ExecutionMetrics] = []
        self._max_history = 1000
        
        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(
            settings.max_concurrent_requests
        )
        
        # Workflow registry
        self._workflow_registry: Dict[WorkflowType, Callable] = {
            WorkflowType.CODE_GENERATION: self._execute_code_generation,
            WorkflowType.CODE_REVIEW: self._execute_code_review,
            WorkflowType.CODE_REFACTORING: self._execute_code_refactoring,
            WorkflowType.DEBUG_ANALYSIS: self._execute_debug_analysis,
            WorkflowType.ARCHITECTURE_ANALYSIS: self._execute_architecture_analysis,
            WorkflowType.TEST_GENERATION: self._execute_test_generation,
            WorkflowType.DOCUMENTATION: self._execute_documentation,
            WorkflowType.IMPACT_ANALYSIS: self._execute_impact_analysis,
            WorkflowType.GENERAL_QA: self._execute_general_qa,
        }
        
        # Multi-agent workflows
        self._multi_agent_workflows = {
            WorkflowType.CODE_REFACTORING: self._execute_refactoring_workflow,
            WorkflowType.DEBUG_ANALYSIS: self._execute_debug_workflow,
        }
        
        logger.info(
            "Orchestrator initialized",
            workflows=list(self._workflow_registry.keys()),
            max_concurrent=settings.max_concurrent_requests,
        )
    
    # ========================================
    # Main Execution Entry Point
    # ========================================
    
    async def execute(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResponse:
        """
        Execute an AI engineering workflow.
        
        Args:
            request: Orchestrator request with all parameters
            
        Returns:
            Orchestrator response with results
        """
        execution_id = str(uuid.uuid4())
        
        # Set request context for logging
        set_request_context(request_id=execution_id)
        
        # Create execution metrics
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            workflow_type=request.workflow_type,
            model_used=request.preferred_model or "auto",
            start_time=datetime.utcnow(),
        )
        self._active_executions[execution_id] = metrics
        
        warnings: List[str] = []
        
        try:
            # Acquire execution slot
            async with self._execution_semaphore:
                metrics.status = ExecutionStatus.RUNNING

                # Auto-refresh intelligence before execution
                if self._intelligence_auto_refresh:
                    freshness = await self.ensure_fresh_intelligence(
                        request.repo_path,
                        force=False,
                    )
                    if freshness["graphify_updated"]:
                        warnings.append("Graphify analysis was auto-updated")
                    if freshness["skills_updated"]:
                        warnings.append("AI skills were auto-updated")
                
                # Track metrics
                agent_executions.labels(
                    agent_type=request.workflow_type.value,
                    model_used=metrics.model_used,
                    status="started",
                ).inc()
                
                async with MetricsTracker(
                    agent_execution_duration,
                    {"agent_type": request.workflow_type.value},
                ):
                    # Step 1: Check repository
                    if not validate_repo_path(request.repo_path):
                        raise ValueError(f"Invalid repository path: {request.repo_path}")
                    
                    # Step 2: Assemble context
                    context = await self._assemble_context(request)
                    metrics.context_tokens = context.total_tokens
                    warnings.extend(context.warnings)
                    
                    # Step 3: Select model
                    if not request.preferred_model:
                        task_category = self._map_workflow_to_task(request.workflow_type)
                        model_name = await self.model_service.select_model(
                            task_type=task_category,
                        )
                    else:
                        model_name = request.preferred_model
                    
                    metrics.model_used = model_name
                    
                    # Step 4: Execute workflow
                    if request.mode == ExecutionMode.STREAMING:
                        # Streaming handled separately
                        response_text = await self._execute_workflow_sync(
                            request, context, model_name, metrics
                        )
                    else:
                        response_text = await self._execute_workflow_sync(
                            request, context, model_name, metrics
                        )
                    
                    # Step 5: Store artifacts
                    artifacts = await self._store_artifacts(
                        execution_id=execution_id,
                        request=request,
                        response=response_text,
                        context=context,
                        metrics=metrics,
                    )
                    
                    # Mark complete
                    tokens = self._estimate_tokens(response_text)
                    metrics.complete(tokens=tokens)
                    
                    # Track success
                    agent_executions.labels(
                        agent_type=request.workflow_type.value,
                        model_used=model_name,
                        status="success",
                    ).inc()
                    
                    agent_tokens_used.labels(
                        agent_type=request.workflow_type.value,
                        model_used=model_name,
                    ).inc(tokens)
                    
                    if request.workflow_type == WorkflowType.CODE_GENERATION:
                        successful_code_generations.inc()
                    
                    logger.info(
                        "Execution completed",
                        execution_id=execution_id,
                        workflow=request.workflow_type.value,
                        model=model_name,
                        tokens=tokens,
                        duration_ms=round(metrics.duration_ms, 2),
                    )
                    
                    return OrchestratorResponse(
                        execution_id=execution_id,
                        status=ExecutionStatus.COMPLETED,
                        workflow_type=request.workflow_type,
                        model_used=model_name,
                        response=response_text,
                        artifacts=artifacts,
                        graphify_available=context.graphify_available,
                        files_analyzed=context.files_included,
                        tokens_used=tokens,
                        duration_ms=metrics.duration_ms,
                        warnings=warnings,
                        metadata=request.metadata,
                    )
                    
        except CircuitBreakerOpenError as e:
            metrics.fail(str(e))
            metrics.fallback_used = True
            
            logger.warning(
                "Circuit breaker open, using fallback",
                execution_id=execution_id,
                error=str(e),
            )
            
            # Try with fallback model
            try:
                return await self._execute_with_fallback(request, execution_id, metrics)
            except Exception as fallback_error:
                return self._error_response(
                    execution_id, request.workflow_type, metrics, str(fallback_error), warnings
                )
                
        except (ModelNotFoundError, ModelTimeoutError) as e:
            metrics.fail(str(e))
            
            logger.error(
                "Model error",
                execution_id=execution_id,
                error=str(e),
            )
            
            return self._error_response(
                execution_id, request.workflow_type, metrics, str(e), warnings
            )
            
        except Exception as e:
            metrics.fail(str(e))
            
            logger.error(
                "Execution failed",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )
            
            return self._error_response(
                execution_id, request.workflow_type, metrics, str(e), warnings
            )
            
        finally:
            # Cleanup
            self._active_executions.pop(execution_id, None)
            self._execution_history.append(metrics)
            
            # Trim history
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]
            
            clear_request_context()
    
    async def execute_streaming(
        self,
        request: OrchestratorRequest,
    ) -> AsyncIterator[str]:
        """
        Execute workflow with streaming response.
        
        Args:
            request: Orchestrator request
            
        Yields:
            Response tokens as they are generated
        """
        execution_id = str(uuid.uuid4())
        set_request_context(request_id=execution_id)
        
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            workflow_type=request.workflow_type,
            model_used=request.preferred_model or "auto",
            start_time=datetime.utcnow(),
            status=ExecutionStatus.STREAMING,
        )
        
        try:
            async with self._execution_semaphore:
                # Assemble context
                context = await self._assemble_context(request)
                
                # Select model
                if not request.preferred_model:
                    task_category = self._map_workflow_to_task(request.workflow_type)
                    model_name = await self.model_service.select_model(task_category)
                else:
                    model_name = request.preferred_model
                
                metrics.model_used = model_name
                
                # Build messages
                messages = [
                    {"role": "system", "content": context.system_prompt},
                    {"role": "user", "content": context.user_prompt},
                ]
                
                # Stream from Ollama
                full_response = []
                
                async for token in self.ollama.chat(
                    model=model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens or 4096,
                    stream=True,
                ):
                    full_response.append(token)
                    yield token
                
                # Complete metrics
                response_text = "".join(full_response)
                tokens = self._estimate_tokens(response_text)
                metrics.complete(tokens=tokens, status=ExecutionStatus.COMPLETED)
                
                # Store artifacts
                await self._store_artifacts(
                    execution_id=execution_id,
                    request=request,
                    response=response_text,
                    context=context,
                    metrics=metrics,
                )
                
        except Exception as e:
            metrics.fail(str(e))
            logger.error("Streaming execution failed", error=str(e))
            yield f"\n\n[Error: {str(e)}]"
            
        finally:
            self._active_executions.pop(execution_id, None)
            self._execution_history.append(metrics)
            clear_request_context()
    
    # ========================================
    # Workflow Executors
    # ========================================
    
    async def _execute_workflow_sync(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute a workflow synchronously."""
        
        # Check if multi-agent workflow
        if request.workflow_type in self._multi_agent_workflows:
            workflow_func = self._multi_agent_workflows[request.workflow_type]
            return await workflow_func(request, context, model_name, metrics)
        
        # Single-agent workflow
        workflow_func = self._workflow_registry.get(
            request.workflow_type,
            self._execute_general_qa,
        )
        
        return await workflow_func(request, context, model_name, metrics)
    
    async def _execute_code_generation(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute code generation workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Generate production-ready code with error handling and documentation."
        )
    
    async def _execute_code_review(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute code review workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Review the code thoroughly. Identify bugs, security issues, and improvement opportunities."
        )
    
    async def _execute_code_refactoring(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute code refactoring workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Suggest refactoring with minimal risk. Preserve functionality while improving code quality."
        )
    
    async def _execute_debug_analysis(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute debug analysis workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Analyze the issue systematically. Identify root causes and provide step-by-step debugging guidance."
        )
    
    async def _execute_architecture_analysis(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute architecture analysis workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Analyze the system architecture. Identify patterns, risks, and improvement opportunities."
        )
    
    async def _execute_test_generation(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute test generation workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Generate comprehensive tests covering edge cases and error conditions."
        )
    
    async def _execute_documentation(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute documentation workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Write clear, comprehensive documentation with usage examples."
        )
    
    async def _execute_impact_analysis(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute impact analysis workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
            task_instruction="Analyze the impact of changes. Identify affected modules and risk levels."
        )
    
    async def _execute_general_qa(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Execute general Q&A workflow."""
        return await self._execute_single_agent(
            request, context, model_name, metrics,
        )
    
    async def _execute_single_agent(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        model_name: str,
        metrics: ExecutionMetrics,
        task_instruction: Optional[str] = None,
    ) -> str:
        """Execute a single agent call with the assembled context."""
        
        # Build messages
        system_prompt = context.system_prompt
        if task_instruction:
            system_prompt += f"\n\nTASK: {task_instruction}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context.user_prompt},
        ]
        
        # Execute
        response = await self.ollama.chat(
            model=model_name,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
            stream=False,
        )
        
        return response
    
    # ========================================
    # Multi-Agent Workflows
    # ========================================
    
    async def _execute_refactoring_workflow(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        primary_model: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Multi-agent refactoring workflow: Analyze → Refactor → Review."""
        arch_model = await self.model_service.select_model(TaskCategory.ARCHITECTURE_ANALYSIS)
        code_model = await self.model_service.select_model(TaskCategory.CODE_REFACTORING)
        result = await self._refactor_workflow.run(
            repo_path=request.repo_path,
            prompt=request.prompt,
            context=context,
            arch_model=arch_model,
            code_model=code_model,
        )
        return result.to_markdown()
    
    async def _execute_debug_workflow(
        self,
        request: OrchestratorRequest,
        context: AssembledContext,
        primary_model: str,
        metrics: ExecutionMetrics,
    ) -> str:
        """Multi-agent debug workflow: Architecture context → Root cause → Solution."""
        debug_model = await self.model_service.select_model(TaskCategory.DEBUGGING)
        code_model = await self.model_service.select_model(TaskCategory.CODE_GENERATION)
        result = await self._debug_workflow.run(
            prompt=request.prompt,
            repo_path=request.repo_path,
            context=context,
            debug_model=debug_model,
            code_model=code_model,
        )
        return result.to_markdown()
    
    # ========================================
    # Fallback & Error Recovery
    # ========================================
    
    async def _execute_with_fallback(
        self,
        request: OrchestratorRequest,
        execution_id: str,
        metrics: ExecutionMetrics,
    ) -> OrchestratorResponse:
        """Execute with fallback model."""
        fallback_model = settings.ollama.fallback_model
        
        logger.info(
            "Attempting fallback execution",
            execution_id=execution_id,
            fallback_model=fallback_model,
        )
        
        context = await self._assemble_context(request)
        
        response = await self._execute_single_agent(
            request, context, fallback_model, metrics,
        )
        
        tokens = self._estimate_tokens(response)
        metrics.complete(tokens=tokens, status=ExecutionStatus.FALLBACK_USED)
        metrics.fallback_used = True
        
        return OrchestratorResponse(
            execution_id=execution_id,
            status=ExecutionStatus.FALLBACK_USED,
            workflow_type=request.workflow_type,
            model_used=fallback_model,
            response=response,
            graphify_available=context.graphify_available,
            files_analyzed=context.files_included,
            tokens_used=tokens,
            duration_ms=metrics.duration_ms,
            warnings=["Fallback model used due to primary model unavailability"],
        )
    
    def _error_response(
        self,
        execution_id: str,
        workflow_type: WorkflowType,
        metrics: ExecutionMetrics,
        error: str,
        warnings: List[str],
    ) -> OrchestratorResponse:
        """Create error response."""
        return OrchestratorResponse(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            workflow_type=workflow_type,
            model_used=metrics.model_used,
            error=error,
            duration_ms=metrics.duration_ms,
            warnings=warnings,
        )
    
    # ========================================
    # Helpers
    # ========================================
    
    async def _assemble_context(
        self,
        request: OrchestratorRequest,
    ) -> AssembledContext:
        """Assemble context for execution."""
        try:
            mode = ContextMode(request.context_mode) if request.context_mode in [
                m.value for m in ContextMode
            ] else ContextMode.BALANCED
        except ValueError:
            mode = ContextMode.BALANCED
        
        return await self.context_service.assemble_context(
            user_prompt=request.prompt,
            repo_path=request.repo_path,
            task_type=request.workflow_type.value,
            mode=mode,
            max_tokens=request.max_tokens,
            include_files=request.include_files,
        )
    
    def _map_workflow_to_task(self, workflow: WorkflowType) -> TaskCategory:
        """Map workflow type to task category for model selection."""
        mapping = {
            WorkflowType.CODE_GENERATION: TaskCategory.CODE_GENERATION,
            WorkflowType.CODE_REVIEW: TaskCategory.CODE_REVIEW,
            WorkflowType.CODE_REFACTORING: TaskCategory.CODE_REFACTORING,
            WorkflowType.DEBUG_ANALYSIS: TaskCategory.DEBUGGING,
            WorkflowType.ARCHITECTURE_ANALYSIS: TaskCategory.ARCHITECTURE_ANALYSIS,
            WorkflowType.TEST_GENERATION: TaskCategory.TEST_GENERATION,
            WorkflowType.DOCUMENTATION: TaskCategory.DOCUMENTATION,
            WorkflowType.IMPACT_ANALYSIS: TaskCategory.ARCHITECTURE_ANALYSIS,
            WorkflowType.GENERAL_QA: TaskCategory.GENERAL_QA,
        }
        return mapping.get(workflow, TaskCategory.GENERAL_QA)
    
    async def _store_artifacts(
        self,
        execution_id: str,
        request: OrchestratorRequest,
        response: str,
        context: AssembledContext,
        metrics: ExecutionMetrics,
    ) -> List[Dict[str, Any]]:
        """Store execution artifacts."""
        artifacts: List[Dict[str, Any]] = []

        try:
            run_id = await self.artifact_store.save_artifact(
                execution_id=execution_id,
                artifact_type="response",
                content=response,
                metadata={
                    "workflow_type": request.workflow_type.value,
                    "model_used": metrics.model_used,
                    "tokens_used": metrics.response_tokens,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            artifacts.append({
                "id": run_id,
                "type": "response",
                "execution_id": execution_id,
                "workflow_type": request.workflow_type.value,
                "model_used": metrics.model_used,
            })
            artifacts.append({
                "id": run_id,
                "type": "prompt",
                "execution_id": execution_id,
                "workflow_type": request.workflow_type.value,
            })
        except Exception as e:
            logger.warning("Failed to store artifacts", error=str(e))

        return artifacts
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Quick token estimation."""
        return len(text) // 4
    
    # ========================================
    # Management
    # ========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive orchestrator health check."""
        return {
            "status": "healthy",
            "active_executions": len(self._active_executions),
            "total_executions": len(self._execution_history),
            "ollama_health": await self.ollama.health_check(),
            "model_service": await self.model_service.health_check(),
            "context_stats": self.context_service.get_stats(),
            "semaphore_available": self._execution_semaphore._value,
        }
    
    async def ensure_fresh_intelligence(
        self,
        repo_path: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Ensure Graphify and skills are up-to-date before execution.
        Called automatically before every agent interaction.
        """
        status = {
            "graphify_updated": False,
            "skills_updated": False,
            "graphify_version": None,
            "skills_version": None,
        }
        
        # 1. Auto-update Graphify
        try:
            graphify_status = await self._auto_update_graphify(repo_path, force)
            status["graphify_updated"] = graphify_status["updated"]
            status["graphify_version"] = graphify_status["version"]
        except Exception as e:
            logger.warning("Graphify auto-update failed", error=str(e))
        
        # 2. Auto-update skills
        try:
            if self.skillfile and self.skillfile.is_installed:
                skills_status = await self._auto_update_skills(force)
                status["skills_updated"] = skills_status["updated"]
                status["skills_version"] = skills_status["version"]
        except Exception as e:
            logger.warning("Skills auto-update failed", error=str(e))
        
        return status
    
    async def _auto_update_graphify(
        self,
        repo_path: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Automatically update Graphify analysis if stale."""
        repo_path_obj = Path(repo_path)
        graphify_dir = repo_path_obj / "graphify-out"
        
        needs_update = False
        
        if not graphify_dir.exists():
            needs_update = True
            logger.info("No Graphify output found, will generate")
        else:
            graph_file = graphify_dir / "graph.json"
            if graph_file.exists():
                age_hours = (datetime.utcnow() - datetime.fromtimestamp(
                    graph_file.stat().st_mtime
                )).total_seconds() / 3600
                
                if age_hours > 24 or force:
                    needs_update = True
                    logger.info(f"Graphify output is {age_hours:.1f}h old, refreshing")
            
            if not needs_update:
                code_changed = await self._has_code_changed_since(repo_path_obj, graphify_dir)
                if code_changed:
                    needs_update = True
                    logger.info("Code has changed since last Graphify analysis")
        
        if needs_update:
            wrapper = GraphifyWrapper(str(repo_path_obj))
            await wrapper.run_graphify(force=True)
            self.context_service.clear_cache()
            
            return {"updated": True, "version": datetime.utcnow().isoformat()}
        
        return {"updated": False, "version": None}
    
    async def _auto_update_skills(self, force: bool = False) -> Dict[str, Any]:
        """Automatically update skillfile skills if updates available."""
        lock_file = Path(str(self.repo_path)) / "Skillfile.lock" if hasattr(self, 'repo_path') else None
        
        needs_update = force
        
        if not force and lock_file and lock_file.exists():
            age_hours = (datetime.utcnow() - datetime.fromtimestamp(
                lock_file.stat().st_mtime
            )).total_seconds() / 3600
            
            if age_hours > 24:
                needs_update = True
        
        if needs_update:
            status = await self.skillfile.get_status()
            
            if status.get("outdated", 0) > 0:
                await self.skillfile.install_skills(update=True)
                self.context_service.clear_cache()
                
                return {
                    "updated": True,
                    "version": datetime.utcnow().isoformat(),
                    "skills_updated": status["outdated"],
                }
        
        return {"updated": False, "version": None}
    
    async def _has_code_changed_since(
        self,
        repo_path: Path,
        graphify_dir: Path,
    ) -> bool:
        """Check if significant code changes happened since last Graphify run."""
        try:
            graph_file = graphify_dir / "graph.json"
            if not graph_file.exists():
                return True
            
            last_run = datetime.fromtimestamp(graph_file.stat().st_mtime)
            last_run_str = last_run.strftime("%Y-%m-%d %H:%M:%S")
            
            process = await asyncio.create_subprocess_exec(
                "git", "diff", "--name-only", f"--since={last_run_str}",
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                changed_files = stdout.decode().strip().split("\n")
                code_files = [
                    f for f in changed_files
                    if f and any(f.endswith(ext) for ext in [
                        ".py", ".js", ".ts", ".go", ".java", ".rs",
                        ".cpp", ".c", ".h", ".dart", ".swift", ".kt",
                    ])
                ]
                
                if len(code_files) > 0:
                    return True
            
            return False
            
        except Exception as e:
            logger.debug("Failed to check code changes", error=str(e))
            return False
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get currently active executions."""
        return [
            {
                "execution_id": eid,
                "workflow": m.workflow_type.value,
                "model": m.model_used,
                "status": m.status.value,
                "started": m.start_time.isoformat(),
                "duration_ms": (
                    (datetime.utcnow() - m.start_time).total_seconds() * 1000
                    if m.status in (ExecutionStatus.RUNNING, ExecutionStatus.STREAMING)
                    else m.duration_ms
                ),
            }
            for eid, m in self._active_executions.items()
        ]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = len(self._execution_history)
        if total == 0:
            return {"total": 0}
        
        completed = sum(
            1 for m in self._execution_history
            if m.status == ExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for m in self._execution_history
            if m.status == ExecutionStatus.FAILED
        )
        fallback = sum(
            1 for m in self._execution_history
            if m.fallback_used
        )
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "fallback_used": fallback,
            "success_rate": round((completed / total) * 100, 2),
            "average_duration_ms": round(
                sum(
                    m.duration_ms for m in self._execution_history
                    if m.duration_ms > 0
                ) / max(completed, 1),
                2,
            ),
            "models_used": list(set(
                m.model_used for m in self._execution_history
            )),
        }
    
    async def close(self):
        """Clean shutdown."""
        logger.info("Shutting down orchestrator")
        await self.ollama.close()
        await self.context_service.close()
        await self.artifact_store.close()


# ============================================================================
# Factory
# ============================================================================

_default_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the default orchestrator."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = Orchestrator()
    return _default_orchestrator


logger.info("Orchestrator module initialized successfully")