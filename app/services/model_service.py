"""
Model routing service with intelligent model selection, load balancing,
and health-aware routing strategies.

This service:
- Selects optimal models based on task requirements
- Implements model chaining for complex tasks
- Monitors model health and availability
- Manages model fallback chains
- Provides cost/performance optimization
- Tracks model usage statistics

Production-grade with circuit breaker awareness and retry budget management.
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import structlog

from app.core.config.settings import settings
from app.core.monitoring.metrics import model_requests, model_errors
from app.integrations.ollama.client import OllamaClient, get_default_client
from app.utils.validators import validate_model

logger = structlog.get_logger(__name__)


class TaskCategory(str, Enum):
    """Task categories for model selection."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    TEST_GENERATION = "test_generation"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    CODE_EXPLANATION = "code_explanation"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    GENERAL_QA = "general_qa"


class ModelTier(str, Enum):
    """Model capability tiers."""
    FAST = "fast"           # Small, fast models
    BALANCED = "balanced"   # Medium, general purpose
    POWERFUL = "powerful"   # Large, capable models
    SPECIALIZED = "specialized"  # Task-specific fine-tuned


# Model registry with capabilities and preferences
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Code models
    "qwen2.5-coder:7b": {
        "tier": ModelTier.BALANCED,
        "specialties": [
            TaskCategory.CODE_GENERATION,
            TaskCategory.CODE_REVIEW,
            TaskCategory.CODE_REFACTORING,
            TaskCategory.TEST_GENERATION,
        ],
        "max_context": 32768,
        "cost_weight": 1.0,
        "quality_weight": 0.8,
    },
    "qwen2.5-coder:14b": {
        "tier": ModelTier.POWERFUL,
        "specialties": [
            TaskCategory.CODE_GENERATION,
            TaskCategory.CODE_REVIEW,
            TaskCategory.CODE_REFACTORING,
            TaskCategory.TEST_GENERATION,
            TaskCategory.ARCHITECTURE_ANALYSIS,
        ],
        "max_context": 32768,
        "cost_weight": 2.0,
        "quality_weight": 0.9,
    },
    "deepseek-coder:6.7b": {
        "tier": ModelTier.BALANCED,
        "specialties": [
            TaskCategory.CODE_GENERATION,
            TaskCategory.CODE_REVIEW,
            TaskCategory.TEST_GENERATION,
        ],
        "max_context": 16384,
        "cost_weight": 1.2,
        "quality_weight": 0.85,
    },
    
    # Reasoning models
    "deepseek-r1:7b": {
        "tier": ModelTier.POWERFUL,
        "specialties": [
            TaskCategory.ARCHITECTURE_ANALYSIS,
            TaskCategory.DEBUGGING,
            TaskCategory.SECURITY_AUDIT,
            TaskCategory.PERFORMANCE_OPTIMIZATION,
        ],
        "max_context": 32768,
        "cost_weight": 2.5,
        "quality_weight": 0.95,
    },
    "deepseek-r1:14b": {
        "tier": ModelTier.POWERFUL,
        "specialties": [
            TaskCategory.ARCHITECTURE_ANALYSIS,
            TaskCategory.DEBUGGING,
            TaskCategory.SECURITY_AUDIT,
            TaskCategory.PERFORMANCE_OPTIMIZATION,
        ],
        "max_context": 32768,
        "cost_weight": 3.0,
        "quality_weight": 0.98,
    },
    
    # General purpose
    "qwen2.5:7b": {
        "tier": ModelTier.BALANCED,
        "specialties": [
            TaskCategory.GENERAL_QA,
            TaskCategory.CODE_EXPLANATION,
            TaskCategory.DOCUMENTATION,
        ],
        "max_context": 32768,
        "cost_weight": 1.0,
        "quality_weight": 0.75,
    },
    "qwen2.5:14b": {
        "tier": ModelTier.POWERFUL,
        "specialties": [
            TaskCategory.GENERAL_QA,
            TaskCategory.CODE_EXPLANATION,
            TaskCategory.DOCUMENTATION,
            TaskCategory.CODE_REVIEW,
        ],
        "max_context": 32768,
        "cost_weight": 2.0,
        "quality_weight": 0.85,
    },
    
    # Writing/Documentation
    "gemma2:9b": {
        "tier": ModelTier.BALANCED,
        "specialties": [
            TaskCategory.DOCUMENTATION,
            TaskCategory.CODE_EXPLANATION,
            TaskCategory.GENERAL_QA,
        ],
        "max_context": 8192,
        "cost_weight": 1.0,
        "quality_weight": 0.8,
    },
    
    # Vision models
    "llava:7b": {
        "tier": ModelTier.SPECIALIZED,
        "specialties": [
            TaskCategory.CODE_EXPLANATION,  # For diagram understanding
        ],
        "max_context": 4096,
        "cost_weight": 1.5,
        "quality_weight": 0.7,
    },
    
    # Fast/fallback models
    "qwen2.5:3b": {
        "tier": ModelTier.FAST,
        "specialties": [
            TaskCategory.GENERAL_QA,
            TaskCategory.CODE_EXPLANATION,
        ],
        "max_context": 8192,
        "cost_weight": 0.5,
        "quality_weight": 0.6,
    },
    "phi3:3.8b": {
        "tier": ModelTier.FAST,
        "specialties": [
            TaskCategory.GENERAL_QA,
            TaskCategory.CODE_GENERATION,
        ],
        "max_context": 4096,
        "cost_weight": 0.5,
        "quality_weight": 0.65,
    },
}


# Task-to-model-type mapping with priority order
TASK_MODEL_MAP: Dict[TaskCategory, List[str]] = {
    TaskCategory.CODE_GENERATION: [
        "qwen2.5-coder:14b", "deepseek-coder:6.7b", "qwen2.5-coder:7b",
        "qwen2.5:14b", "qwen2.5:7b",
    ],
    TaskCategory.CODE_REVIEW: [
        "qwen2.5-coder:14b", "qwen2.5-coder:7b", "deepseek-r1:7b",
        "qwen2.5:14b",
    ],
    TaskCategory.CODE_REFACTORING: [
        "qwen2.5-coder:14b", "qwen2.5-coder:7b", "deepseek-r1:7b",
    ],
    TaskCategory.TEST_GENERATION: [
        "qwen2.5-coder:14b", "deepseek-coder:6.7b", "qwen2.5-coder:7b",
        "qwen2.5:14b",
    ],
    TaskCategory.ARCHITECTURE_ANALYSIS: [
        "deepseek-r1:14b", "deepseek-r1:7b", "qwen2.5:14b",
        "qwen2.5-coder:14b",
    ],
    TaskCategory.DEBUGGING: [
        "deepseek-r1:14b", "deepseek-r1:7b", "qwen2.5-coder:14b",
        "qwen2.5-coder:7b",
    ],
    TaskCategory.DOCUMENTATION: [
        "gemma2:9b", "qwen2.5:14b", "qwen2.5:7b",
    ],
    TaskCategory.CODE_EXPLANATION: [
        "qwen2.5:14b", "qwen2.5:7b", "gemma2:9b", "deepseek-r1:7b",
    ],
    TaskCategory.SECURITY_AUDIT: [
        "deepseek-r1:14b", "deepseek-r1:7b", "qwen2.5-coder:14b",
    ],
    TaskCategory.PERFORMANCE_OPTIMIZATION: [
        "deepseek-r1:14b", "deepseek-r1:7b", "qwen2.5-coder:14b",
    ],
    TaskCategory.GENERAL_QA: [
        "qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b", "phi3:3.8b",
    ],
}


class ModelService:
    """
    Intelligent model selection and routing service.
    
    Features:
    - Task-based model selection with priority lists
    - Health-aware routing (skips unhealthy models)
    - Model chaining for complex multi-step tasks
    - Fallback chains for reliability
    - Cost/quality optimization
    - Usage tracking and statistics
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Initialize model service.
        
        Args:
            ollama_client: OllamaClient instance (uses default if not provided)
        """
        self.ollama = ollama_client or get_default_client()
        
        # Model availability cache
        self._availability_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._availability_ttl = timedelta(seconds=60)
        
        # Selection statistics
        self._selection_counts: Dict[str, int] = {}
        self._fallback_counts: Dict[str, int] = {}
        
        logger.info("Model service initialized")
    
    async def select_model(
        self,
        task_type: TaskCategory,
        preferred_model: Optional[str] = None,
        require_available: bool = True,
        tier: Optional[ModelTier] = None,
    ) -> str:
        """
        Select the best model for a given task.
        
        Selection priority:
        1. User-specified preferred_model (if available)
        2. Task-appropriate model (from TASK_MODEL_MAP)
        3. Tier-filtered models
        4. Default model from settings
        5. Fast fallback model
        
        Args:
            task_type: Category of task
            preferred_model: Specific model requested by user
            require_available: Only return available models
            tier: Limit to specific capability tier
            
        Returns:
            Selected model name
            
        Raises:
            ValueError: If no suitable model found
        """
        # If preferred model specified, check availability
        if preferred_model:
            if validate_model(preferred_model):
                if not require_available or await self._is_available(preferred_model):
                    self._record_selection(preferred_model)
                    logger.debug(
                        "Using preferred model",
                        model=preferred_model,
                        task=task_type.value,
                    )
                    return preferred_model
        
        # Get task-appropriate models in priority order
        candidates = TASK_MODEL_MAP.get(task_type, [])
        
        # Filter by tier if specified
        if tier:
            candidates = [
                m for m in candidates
                if MODEL_REGISTRY.get(m, {}).get("tier") == tier
            ]
        
        # Add default and fallback models to end of list
        if settings.ollama.default_model not in candidates:
            candidates.append(settings.ollama.default_model)
        if settings.ollama.fallback_model not in candidates:
            candidates.append(settings.ollama.fallback_model)
        
        # Try each candidate
        for model in candidates:
            if not validate_model(model):
                continue
            
            if not require_available:
                self._record_selection(model)
                return model
            
            if await self._is_available(model):
                self._record_selection(model)
                logger.info(
                    "Model selected",
                    model=model,
                    task=task_type.value,
                )
                return model
        
        # If no candidate is available, use fallback without availability check
        fallback = settings.ollama.fallback_model
        self._record_selection(fallback)
        self._record_fallback(fallback, task_type.value)
        
        logger.warning(
            "No preferred model available, using fallback",
            fallback=fallback,
            task=task_type.value,
        )
        
        return fallback
    
    async def create_model_chain(
        self,
        task_type: TaskCategory,
        chain_size: int = 2,
    ) -> List[str]:
        """
        Create a chain of models for complex multi-step tasks.
        
        Different models handle different aspects:
        - First model: Initial reasoning/analysis
        - Second model: Refinement/code generation
        - Third model (optional): Review/validation
        
        Args:
            task_type: Primary task category
            chain_size: Number of models in chain (2-3)
            
        Returns:
            List of model names in execution order
        """
        chain = []
        
        if task_type in (TaskCategory.CODE_GENERATION, TaskCategory.CODE_REFACTORING):
            # Reasoning first, then code generation
            reasoning_model = await self.select_model(
                TaskCategory.ARCHITECTURE_ANALYSIS,
                require_available=True,
            )
            chain.append(reasoning_model)
            
            code_model = await self.select_model(
                task_type,
                require_available=True,
            )
            chain.append(code_model)
            
            if chain_size >= 3:
                review_model = await self.select_model(
                    TaskCategory.CODE_REVIEW,
                    require_available=True,
                )
                chain.append(review_model)
        
        elif task_type == TaskCategory.ARCHITECTURE_ANALYSIS:
            # Deep analysis then documentation
            analysis_model = await self.select_model(
                task_type,
                require_available=True,
            )
            chain.append(analysis_model)
            
            doc_model = await self.select_model(
                TaskCategory.DOCUMENTATION,
                require_available=True,
            )
            chain.append(doc_model)
        
        else:
            # Default: use best model, then general model for refinement
            primary = await self.select_model(task_type, require_available=True)
            chain.append(primary)
            
            if chain_size >= 2:
                secondary = await self.select_model(
                    TaskCategory.GENERAL_QA,
                    require_available=True,
                )
                if secondary != primary:
                    chain.append(secondary)
        
        # Ensure we have at least one model
        if not chain:
            chain.append(settings.ollama.default_model)
        
        logger.info(
            "Model chain created",
            chain=chain,
            task=task_type.value,
        )
        
        return chain
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive model information.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        registry_info = MODEL_REGISTRY.get(model_name, {})
        availability = await self._is_available(model_name)
        
        # Get runtime info from Ollama
        ollama_info = await self.ollama.get_model_info(model_name)
        
        return {
            "name": model_name,
            "available": availability,
            "tier": registry_info.get("tier", ModelTier.FAST).value,
            "specialties": [
                s.value for s in registry_info.get("specialties", [])
            ],
            "max_context": registry_info.get("max_context", 4096),
            "quality_score": registry_info.get("quality_weight", 0.5),
            "size": ollama_info.size_formatted if ollama_info else "unknown",
            "modified": (
                ollama_info.modified_at.isoformat()
                if ollama_info else None
            ),
        }
    
    async def list_available_models(
        self,
        task_type: Optional[TaskCategory] = None,
        tier: Optional[ModelTier] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available models, optionally filtered by task/tier.
        
        Args:
            task_type: Filter by task compatibility
            tier: Filter by capability tier
            
        Returns:
            List of model information dictionaries
        """
        all_models = await self.ollama.list_models()
        
        result = []
        for model_info in all_models:
            # Check registry for capabilities
            registry = MODEL_REGISTRY.get(model_info.name, {})
            
            # Apply filters
            if task_type:
                specialties = registry.get("specialties", [])
                if task_type not in specialties:
                    continue
            
            if tier:
                model_tier = registry.get("tier")
                if model_tier != tier:
                    continue
            
            result.append({
                "name": model_info.name,
                "size": model_info.size_formatted,
                "tier": (registry.get("tier") or ModelTier.FAST).value,
                "specialties": [
                    s.value for s in registry.get("specialties", [])
                ],
                "quality_score": registry.get("quality_weight", 0.5),
            })
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for all registered models.
        
        Returns:
            Health status per model
        """
        models_health = {}
        
        for model_name in self._get_all_registered_models():
            is_available = await self._is_available(model_name, force_check=True)
            
            stats = self.ollama.model_stats.get(model_name)
            
            models_health[model_name] = {
                "available": is_available,
                "tier": MODEL_REGISTRY.get(model_name, {}).get("tier", "unknown").value,
                "success_rate": round(stats.success_rate, 2) if stats else None,
                "selection_count": self._selection_counts.get(model_name, 0),
                "fallback_count": self._fallback_counts.get(model_name, 0),
            }
        
        return {
            "total_models": len(models_health),
            "available_models": sum(
                1 for h in models_health.values() if h["available"]
            ),
            "models": models_health,
        }
    
    # ========================================
    # Private Helpers
    # ========================================
    
    async def _is_available(
        self,
        model_name: str,
        force_check: bool = False,
    ) -> bool:
        """
        Check if a model is available, with caching.
        
        Args:
            model_name: Model to check
            force_check: Bypass cache
            
        Returns:
            True if model is available
        """
        # Check cache
        if not force_check and model_name in self._availability_cache:
            available, timestamp = self._availability_cache[model_name]
            if datetime.utcnow() - timestamp < self._availability_ttl:
                return available
        
        # Check with Ollama
        is_available = await self.ollama.is_model_available(model_name)
        
        # Update cache
        self._availability_cache[model_name] = (is_available, datetime.utcnow())
        
        return is_available
    
    def _get_all_registered_models(self) -> List[str]:
        """Get all models from registry."""
        return list(MODEL_REGISTRY.keys())
    
    def _record_selection(self, model_name: str) -> None:
        """Record model selection for statistics."""
        self._selection_counts[model_name] = (
            self._selection_counts.get(model_name, 0) + 1
        )
    
    def _record_fallback(self, model_name: str, task_type: str) -> None:
        """Record fallback usage."""
        self._fallback_counts[model_name] = (
            self._fallback_counts.get(model_name, 0) + 1
        )
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get model selection statistics."""
        return {
            "selections": self._selection_counts,
            "fallbacks": self._fallback_counts,
            "total_selections": sum(self._selection_counts.values()),
            "total_fallbacks": sum(self._fallback_counts.values()),
            "fallback_rate": (
                sum(self._fallback_counts.values()) /
                max(sum(self._selection_counts.values()), 1) * 100
            ),
        }

    def update_model_weights(self, model_weights: Dict[str, float]) -> Dict[str, float]:
        """Update quality_weight for the given models in the live registry.

        Called by EvolutionService to reflect empirical performance data.
        Returns the previous weights so callers can issue a rollback.

        Args:
            model_weights: mapping of model_name → new quality_weight (clamped to [0, 1])
        """
        previous: Dict[str, float] = {}
        for model_name, new_weight in model_weights.items():
            if model_name in MODEL_REGISTRY:
                previous[model_name] = MODEL_REGISTRY[model_name].get("quality_weight", 0.5)
                clamped = max(0.0, min(1.0, new_weight))
                MODEL_REGISTRY[model_name]["quality_weight"] = clamped
                logger.info(
                    "Model quality weight updated",
                    model=model_name,
                    old=previous[model_name],
                    new=clamped,
                )
        return previous

    def get_current_weights(self) -> Dict[str, Dict[str, float]]:
        """Return quality_weight and cost_weight for every registered model.

        Used by EvolutionService to snapshot state before applying changes.
        """
        return {
            name: {
                "quality_weight": info.get("quality_weight", 0.5),
                "cost_weight": info.get("cost_weight", 1.0),
            }
            for name, info in MODEL_REGISTRY.items()
        }


# Default service instance
_default_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the default model service."""
    global _default_service
    if _default_service is None:
        _default_service = ModelService()
    return _default_service


logger.info("Model service module initialized successfully")