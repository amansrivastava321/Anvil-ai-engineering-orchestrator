# Graph Report - .  (2026-05-15)

## Corpus Check
- 175 files · ~110,796 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2864 nodes · 7891 edges · 94 communities detected
- Extraction: 44% EXTRACTED · 56% INFERRED · 0% AMBIGUOUS · INFERRED: 4422 edges (avg confidence: 0.5)
- Token cost: 20,150 input · 5,100 output

## God Nodes (most connected - your core abstractions)
1. `InputSanitizer` - 209 edges
2. `OllamaClient` - 196 edges
3. `MetricsTracker` - 170 edges
4. `PathValidator` - 158 edges
5. `LogContext` - 136 edges
6. `SkillfileClient` - 132 edges
7. `CircuitBreakerOpenError` - 118 edges
8. `AgentTask` - 114 edges
9. `GraphifyParser` - 113 edges
10. `OllamaClientError` - 103 edges

## Surprising Connections (you probably didn't know these)
- `Tests for app.tools.code_analysis.dependency_analyzer — DependencyAnalyzer tool.` --uses--> `DependencyAnalyzer`  [INFERRED]
  tests/unit/test_dependency_analyzer.py → app/tools/code_analysis/dependency_analyzer.py
- `Tests for app.tools.code_analysis.security_scanner — SecurityScanner tool.` --uses--> `SecurityScanner`  [INFERRED]
  tests/unit/test_security_scanner.py → app/tools/code_analysis/security_scanner.py
- `Tests for app.tools.file_system.file_search — FileSearch tool.` --uses--> `FileSearch`  [INFERRED]
  tests/unit/test_file_search.py → app/tools/file_system/file_search.py
- `Tests for app.memory.pattern_store — PatternStore learning patterns.` --uses--> `PatternStore`  [INFERRED]
  tests/unit/test_pattern_store.py → app/memory/pattern_store.py
- `Tests for app.memory.execution_memory — ExecutionMemory run tracking.` --uses--> `ExecutionMemory`  [INFERRED]
  tests/unit/test_execution_memory.py → app/memory/execution_memory.py

## Hyperedges (group relationships)
- **Multi-Agent Workflow Execution Pipeline** — orchestrator_service, context_service, agent_factory, artifact_store, execution_memory [EXTRACTED 1.00]
- **AAOS Autonomous Detection and Execution Loop** — file_watcher_service, detection_engine, autonomous_work_queue, planner_agent, verification_agent [EXTRACTED 1.00]
- **Memory and Learning Feedback Loop** — artifact_store, execution_memory, pattern_store, learning_service [EXTRACTED 1.00]

## Communities

### Community 0 - "Agent Orchestration Core"
Cohesion: 0.03
Nodes (258): AgentFactory, AssembledContext, ChatResponse, get_skillfile_client(), ModelNotFoundError, ModelTimeoutError, Production-grade Ollama client with enterprise features: - Circuit breaker patte, Loaded skill content with metadata. (+250 more)

### Community 1 - "Config & Logging Infrastructure"
Cohesion: 0.01
Nodes (113): BaseSettings, Enum, add_context_vars(), add_environment(), add_exception_info(), add_host_info(), add_log_level(), add_process_info() (+105 more)

### Community 2 - "Agent Framework Base"
Cohesion: 0.02
Nodes (123): ABC, get_agent_factory(), Agent factory — creates and caches specialized agents by type name.  Registry us, Creates and caches agent instances by type name., Return a cached agent instance for the given type.          Raises KeyError if a, Return registered agent types mapped to their class import paths., Dynamically import the agent class for the given type., Return the global AgentFactory singleton. (+115 more)

### Community 3 - "CEO Decision Engine"
Cohesion: 0.04
Nodes (100): BaseModel, CEO, get_ceo(), _ollama_llm(), CEO AI — Chief Engineering Officer.  The most important component. The CEO: - Re, Record the outcome of a decision and update learned patterns., Mode 4: CEO-initiated strategic work without being asked.          Analyzes syst, Ask the CEO to analyze its own history and discover new patterns. (+92 more)

### Community 4 - "Agent Execution & Tool Interface"
Cohesion: 0.03
Nodes (93): AgentStep, Identifier used in agent reasoning and logging., Human-readable description injected into the agent's system prompt., Run the tool. Must not raise — return ToolResult with success=False on error., Agent identifier — used in logs and metrics., One-line description of this agent's specialization., Core system prompt. Defines the agent's expertise and discipline., Tools available to this agent. Override to add domain-specific tools. (+85 more)

### Community 5 - "Evolution & Self-Improvement"
Cohesion: 0.03
Nodes (99): evolution_history(), evolution_status(), EvolutionCycle, EvolutionRecommendation, ModelPerformanceRecord, Evolution API endpoints — trigger, inspect, and rollback evolution cycles., Run a full evolution cycle immediately and return the results.      Pass ``force, How well a specific model performs on a specific task type. (+91 more)

### Community 6 - "API Layer"
Cohesion: 0.02
Nodes (84): get_active_executions(), get_ceo_decisions(), get_ceo_status(), get_execution_stats(), Agent API endpoints — all requests go through the CEO AI., Return recent CEO decisions for the dashboard., Return aggregate CEO statistics for the dashboard., Record the outcome of a CEO decision (for learning). (+76 more)

### Community 7 - "Context Assembly Tests"
Cohesion: 0.02
Nodes (5): test_load_graphify_context_all_sections_full_budget(), test_load_graphify_context_valid_is_valid_false(), test_load_graphify_context_with_app_map(), test_load_graphify_context_with_graph_data(), test_load_graphify_context_with_valid_output_summary()

### Community 8 - "Monitoring & Change Detection"
Cohesion: 0.03
Nodes (28): ChangeEvent, _current_branch(), _current_commit(), _diff_files(), MonitorAgent, RepoStatus, get_proactive_service(), ProactiveService (+20 more)

### Community 9 - "Skillfile Integration"
Cohesion: 0.04
Nodes (50): Result from searching community skill registries., skillfile CLI not installed., AI coding platforms that skills deploy to., SkillfileNotInstalledError, SkillPlatform, SkillSearchResult, _make_client(), Tests for app.integrations.skillfile.client — data models and SkillfileClient. (+42 more)

### Community 10 - "Autonomous OS Vision"
Cohesion: 0.04
Nodes (50): AI Engineering Intelligence OS, ArtifactStore, Autonomous Audit Operating System (AAOS), Autonomous Audit Roadmap, AutonomousWorkQueue, Business Context Model, Current System Audit 2026-05-14, deepseek-r1:7b Model (+42 more)

### Community 11 - "Orchestrator Tests"
Cohesion: 0.04
Nodes (24): _make_mock_context(), mock_context_service(), test_assemble_context_with_invalid_mode_falls_back_to_balanced(), test_assemble_context_with_valid_mode(), test_execute_architecture_analysis_calls_single_agent(), test_execute_circuit_breaker_open_uses_fallback(), test_execute_code_generation(), test_execute_code_refactoring_calls_single_agent() (+16 more)

### Community 12 - "CLI Tests"
Cohesion: 0.06
Nodes (16): _fake_thread_factory(), _make_client_ctx(), _mock_response(), _orch_response(), Tests for the ae CLI tool., Thread whose start() calls target() synchronously., Success rate shows — when total_executions is 0., main() inserts 'run' when first positional is not a subcommand. (+8 more)

### Community 13 - "AI Organization Tests"
Cohesion: 0.04
Nodes (6): ceo(), mock_llm(), test_50_decisions_builds_history(), test_confidence_increases_with_successes(), test_high_confidence_low_risk_decide_alone(), test_moderate_confidence_consults_experts()

### Community 14 - "Model Registry"
Cohesion: 0.04
Nodes (12): ModelTier, Model capability tiers., get_model_info(), list_models(), models_health(), Model management endpoints., List available models, optionally filtered by task or tier.          Task types:, Get detailed information about a specific model. (+4 more)

### Community 15 - "API Endpoint Tests"
Cohesion: 0.04
Nodes (1): _make_app()

### Community 16 - "Settings Tests"
Cohesion: 0.05
Nodes (0): 

### Community 17 - "Memory Store"
Cohesion: 0.09
Nodes (5): get_memory_store(), MemoryStore, Base memory store — JSON file-backed key/value store for agent memory., Lightweight JSON-backed memory store for agent execution history., Tests for app.memory.memory_store — MemoryStore key-value persistence.

### Community 18 - "CLI Interface"
Cohesion: 0.15
Nodes (25): cli(), _client(), dashboard(), detect_workflow(), _format_duration(), get_base_url(), get_repo_path(), _handle_connection_error() (+17 more)

### Community 19 - "Database Layer"
Cohesion: 0.11
Nodes (14): DatabaseError, get_db(), InMemoryDatabase, Database helper for ShopFlow (in-memory store for demo purposes)., Raised when a database operation fails., Thread-safe in-memory key-value store that mimics basic DB operations., Insert a record into a table. Assigns an auto-increment id., Retrieve a record by id. Returns None if not found. (+6 more)

### Community 20 - "Context Service Tests"
Cohesion: 0.1
Nodes (0): 

### Community 21 - "Dependency Analyzer Tests"
Cohesion: 0.11
Nodes (1): Tests for app.tools.code_analysis.dependency_analyzer — DependencyAnalyzer tool.

### Community 22 - "Retry & Circuit Breaker Tests"
Cohesion: 0.11
Nodes (0): 

### Community 23 - "Agent Unit Tests"
Cohesion: 0.11
Nodes (0): 

### Community 24 - "Artifact Store Tests"
Cohesion: 0.12
Nodes (0): 

### Community 25 - "Model Config"
Cohesion: 0.12
Nodes (3): ModelConfig, Model configuration — defines available models and their capabilities., Tests for app.core.config.models_config — ModelConfig and helpers.

### Community 26 - "File Tool Tests"
Cohesion: 0.12
Nodes (0): 

### Community 27 - "Security Scanner Tests"
Cohesion: 0.13
Nodes (1): Tests for app.tools.code_analysis.security_scanner — SecurityScanner tool.

### Community 28 - "Execution Memory Tests"
Cohesion: 0.13
Nodes (1): Tests for app.memory.execution_memory — ExecutionMemory run tracking.

### Community 29 - "Graphify Parser Tests"
Cohesion: 0.13
Nodes (1): Tests for app.integrations.graphify.parser — GraphifyParser + GraphifyWrapper.

### Community 30 - "Auth Tests"
Cohesion: 0.13
Nodes (3): Tests for the auth module — all tests pass., TestPasswordHashing, TestTokenLifecycle

### Community 31 - "Ollama Client Tests"
Cohesion: 0.14
Nodes (0): 

### Community 32 - "File Search Tests"
Cohesion: 0.14
Nodes (1): Tests for app.tools.file_system.file_search — FileSearch tool.

### Community 33 - "Artifact Store Core"
Cohesion: 0.17
Nodes (5): Legacy: save a single artifact. Delegates to save_run., Legacy: get artifact. Delegates to get_run., Legacy: list artifacts for an execution., Persist a run and return its run_id., Load a run by ID. Returns None if not found.

### Community 34 - "Graphify Parser Core"
Cohesion: 0.17
Nodes (6): Parse a repository and return its graphify output dict., Return modules affected by changes to target_files., Return dependency graph for a specific module., Check if graphify has already been run on this repo., Execute graphify on the repository., Load existing graphify output.

### Community 35 - "Health Endpoints"
Cohesion: 0.17
Nodes (11): liveness_check(), System health and status endpoints., Get comprehensive system status including Ollama, CEO, and orchestrator health., Get execution statistics., Get basic system information., Kubernetes-style readiness probe.     Returns 200 if the system is ready to acce, Kubernetes-style liveness probe.     Returns 200 if the server is alive., readiness_check() (+3 more)

### Community 36 - "Retry & Circuit Breaker"
Cohesion: 0.2
Nodes (5): async_retry(), Retry utilities with exponential backoff and circuit breaker., Decorator: retry an async function with exponential backoff., Advance OPEN → HALF_OPEN if recovery_timeout has elapsed., state()

### Community 37 - "Pattern Store Tests"
Cohesion: 0.18
Nodes (1): Tests for app.memory.pattern_store — PatternStore learning patterns.

### Community 38 - "Security Agent Tests"
Cohesion: 0.18
Nodes (0): 

### Community 39 - "Performance Agent Tests"
Cohesion: 0.18
Nodes (0): 

### Community 40 - "Monitor Endpoints"
Cohesion: 0.2
Nodes (9): monitor_status(), MonitorStartRequest, MonitorStopRequest, Monitor API endpoints — start/stop proactive repository monitoring., Begin proactive monitoring for the given repository paths.      Already-watched, Stop monitoring specific repositories, or all if ``repo_paths`` is     omitted., Return real-time status for all watched repositories., start_monitoring() (+1 more)

### Community 41 - "Workflow Tests"
Cohesion: 0.39
Nodes (7): make_mock_factory(), test_debug_workflow_calls_agents_in_order(), test_debug_workflow_returns_result(), test_refactor_workflow_calls_three_agents(), test_refactor_workflow_returns_result(), test_testing_workflow_calls_two_agents(), test_testing_workflow_returns_result()

### Community 42 - "Metrics Tests"
Cohesion: 0.22
Nodes (0): 

### Community 43 - "Audit Workflow Tests"
Cohesion: 0.33
Nodes (5): _make_agent_result(), mock_factory(), test_audit_workflow_returns_result(), test_audit_workflow_risk_score_critical_on_critical_findings(), test_audit_workflow_risk_score_low_on_no_findings()

### Community 44 - "Report Workflow Tests"
Cohesion: 0.28
Nodes (3): _make_agent_result(), mock_factory(), test_report_workflow_extracts_recommendations()

### Community 45 - "Coverage JS UI"
Cohesion: 0.29
Nodes (2): getCellValue(), rowComparator()

### Community 46 - "Coverage.py Assets"
Cohesion: 0.33
Nodes (7): Coverage.py Tool, coverage.py Tool, Coverage.py Favicon (32px), HTML Coverage Report, Keyboard Closed Icon, Keyboard Shortcuts UI Toggle, Python Snake Logo

### Community 47 - "Artifact Store Singleton"
Cohesion: 0.5
Nodes (3): get_artifact_store(), Artifact store — persists every agent run to data/artifacts/runs/{run_id}/., Return the process-wide ArtifactStore singleton.

### Community 48 - "Prometheus Metrics"
Cohesion: 0.5
Nodes (3): Prometheus metrics for the orchestrator.  Exports required by orchestrator.py:, Instrument the FastAPI app with Prometheus metrics middleware., setup_metrics()

### Community 49 - "API Router"
Cohesion: 1.0
Nodes (1): API v1 router - aggregates all endpoint routers.

### Community 50 - "AST Analyzer"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Coverage Analyzer"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Security Middleware"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Memory Cache"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Redis Cache"
Cohesion: 1.0
Nodes (0): 

### Community 55 - "Database URL Builder"
Cohesion: 1.0
Nodes (1): Construct database URL.

### Community 56 - "Sync Database URL"
Cohesion: 1.0
Nodes (1): Construct synchronous database URL (for Alembic).

### Community 57 - "Pool Size Validator"
Cohesion: 1.0
Nodes (1): Validate pool size based on environment.

### Community 58 - "Connection Validator"
Cohesion: 1.0
Nodes (1): Validate connections based on environment.

### Community 59 - "Ollama URL Validator"
Cohesion: 1.0
Nodes (1): Validate and normalize Ollama URL.

### Community 60 - "Secret Key Validator"
Cohesion: 1.0
Nodes (1): Ensure secret key is strong in production.

### Community 61 - "Environment Normalizer"
Cohesion: 1.0
Nodes (1): Normalize environment value.

### Community 62 - "Production Debug Guard"
Cohesion: 1.0
Nodes (1): Ensure debug is disabled in production.

### Community 63 - "Worker Count Validator"
Cohesion: 1.0
Nodes (1): Validate worker count against CPU cores.

### Community 64 - "Directory Initializer"
Cohesion: 1.0
Nodes (1): Create necessary directories if they don't exist.

### Community 65 - "Production Validator"
Cohesion: 1.0
Nodes (1): Additional validation for production environment.

### Community 66 - "Production Checker"
Cohesion: 1.0
Nodes (1): Check if running in production.

### Community 67 - "Development Checker"
Cohesion: 1.0
Nodes (1): Check if running in development.

### Community 68 - "Test Mode Checker"
Cohesion: 1.0
Nodes (1): Check if running tests.

### Community 69 - "Config Source Priority"
Cohesion: 1.0
Nodes (1): Customize configuration source priority.

### Community 70 - "Tracing"
Cohesion: 1.0
Nodes (0): 

### Community 71 - "Formatters"
Cohesion: 1.0
Nodes (0): 

### Community 72 - "Async Utilities"
Cohesion: 1.0
Nodes (0): 

### Community 73 - "Path Allowlist"
Cohesion: 1.0
Nodes (1): Set the allowed base directories for path validation.

### Community 74 - "Path Security Validator"
Cohesion: 1.0
Nodes (1): Comprehensive path validation with security checks.                  Args:

### Community 75 - "Type Validator"
Cohesion: 1.0
Nodes (1): Validate that a value matches the expected type.                  Args:

### Community 76 - "Optional Type Validator"
Cohesion: 1.0
Nodes (1): Validate optional type (allows None).

### Community 77 - "List Type Validator"
Cohesion: 1.0
Nodes (1): Validate list with specific item type.

### Community 78 - "Dict Type Validator"
Cohesion: 1.0
Nodes (1): Validate dictionary with specific key/value types.

### Community 79 - "Input Sanitizer"
Cohesion: 1.0
Nodes (1): Sanitize string input to prevent injection attacks.                  Args:

### Community 80 - "Filename Sanitizer"
Cohesion: 1.0
Nodes (1): Sanitize a filename to prevent path traversal.

### Community 81 - "Pydantic Schema Validator"
Cohesion: 1.0
Nodes (1): Validate data against a Pydantic schema.                  Args:             data

### Community 82 - "API Response Models"
Cohesion: 1.0
Nodes (0): 

### Community 83 - "API Schemas"
Cohesion: 1.0
Nodes (0): 

### Community 84 - "API Request Models"
Cohesion: 1.0
Nodes (0): 

### Community 85 - "SQLAlchemy Models"
Cohesion: 1.0
Nodes (0): 

### Community 86 - "Repository Base"
Cohesion: 1.0
Nodes (0): 

### Community 87 - "Windsurf Integration"
Cohesion: 1.0
Nodes (0): 

### Community 88 - "Streaming"
Cohesion: 1.0
Nodes (0): 

### Community 89 - "Test Config"
Cohesion: 1.0
Nodes (0): 

### Community 90 - "Architecture Analysis Artifact"
Cohesion: 1.0
Nodes (1): Artifact Run: architecture_analysis minimal (16f2d24c)

### Community 91 - "General Repo Analysis Artifact"
Cohesion: 1.0
Nodes (1): Artifact Run: general repo analysis (c946542d)

### Community 92 - "Coverage Favicon"
Cohesion: 1.0
Nodes (1): Coverage.py Favicon (32px)

### Community 93 - "Keyboard Help Icon"
Cohesion: 1.0
Nodes (1): Keyboard Shortcut Help Panel

## Knowledge Gaps
- **208 isolated node(s):** `Artifact store — persists every agent run to data/artifacts/runs/{run_id}/.`, `Stores and retrieves run artifacts from the local filesystem.`, `Persist a run and return its run_id.`, `Load a run by ID. Returns None if not found.`, `List runs, optionally filtered by repo_path, newest first.` (+203 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `API Router`** (2 nodes): `router.py`, `API v1 router - aggregates all endpoint routers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AST Analyzer`** (1 nodes): `ast_analyzer.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Coverage Analyzer`** (1 nodes): `coverage_analyzer.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Security Middleware`** (1 nodes): `security.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Memory Cache`** (1 nodes): `memory_cache.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Redis Cache`** (1 nodes): `redis_cache.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Database URL Builder`** (1 nodes): `Construct database URL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sync Database URL`** (1 nodes): `Construct synchronous database URL (for Alembic).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pool Size Validator`** (1 nodes): `Validate pool size based on environment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Connection Validator`** (1 nodes): `Validate connections based on environment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ollama URL Validator`** (1 nodes): `Validate and normalize Ollama URL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Secret Key Validator`** (1 nodes): `Ensure secret key is strong in production.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Environment Normalizer`** (1 nodes): `Normalize environment value.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Production Debug Guard`** (1 nodes): `Ensure debug is disabled in production.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Worker Count Validator`** (1 nodes): `Validate worker count against CPU cores.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Directory Initializer`** (1 nodes): `Create necessary directories if they don't exist.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Production Validator`** (1 nodes): `Additional validation for production environment.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Production Checker`** (1 nodes): `Check if running in production.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Development Checker`** (1 nodes): `Check if running in development.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Mode Checker`** (1 nodes): `Check if running tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Source Priority`** (1 nodes): `Customize configuration source priority.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tracing`** (1 nodes): `tracing.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Formatters`** (1 nodes): `formatters.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Async Utilities`** (1 nodes): `async_utils.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Path Allowlist`** (1 nodes): `Set the allowed base directories for path validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Path Security Validator`** (1 nodes): `Comprehensive path validation with security checks.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Type Validator`** (1 nodes): `Validate that a value matches the expected type.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Optional Type Validator`** (1 nodes): `Validate optional type (allows None).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `List Type Validator`** (1 nodes): `Validate list with specific item type.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dict Type Validator`** (1 nodes): `Validate dictionary with specific key/value types.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Input Sanitizer`** (1 nodes): `Sanitize string input to prevent injection attacks.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Filename Sanitizer`** (1 nodes): `Sanitize a filename to prevent path traversal.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pydantic Schema Validator`** (1 nodes): `Validate data against a Pydantic schema.                  Args:             data`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Response Models`** (1 nodes): `responses.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Schemas`** (1 nodes): `schemas.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Request Models`** (1 nodes): `requests.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SQLAlchemy Models`** (1 nodes): `sqlalchemy_models.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Repository Base`** (1 nodes): `base.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Windsurf Integration`** (1 nodes): `windsurf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Streaming`** (1 nodes): `streaming.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Architecture Analysis Artifact`** (1 nodes): `Artifact Run: architecture_analysis minimal (16f2d24c)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `General Repo Analysis Artifact`** (1 nodes): `Artifact Run: general repo analysis (c946542d)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Coverage Favicon`** (1 nodes): `Coverage.py Favicon (32px)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Keyboard Help Icon`** (1 nodes): `Keyboard Shortcut Help Panel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OllamaClient` connect `Agent Execution & Tool Interface` to `Agent Orchestration Core`, `Agent Framework Base`, `Monitoring & Change Detection`, `Autonomous OS Vision`, `Model Registry`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `LogContext` connect `Agent Orchestration Core` to `Config & Logging Infrastructure`, `Agent Execution & Tool Interface`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `AgentTask` connect `Agent Framework Base` to `Agent Orchestration Core`, `Monitoring & Change Detection`, `CEO Decision Engine`, `Agent Execution & Tool Interface`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Are the 207 inferred relationships involving `InputSanitizer` (e.g. with `SkillType` and `SkillSource`) actually correct?**
  _`InputSanitizer` has 207 INFERRED edges - model-reasoned connections that need verification._
- **Are the 176 inferred relationships involving `OllamaClient` (e.g. with `AgentFactory` and `Agent factory — creates and caches specialized agents by type name.  Registry us`) actually correct?**
  _`OllamaClient` has 176 INFERRED edges - model-reasoned connections that need verification._
- **Are the 165 inferred relationships involving `MetricsTracker` (e.g. with `ModelStatus` and `ChatRole`) actually correct?**
  _`MetricsTracker` has 165 INFERRED edges - model-reasoned connections that need verification._
- **Are the 156 inferred relationships involving `PathValidator` (e.g. with `FileReader` and `FileReader — read repository files safely.`) actually correct?**
  _`PathValidator` has 156 INFERRED edges - model-reasoned connections that need verification._