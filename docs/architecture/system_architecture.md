# System Architecture — AI Engineering Orchestrator

## Overview

The AI Engineering Orchestrator is an **Autonomous Engineering Intelligence OS** — a graph-aware, multi-agent system that analyzes and improves codebases using local LLMs via Ollama, knowledge graphs via Graphify, and structured AI skills via Skillfile. It is designed to be deployed entirely on-premise, with no data leaving the developer's machine.

The system is built around three core principles:

1. **Graph-aware reasoning** — every request is enriched with structural knowledge about the codebase extracted by Graphify, so the LLM sees relationships between modules, not just isolated files.
2. **Specialized agents** — a registry of purpose-built agents (CodeAgent, ArchitectureAgent, SecurityAgent, etc.) each with focused system prompts, domain tools, and narrow responsibilities.
3. **Observable execution** — every run is logged, every artifact is persisted, every model call is measured. No invisible state.

---

## Full System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AI Engineering Orchestrator                            │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        HTTP API Layer (FastAPI)                         │   │
│   │                                                                         │   │
│   │   POST /api/v1/agent/run          GET /api/v1/agent/status/{id}        │   │
│   │   POST /api/v1/agent/stream       GET /api/v1/artifacts/{id}           │   │
│   │   POST /api/v1/workflows/audit    GET /api/v1/health                   │   │
│   └────────────────────────────┬────────────────────────────────────────────┘   │
│                                │ Request validated by Pydantic                  │
│                                ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     Orchestrator Service (1200+ lines)                  │   │
│   │                                                                         │   │
│   │  ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │   │
│   │  │  Request Router  │   │  Model Selector   │   │  Retry Manager    │  │   │
│   │  │  (workflow type) │   │  (TaskCategory)   │   │  (CircuitBreaker) │  │   │
│   │  └────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘  │   │
│   │           │                       │                         │           │   │
│   │           └───────────────────────┼─────────────────────────┘           │   │
│   │                                   │                                     │   │
│   │  ┌────────────────────────────────▼────────────────────────────────┐   │   │
│   │  │                   Execution Pipeline                             │   │   │
│   │  │                                                                  │   │   │
│   │  │  1. Validate & sanitize input (PathValidator, InputSanitizer)   │   │   │
│   │  │  2. Select model (ModelService → TaskCategory → model name)     │   │   │
│   │  │  3. Assemble context (ContextAssembler → AssembledContext)      │   │   │
│   │  │  4. Route to workflow or direct agent run                       │   │   │
│   │  │  5. Execute with metrics tracking (MetricsTracker)              │   │   │
│   │  │  6. Persist artifacts (ArtifactStore)                           │   │   │
│   │  │  7. Record in execution memory (ExecutionMemory)                │   │   │
│   │  └──────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌──────────────────────────┐   ┌───────────────────────────────────────────┐  │
│   │     Context Assembly     │   │              Agent Layer                  │  │
│   │   (ContextAssembler)     │   │                                           │  │
│   │                          │   │  ┌────────────┐  ┌──────────────────────┐ │  │
│   │  ┌──────────────────┐    │   │  │ AgentFactory│  │     BaseAgent        │ │  │
│   │  │  Graphify Parser │    │   │  │ (singleton) │  │  ┌────────────────┐  │ │  │
│   │  │  · graph summary │    │   │  │  Registry:  │  │  │ system_prompt  │  │ │  │
│   │  │  · app map       │    │   │  │  code       │  │  │ tools[]        │  │ │  │
│   │  │  · dependency    │    │   │  │  arch       │  │  │ _execute()     │  │ │  │
│   │  │    graph         │    │   │  │  testing    │  │  │ run()          │  │ │  │
│   │  └──────────────────┘    │   │  │  docs       │  │  │ run_streaming()│  │ │  │
│   │  ┌──────────────────┐    │   │  │  security   │  │  └────────────────┘  │ │  │
│   │  │  Skillfile Client│    │   │  │  perf       │  └──────────────────────┘ │  │
│   │  │  · skill text    │    │   │  └─────────────┘                          │  │
│   │  │  · best practice │    │   │                                           │  │
│   │  └──────────────────┘    │   │  ┌────────────────────────────────────┐  │  │
│   │  ┌──────────────────┐    │   │  │         Workflow Layer             │  │  │
│   │  │  File Scanner    │    │   │  │                                    │  │  │
│   │  │  · source files  │    │   │  │  AuditWorkflow    DebugWorkflow    │  │  │
│   │  │  · test files    │    │   │  │  RefactorWorkflow TestingWorkflow  │  │  │
│   │  │  · config files  │    │   │  │  ReportWorkflow                   │  │  │
│   │  └──────────────────┘    │   │  └────────────────────────────────────┘  │  │
│   │  Token Budget Manager    │   └───────────────────────────────────────────┘  │
│   └──────────────────────────┘                                                  │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                         Integration Layer                                │  │
│   │                                                                          │  │
│   │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │  │
│   │  │  Ollama Client  │  │  Graphify Parser  │  │   Skillfile Client     │  │  │
│   │  │                 │  │                  │  │                        │  │  │
│   │  │  CircuitBreaker │  │  GraphifyWrapper  │  │  HTTP REST client      │  │  │
│   │  │  Retry logic    │  │  GraphifyParser   │  │  Skill text injection  │  │  │
│   │  │  Streaming      │  │  Graph analysis   │  │                        │  │  │
│   │  │  Health checks  │  │                  │  │                        │  │  │
│   │  └────────┬────────┘  └────────┬─────────┘  └────────────────────────┘  │  │
│   └───────────┼───────────────────┼──────────────────────────────────────────┘  │
│               │                   │                                              │
│   ┌───────────▼───────┐   ┌───────▼─────────┐  ┌──────────────────────────┐   │
│   │   Ollama (local)  │   │   Graphify OS    │  │     Memory & Storage     │   │
│   │                   │   │                  │  │                          │   │
│   │  qwen2.5-coder:7b │   │  Knowledge graph │  │  ExecutionMemory (JSON)  │   │
│   │  codellama        │   │  Code structure  │  │  PatternStore (JSON)     │   │
│   │  llama3.2         │   │  Module map      │  │  MemoryStore (JSON)      │   │
│   │  deepseek-coder   │   │                  │  │  ArtifactStore (files)   │   │
│   └───────────────────┘   └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### Orchestrator Service (`app/services/orchestrator.py`)

The orchestrator is the central coordinator — the "conductor" that sequences every other subsystem. It is ~1200 lines of async Python and handles:

- **Request routing**: determines whether a prompt maps to a direct agent call, a structured workflow (audit, debug, refactor, testing), or a streaming response
- **Model selection**: delegates to `ModelService` to pick the right Ollama model for the inferred task category
- **Context orchestration**: invokes `ContextAssembler` and waits for the enriched `AssembledContext` before forwarding to an agent
- **Execution lifecycle**: wraps every run with `MetricsTracker`, catches errors, records results in `ExecutionMemory`, and writes artifacts to `ArtifactStore`
- **Concurrency**: fully async; concurrent requests share the same `OllamaClient` connection pool
- **Fallback strategy**: if the preferred model is unavailable, the orchestrator retries with progressively smaller models before returning a structured error

The orchestrator never generates text itself. It only routes, coordinates, and observes.

### Context Service (`app/services/context_service.py`)

The context service is the intelligence layer — it determines what the LLM sees. At ~1380 lines it is the most complex single file in the system. Its key responsibilities:

- **Token budget management**: tracks a per-request token budget and allocates it across context sources in priority order (CRITICAL → HIGH → MEDIUM → LOW → OPTIONAL)
- **Graphify integration**: pulls the knowledge graph summary, dependency graph, and app map for the target repo and serializes them into prompt-safe text
- **Skillfile injection**: fetches relevant skills from Skillfile and prepends them to context so small (7B) models have structured guidance
- **File content**: scans and ranks source files, test files, config files; reads and includes the most relevant ones within budget
- **Context caching**: hashes the (repo_path, mode, prompt) tuple and serves cached `AssembledContext` for repeated queries within a TTL
- **Compression**: if context exceeds the model's token limit, it summarizes lower-priority sections and truncates gracefully

The result is an `AssembledContext` dataclass containing all assembled sections, token counts per source, and metadata about what was included or dropped.

### Model Service (`app/services/model_service.py`)

Maintains a `TaskCategory` enum (CODE_GENERATION, ARCHITECTURE_REVIEW, SECURITY_SCAN, TESTING, DOCUMENTATION, DEBUGGING, PERFORMANCE_ANALYSIS) and a mapping from category to preferred Ollama model name. Selection logic:

1. Inspect the prompt for keywords and infer the task category
2. Look up the preferred model for that category
3. Validate the model is registered in Ollama; fall back if not
4. Return the model name to the orchestrator

This service also surfaces per-model performance statistics from `ExecutionMemory` to enable future data-driven model selection.

### Agent Layer (`app/agents/`)

Agents are **reasoning entities**, not thin wrappers around Ollama. Each agent:

- Has a focused `system_prompt` expressing its domain expertise and constraints
- Exposes a `tools` list (e.g., `FileReader`, `FileWriter`, `TestRunner`) that it can invoke during execution
- Implements `_execute(task)` to produce a raw response string, and `run(task)` to wrap it in a measured `AgentResult`
- Supports streaming via `run_streaming(task)` which yields token chunks as an async generator

Agents are created and cached by `AgentFactory`, which uses a string registry and dynamic imports to avoid circular dependencies. See [agent_framework.md](agent_framework.md) for the full breakdown.

### Workflow Layer (`app/workflows/`)

Workflows coordinate **multiple agents in sequence** to accomplish compound tasks:

| Workflow | Agents Used | Description |
|---|---|---|
| `AuditWorkflow` | ArchitectureAgent → SecurityAgent → DocumentationAgent | Full codebase audit: architecture summary, security scan, report |
| `DebugWorkflow` | CodeAgent → TestingAgent | Reproduce bug, generate fix, verify with tests |
| `RefactorWorkflow` | CodeAgent → ArchitectureAgent | Identify refactor targets, apply changes, validate |
| `TestingWorkflow` | TestingAgent → CodeAgent | Generate test suite, run tests, fix failures |
| `ReportWorkflow` | DocumentationAgent | Aggregate findings from other workflows into a Markdown report |

Workflows are invoked by the orchestrator based on the request's `workflow_type` field. They return typed result dataclasses (e.g., `AuditWorkflowResult`) with a `.to_markdown()` method.

### Integration Layer (`app/integrations/`)

#### Ollama Client (`app/integrations/ollama/client.py`)

Production-grade HTTP client for Ollama's local API. Built on `httpx` with:

- **Circuit breaker**: three states (CLOSED, OPEN, HALF_OPEN). Opens after 5 consecutive failures; half-opens after a cooldown period to test recovery
- **Retry with exponential backoff**: uses `tenacity` with jitter to avoid thundering herd on retry storms
- **Connection pooling**: single `httpx.AsyncClient` shared across all requests
- **Streaming**: async generator that yields token chunks directly from the Ollama `/api/generate` SSE stream
- **Health checks**: periodic `/api/tags` poll to track model availability

#### Graphify Parser (`app/integrations/graphify/parser.py`)

Two classes handle the Graphify integration:

- `GraphifyWrapper`: thin subprocess wrapper that runs the `graphify` CLI against the repo path and captures JSON output
- `GraphifyParser`: parses the JSON output into structured objects — module nodes, dependency edges, complexity scores, call graphs

The parsed output is handed to `ContextAssembler` which serializes it into prompt-friendly text blocks.

#### Skillfile Client (`app/integrations/skillfile/client.py`)

REST client for the Skillfile service. Fetches skill definitions (structured best practices and task-specific instructions) and injects them into context at CRITICAL priority so even small 7B models have the guidance they need to produce quality output.

### Memory Layer (`app/memory/`)

See [memory_and_learning.md](memory_and_learning.md) for full details. Brief summary:

- `ExecutionMemory`: append-only log of every run (model, duration, success/failure), capped at 1000 entries, persisted as JSON
- `PatternStore`: keyed by `(pattern_type, pattern_key)`, accumulates outcome histories; used to find the best model per task type
- `MemoryStore`: simple string key-value store backed by JSON file; used for arbitrary persistent state

### Artifact Store (`app/artifacts/store.py`)

Persists the full record of every run to `data/artifacts/runs/{run_id}/`:

```
data/artifacts/runs/20260514_142301_a3f9b1c2/
├── run.json        ← canonical record (all metadata)
├── prompt.md       ← the exact prompt sent to the model
├── response.md     ← the model's raw response
├── context.json    ← the assembled context that was used
├── logs.txt        ← structured log lines from the run
├── test_results.json  ← if tests were run
└── patches.diff    ← if code was modified
```

Run IDs are `{timestamp}_{uuid8}` to be both human-readable and globally unique.

---

## Request Lifecycle: POST /api/v1/agent/run

```
Client
  │
  │  POST /api/v1/agent/run
  │  { "prompt": "...", "repo_path": "/projects/myapp",
  │    "agent_type": "code", "model": "qwen2.5-coder:7b" }
  │
  ▼
FastAPI Router
  │  Pydantic validates request body
  │  PathValidator checks repo_path is within allowed directories
  │
  ▼
Orchestrator.run_agent()
  │
  ├─ 1. InputSanitizer.sanitize(prompt) — strip injection attempts
  │
  ├─ 2. ModelService.resolve(agent_type, requested_model)
  │       → returns "qwen2.5-coder:7b" (or fallback)
  │
  ├─ 3. ContextAssembler.assemble(repo_path, prompt, mode=BALANCED)
  │       │
  │       ├─ GraphifyParser.parse(repo_path)
  │       │     → module graph, dependency edges, complexity scores
  │       │
  │       ├─ SkillfileClient.get_skill("code")
  │       │     → structured best-practice instructions
  │       │
  │       ├─ FileScanner.scan(repo_path)
  │       │     → ranked list of relevant source files
  │       │
  │       └─ TokenBudgetManager.allocate()
  │             → AssembledContext (sections, token_counts, metadata)
  │
  ├─ 4. AgentFactory.get_agent("code") → CodeAgent (cached)
  │
  ├─ 5. CodeAgent.run(AgentTask)
  │       │
  │       ├─ Build full prompt: system_prompt + context + user_prompt
  │       │
  │       ├─ OllamaClient.generate(model, prompt)
  │       │     │
  │       │     ├─ CircuitBreaker checks state → CLOSED, proceed
  │       │     ├─ httpx POST /api/generate
  │       │     ├─ Response validated and sanitized
  │       │     └─ Returns response text
  │       │
  │       ├─ If tools called: FileReader / FileWriter / TestRunner
  │       │     → each tool result appended to agent step log
  │       │
  │       └─ Returns AgentResult(status=COMPLETED, response=..., steps=[...])
  │
  ├─ 6. MetricsTracker records: tokens, duration, model, success
  │
  ├─ 7. ArtifactStore.save_run(run_data) → run_id
  │
  ├─ 8. ExecutionMemory.record(run_id, workflow, model, success, duration_ms)
  │
  └─ 9. Return AgentRunResponse to client
           { "run_id": "...", "response": "...", "status": "completed",
             "duration_ms": 1423, "tokens_approx": 892 }
```

---

## Context Assembly Data Flow

```
repo_path = "/projects/myapp"
prompt    = "Why does auth.py crash on concurrent login?"

                         ContextAssembler.assemble()
                                    │
           ┌───────────────────────┼────────────────────────┐
           │                       │                        │
           ▼                       ▼                        ▼
   GraphifyParser            SkillfileClient         FileScanner
   ─────────────            ───────────────         ───────────
   parse(repo_path)         get_skill("code")       scan(repo_path, prompt)
        │                        │                       │
        ▼                        ▼                       ▼
   {                        "You are an         [ "app/auth.py",
     nodes: [module...],     expert Python        "tests/test_auth.py",
     edges: [dep...],        engineer.            "app/core/db.py",
     complexity: {...},      Follow TDD..."       "app/models/user.py" ]
     entry_points: [...]   }
   }
        │                        │                       │
        ▼                        ▼                       ▼
   Serialize to text       Prepend as CRITICAL     Read file contents
   "Module graph:          priority context        within token budget
    auth.py depends on
    db.py, session.py..."

           └───────────────────────┬────────────────────────┘
                                   ▼
                          TokenBudgetManager
                          ─────────────────
                          budget: 8192 tokens
                          allocated:
                            CRITICAL (skill): 800 tokens
                            HIGH (graph):    1200 tokens
                            HIGH (auth.py):   400 tokens
                            MEDIUM (deps):    600 tokens
                            LOW (tests):      900 tokens
                            user_prompt:      150 tokens
                          total: 4050 / 8192 tokens used
                                   │
                                   ▼
                          AssembledContext
                          {
                            sections: [skill, graph, auth.py, deps, tests],
                            total_tokens: 4050,
                            sources_used: [SKILLFILE_SKILL, GRAPHIFY_GRAPH,
                                           CODE_FILE, CODE_FILE, TEST_FILE],
                            sources_dropped: [],
                            cache_key: "sha256:..."
                          }
```

---

## How Graphify Context Enriches Prompts

Without Graphify, the model sees individual files in isolation. With Graphify:

1. **Module relationships**: the graph shows that `auth.py` imports from `db.py` and `session.py`, so the model knows to look at connection pool exhaustion as a concurrent-login failure mode
2. **Complexity hotspots**: modules above a cyclomatic complexity threshold are flagged; the model prioritizes reviewing them
3. **Entry points**: the graph identifies HTTP route handlers vs. internal utilities, so the model understands call paths
4. **Dependency health**: outdated or CVE-flagged packages surface in the graph and are automatically included in security context

The graph data is serialized as structured text (not raw JSON) and injected at HIGH priority, ensuring it survives token budget compression.

---

## Artifact Storage

Every run produces a directory under `data/artifacts/runs/`:

```
data/
└── artifacts/
    └── runs/
        ├── 20260514_142301_a3f9b1c2/    ← production run
        │   ├── run.json
        │   ├── prompt.md
        │   ├── response.md
        │   ├── context.json
        │   ├── logs.txt
        │   └── test_results.json
        └── 20260514_150912_d8e2f4a1/    ← another run
            └── ...
```

The `GET /api/v1/artifacts/{run_id}` endpoint reads from this directory structure and returns the full run record. Artifacts are never deleted automatically; operators can prune `data/artifacts/runs/` with standard filesystem tools.

---

## Design Decisions

### Why async throughout?

Ollama model inference is I/O-bound and slow (1–30 seconds per call). Blocking synchronous code would pin a thread per request, limiting throughput to the number of available threads. Async allows many concurrent in-flight requests against a single Ollama instance, with the event loop efficiently interleaving waiting calls.

### Why circuit breakers?

Ollama runs locally but can become unavailable (model loading, resource pressure, restart). Without a circuit breaker, every request during an outage would wait for the full timeout (30s by default), exhausting the connection pool and causing cascading latency. The circuit breaker:

- Detects failures after 5 consecutive errors
- Opens immediately, rejecting requests with a fast error
- Half-opens after the cooldown period to test recovery
- Closes again once a successful call completes

This keeps the API responsive even when the LLM backend is temporarily down.

### Why knowledge graphs?

Code is a graph, not a list of files. Import relationships, call chains, and data flow patterns are structural properties that flat file reading cannot capture. By running Graphify against the repo before every significant request:

- The model understands module boundaries and dependency direction
- Refactoring suggestions respect existing architectural layers
- Security scans can trace data flow from HTTP input to database write
- Architecture reviews can identify circular dependencies and high-coupling hotspots

The graph context transforms the LLM from a file reader into a genuine architectural reasoner.
