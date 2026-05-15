# AI Engineering Intelligence OS

**Not just a coding assistant — an autonomous engineering intelligence that watches,
understands, and improves your codebase.**

---

## What It Is

The AI Engineering Orchestrator is a production-grade, locally-run AI system that
combines multi-agent reasoning, a structural code knowledge graph, and a skill
catalog to form a persistent engineering intelligence layer on top of any Python
codebase. Unlike a chat-based coding assistant, the orchestrator maintains execution
memory, tracks patterns across runs, routes tasks to the most capable model for each
job, and stores every analysis and generated artifact for later retrieval. It is built
to run entirely on-premise — no API keys required, no code leaves your machine.

**Capabilities:**

- Static code analysis via AST tools and graph-aware context (Graphify integration)
- Multi-agent routing: six specialized agents cover debugging, architecture,
  testing, code generation, performance, and documentation
- Intelligent model selection: routes each task to the best locally available Ollama
  model based on capability tiers and past performance
- Streaming and non-streaming response modes
- Persistent artifact store with per-execution run metadata
- Execution memory and pattern store that improve routing decisions over time
- Kubernetes-style liveness and readiness probes for deployment
- 858 tests, 85.14% coverage, fully typed Python 3.11/3.12

---

## Architecture

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    HTTP API (port 8008)              │
                         │          FastAPI · uvicorn · CORS · Prometheus       │
                         └────────────────────┬────────────────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────────────────┐
                         │                  Orchestrator                        │
                         │   - Request validation & routing                     │
                         │   - Multi-agent workflow coordination                │
                         │   - Retry / circuit-breaker logic                    │
                         │   - Streaming & non-streaming execution              │
                         └──────────┬──────────────────────┬───────────────────┘
                                    │                      │
              ┌─────────────────────▼──────┐   ┌──────────▼───────────────────┐
              │          Agents             │   │       Context Service         │
              │  - CodeAgent               │   │  - Graphify parser           │
              │  - ArchitectureAgent       │   │  - Skillfile client          │
              │  - TestingAgent            │   │  - File reader / searcher    │
              │  - DocumentationAgent      │   │  - AST analyzer              │
              │  - PerformanceAgent        │   │  - Dependency analyzer       │
              │  - SecurityAgent           │   │  - Security scanner          │
              └─────────────┬──────────────┘   └──────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       Model Service         │
              │  - Task-based routing       │
              │  - Tier selection           │
              │  - Fallback chains          │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     Ollama (local LLMs)     │
              │  HTTP · streaming · retry   │
              └────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                            Memory Layer                                  │
  │    Execution Memory · Pattern Store · Memory Store · Cache (Redis/mem)   │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           Artifact Store                                 │
  │          data/artifacts/runs/<execution_id>/{run.json, response.md}      │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Requirements

- Python 3.11 or 3.12 (Python 3.13 is not yet supported)
- [Ollama](https://ollama.com) installed and running locally
- `pip install -e .` (editable install, installs all runtime dependencies)

```bash
# 1. Clone and install
git clone <repo-url> ai-engineering-orchestrator
cd ai-engineering-orchestrator
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Pull a model
ollama pull deepseek-r1:7b

# 3. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload

# 4. Verify
curl http://localhost:8008/health
```

The server is ready when it prints:

```
INFO  Ollama is connected
INFO  Application startup complete
INFO  Uvicorn running on http://0.0.0.0:8008
```

Interactive Swagger docs are available at http://localhost:8008/docs while
`debug: true` (the default in development).

---

## Ollama Setup

Ollama is the only inference backend. Models are pulled on demand and cached
locally; nothing is sent to external APIs.

```bash
# Install Ollama (macOS)
brew install ollama
# Or download from https://ollama.com

# Start the Ollama daemon (if not running as a system service)
ollama serve

# Pull models — the orchestrator selects the best one per task type
ollama pull deepseek-r1:7b        # recommended: strong reasoning, 8 GB VRAM
ollama pull qwen2.5-coder:7b      # code generation and review (fast)
ollama pull qwen2.5-coder:14b     # code review and architecture (powerful)
ollama pull llama3.1:8b           # documentation and general Q&A
ollama pull deepseek-coder:6.7b   # code generation fallback
```

**Available model summary:**

| Model | Size | Best For | Context Window |
|-------|------|----------|----------------|
| `deepseek-r1:7b` | ~4.7 GB | Debugging, reasoning, general analysis | 8192 |
| `qwen2.5-coder:7b` | ~4.5 GB | Code generation, review, testing | 32768 |
| `qwen2.5-coder:14b` | ~9 GB | Architecture analysis, deep code review | 32768 |
| `llama3.1:8b` | ~4.9 GB | Documentation, reports, general Q&A | 128000 |
| `deepseek-coder:6.7b` | ~3.8 GB | Code generation fallback | 16384 |

You do not need all models. One model is enough to start. The orchestrator falls
back through the configured fallback chain if the preferred model is unavailable.

Default task routing (configured in `config/models/default.yaml`):

| Task Type | Default Model |
|-----------|---------------|
| `code_generation` | `qwen2.5-coder:7b` |
| `code_review` | `qwen2.5-coder:14b` |
| `debugging` | `qwen2.5-coder:14b` |
| `architecture_analysis` | `qwen2.5-coder:14b` |
| `testing` | `qwen2.5-coder:7b` |
| `documentation` | `llama3.1:8b` |
| `general_qa` | `llama3.1:8b` |
| `report` | `llama3.1:8b` |

Override the model for any individual request by setting `"model"` in the
request body.

---

## Graphify Setup

Graphify performs static AST extraction and builds a knowledge graph of your
codebase. When available, the orchestrator feeds graph data — god-nodes, community
clusters, critical paths, and cross-module dependencies — into agent prompts,
significantly improving analysis quality.

```bash
# Install the Python package
pip install graphifyy

# Install the CLI tool
graphify install

# Run on any repository (from the repo root)
graphify run .

# Output is written to graphify-out/
# graph.json       — full graph with nodes and edges
# GRAPH_REPORT.md  — human-readable summary
```

The orchestrator detects Graphify automatically at startup. If the CLI is not
found it prints a warning and continues — agents fall back to file-based context
assembly.

**Sample graph statistics for the included ShopFlow demo repo:**

- 93 nodes, 114 edges across 8 Python files
- Top god-node: `auth` module (24 connections)
- Four communities: Payment Pipeline, Authentication, Data Persistence, Entry Point

---

## API Reference

All agent endpoints are under `/api/v1`. The server also exposes root-level
health endpoints.

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Fast liveness check — returns `{"status": "ok"}` |
| `GET` | `/health/detailed` | Full health check including Ollama and model status |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/system/status` | Comprehensive status: Ollama, models, Graphify, executions |
| `GET` | `/api/v1/system/live` | Kubernetes liveness probe |
| `GET` | `/api/v1/system/ready` | Kubernetes readiness probe (returns 200 only when Ollama is healthy) |
| `GET` | `/api/v1/system/info` | Application name, version, and environment |
| `GET` | `/api/v1/system/stats` | Aggregated execution statistics |

### Models

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/models/` | List all registered models (supports `?task_type=` and `?tier=` filters) |
| `GET` | `/api/v1/models/{model_name}` | Detailed info for a specific model |
| `GET` | `/api/v1/models/health` | Health check for all registered models |
| `GET` | `/api/v1/models/stats/selection` | Model selection statistics and routing history |

### Agent

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/agent/run` | Execute a workflow, returns full response when complete |
| `POST` | `/api/v1/agent/run/stream` | Execute a workflow with token-by-token streaming |
| `GET` | `/api/v1/agent/executions` | List currently active executions |
| `GET` | `/api/v1/agent/stats` | Per-workflow execution statistics |

### Agent Run Request Body

```json
{
  "workflow_type": "debug_analysis",
  "repo_path": "/absolute/path/to/your/repo",
  "query": "Natural language description of what you want",
  "model": "deepseek-r1:7b",
  "context": {
    "focus_files": ["app/payment.py"],
    "test_files": ["tests/test_payment.py"]
  }
}
```

All fields except `workflow_type`, `repo_path`, and `query` are optional.
When `model` is omitted the orchestrator selects the best available model
for the given workflow type.

### Agent Run Response Body

```json
{
  "status": "completed",
  "execution_id": "5c9337a0-2dce-45fb-9911-f47ba640e4c4",
  "workflow_type": "debug_analysis",
  "model_used": "deepseek-r1:7b",
  "response": "## Root Cause Analysis\n\n...",
  "tokens_used": 698,
  "duration_seconds": 139,
  "artifacts": {
    "run_json": "data/artifacts/runs/<execution_id>/run.json",
    "response_md": "data/artifacts/runs/<execution_id>/response.md"
  }
}
```

---

## Workflow Types

| `workflow_type` | What the Agent Does |
|-----------------|---------------------|
| `code_generation` | Generates new code or functions from a natural language description |
| `code_review` | Reviews code for correctness, style, and potential issues |
| `code_refactoring` | Restructures existing code while preserving behavior |
| `debug_analysis` | Identifies root causes of bugs or failing tests and proposes fixes |
| `architecture_analysis` | Maps module structure, dependencies, and risk surface |
| `test_generation` | Generates pytest tests for specified modules or functions |
| `documentation` | Generates docstrings, README sections, or API documentation |
| `impact_analysis` | Estimates which parts of the codebase are affected by a proposed change |
| `general_qa` | General questions about the codebase answered with repository context |

---

## Configuration

All configuration lives in `config/`. The development defaults are:

**`config/environments/development.yaml`**

```yaml
environment: development
debug: true
host: 0.0.0.0
port: 8008
workers: 1

ollama:
  base_url: http://localhost:11434
  timeout: 120
  max_retries: 3
  default_model: qwen2.5-coder:7b

logging:
  level: INFO
  format: json

artifacts:
  base_path: data/artifacts

memory:
  base_path: data/memory

performance_db:
  base_path: data/performance_db
```

**`config/models/default.yaml`** — override task-to-model routing and the
fallback order.

Key environment variables (set in `.env` or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development`, `testing`, or `production` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `PORT` | `8008` | Server port |
| `DEBUG` | `true` | Enables /docs, verbose errors, reload |

---

## Running Tests

```bash
# Run the full test suite with coverage
pytest tests/ --cov=app

# Run a specific module
pytest tests/unit/test_orchestrator.py -v

# Run only fast unit tests (skip integration tests)
pytest tests/ -m "not integration" -v

# Generate HTML coverage report
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

Current coverage: **85.14%** across 858 tests. The minimum enforced threshold is
85% (configured in `pyproject.toml` under `--cov-fail-under=85`).

---

## Project Structure

```
ai-engineering-orchestrator/
├── app/
│   ├── main.py                     # FastAPI app factory and lifecycle events
│   ├── api/v1/
│   │   ├── router.py               # Top-level API v1 router
│   │   └── endpoints/
│   │       ├── agent.py            # /agent/run, /agent/run/stream, /agent/stats
│   │       ├── health.py           # /system/status, /live, /ready, /info
│   │       └── models.py           # /models/ listing and health
│   ├── agents/
│   │   ├── base_agent.py           # Abstract agent base class
│   │   ├── agent_factory.py        # Agent instantiation and routing
│   │   └── specialized/
│   │       ├── code_agent.py
│   │       ├── architecture_agent.py
│   │       ├── testing_agent.py
│   │       ├── documentation_agent.py
│   │       ├── performance_agent.py
│   │       └── security_agent.py
│   ├── services/
│   │   ├── orchestrator.py         # Central coordination (the conductor)
│   │   ├── context_service.py      # Context assembly from files + graph
│   │   ├── model_service.py        # Model selection and capability tiers
│   │   └── learning_service.py     # Feedback loop and pattern extraction
│   ├── integrations/
│   │   ├── ollama/                 # Ollama HTTP client + streaming
│   │   ├── graphify/               # Graph parser and client wrapper
│   │   ├── skillfile/              # Skillfile catalog client
│   │   └── ide/                    # MCP server, Windsurf adapter
│   ├── tools/
│   │   ├── file_system/            # file_reader, file_writer, file_search
│   │   ├── code_analysis/          # AST analyzer, dependency analyzer, security scanner
│   │   └── testing/                # test_runner, coverage_analyzer
│   ├── memory/
│   │   ├── execution_memory.py     # Per-session execution log
│   │   ├── pattern_store.py        # Cross-session pattern learning
│   │   └── memory_store.py         # Long-term key-value memory
│   ├── artifacts/
│   │   └── store.py                # Artifact persistence and retrieval
│   ├── workflows/
│   │   ├── debug_workflow.py
│   │   ├── refactor_workflow.py
│   │   ├── testing_workflow.py
│   │   ├── audit_workflow.py
│   │   └── report_workflow.py
│   ├── core/
│   │   ├── config/                 # Settings (pydantic-settings), models config
│   │   ├── cache/                  # In-memory and Redis cache backends
│   │   ├── monitoring/             # Structured logging, Prometheus metrics, tracing
│   │   └── security.py             # Path validation, input sanitization
│   ├── models/
│   │   ├── requests.py             # OrchestratorRequest schema
│   │   ├── responses.py            # OrchestratorResponse schema
│   │   └── schemas.py              # Shared Pydantic models
│   └── utils/
│       ├── validators.py           # Path security, input validation
│       ├── retry.py                # Async retry + circuit breaker
│       ├── formatters.py           # Response formatting helpers
│       └── async_utils.py          # Async concurrency utilities
├── config/
│   ├── environments/development.yaml
│   └── models/default.yaml
├── examples/
│   └── sample_repo/               # ShopFlow demo app with intentional bug
│       ├── app/                   # payment.py, auth.py, database.py, main.py
│       ├── tests/                 # test_payment.py (1 failing), test_auth.py
│       └── graphify-out/          # Pre-generated graph (93 nodes, 114 edges)
├── data/
│   ├── artifacts/runs/            # Per-execution run.json + response.md
│   ├── memory/                    # Persistent memory store
│   └── logs/                      # Structured JSON logs
├── docs/
│   ├── demo/end_to_end_demo.md    # Full walkthrough with real results
│   ├── api/                       # API reference docs
│   └── architecture/              # Architecture decision records
├── tests/                         # 858 tests, 85.14% coverage
├── prompts/                       # Agent system prompt templates
├── scripts/                       # Utility scripts (migrations, seed data)
└── pyproject.toml                 # Build config, dependencies, tool settings
```

---

## Roadmap

The current release handles on-demand workflows triggered by API calls. The
following capabilities are planned for future releases:

**Proactive Monitoring**
- File-system watcher that detects changes in the repository and automatically
  queues a relevant workflow (e.g., run `debug_analysis` when a test starts failing)
- Periodic graph regeneration to keep the Graphify context current without manual
  invocation

**Autonomous Detection**
- Security scanner that continuously monitors committed code for known vulnerability
  patterns and files a structured report without requiring a user prompt
- Coverage drift detection: alert when test coverage drops below a configurable
  threshold after a commit

**Multi-Agent Collaboration**
- Parallel agent execution: `architecture_analysis` and `debug_analysis` running
  simultaneously and synthesizing a combined report
- Agent-to-agent handoff: the testing agent automatically receives output from
  the debug agent to generate regression tests for the identified bug

**IDE Integration**
- MCP server (already scaffolded in `app/integrations/ide/mcp_server.py`) for
  Windsurf and other MCP-compatible editors
- In-editor overlay of graph data: hover a function to see its connectivity score
  and community membership

**Performance and Scale**
- Redis-backed execution queue for high-concurrency environments
- Distributed worker mode using Celery or Ray for long-running analysis jobs

---

## License

MIT License. See `pyproject.toml` for full license text and author information.

```
Copyright (c) 2026 Aman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
