# Anvil — Autonomous Engineering Intelligence

**Not just a coding assistant — an autonomous engineering intelligence that watches,
understands, and improves your codebase.**

---

## What It Is

Anvil is a production-grade, locally-run AI system that combines a self-governing
AI organization, semantic code search, and a structural knowledge graph to form a
persistent engineering intelligence layer on top of any Python codebase. Unlike a
chat-based coding assistant, Anvil maintains execution memory, tracks patterns across
runs, routes tasks to the most capable model for each job, and stores every analysis
and generated artifact for later retrieval. It is built to run entirely on-premise —
no API keys required, no code leaves your machine.

**Capabilities:**

- Self-governing AI organization: CEO AI, 6-member Council, Synthesizer, and Intuition Engine
- Static code analysis via AST tools and graph-aware context (Graphify integration)
- Semantic code search via bge-m3 embeddings — finds relevant functions by meaning, not just name
- Multi-agent routing: six specialized agents cover debugging, architecture,
  testing, code generation, performance, and documentation
- Intelligent model selection: routes each task to the best locally available Ollama
  model based on capability tiers and past performance
- Streaming and non-streaming response modes
- Persistent artifact store with per-execution run metadata
- Execution memory and pattern store that improve routing decisions over time
- Weekly self-improvement cycles via evolution service
- Continuous repo monitoring via proactive service
- Kubernetes-style liveness and readiness probes for deployment
- 1,103 tests, 85.14% coverage, fully typed Python 3.11/3.12

---

## Architecture

### AI Organization Layer

Anvil uses a self-governing AI organization, not a fixed pipeline:

- **CEO AI** — Receives every request, decides operating mode, learns from outcomes
- **Mode Selector** — AI decides: Mode 1 (alone), Mode 2 (consult experts), Mode 3 (full council)
- **AI Council** — 6 specialized agents (Architect, Security, Performance, Testing, Memory, Domain) debate, vote, and synthesize solutions
- **Synthesizer** — Combines all proposals into one unified plan, resolves conflicts
- **Intuition Engine** — Discovers patterns from execution history using bge-m3 semantic matching

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    HTTP API (port 8008)              │
                         │          FastAPI · uvicorn · CORS · Prometheus       │
                         └────────────────────┬────────────────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────────────────┐
                         │                   CEO AI                             │
                         │   - Analyzes request + repo patterns                 │
                         │   - Selects operating mode (1 / 2 / 3)              │
                         │   - Routes to Council or acts directly               │
                         │   - Records outcomes, develops intuition             │
                         └──────────┬──────────────────────┬───────────────────┘
                                    │                      │
              ┌─────────────────────▼──────┐   ┌──────────▼───────────────────┐
              │         AI Council          │   │       Context Service         │
              │  - Architect               │   │  - Graphify parser           │
              │  - Security                │   │  - bge-m3 semantic search    │
              │  - Performance             │   │  - File reader / searcher    │
              │  - Testing                 │   │  - AST analyzer              │
              │  - Memory                  │   │  - Dependency analyzer       │
              │  - Domain Expert           │   │  - Security scanner          │
              └─────────────┬──────────────┘   └──────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │        Synthesizer          │
              │  - Merges all proposals     │
              │  - Resolves conflicts       │
              │  - Produces unified plan    │
              └─────────────┬──────────────┘
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

## Interfaces

Anvil is accessible through three interfaces:

| Interface | How to Access | Best For |
|-----------|---------------|----------|
| **Web Dashboard** | `http://localhost:8008/dashboard` | Drag & drop folders, health reports |
| **CLI Tool** | `anvil` command in terminal | Quick requests, interactive folder picker |
| **VSCode Extension** | Install from `vscode-extension/` | Inline annotations, side panel, CodeLens |

---

## How It Works

Anvil is not a chatbot. It's a self-governing AI engineering organization:

1. **CEO AI** receives your request and decides how to handle it
2. **Mode Selection** — AI decides: act alone, consult experts, or convene full council
3. **AI Council** — 6 specialized agents debate, vote, and synthesize the best approach
4. **Execution** — The chosen plan runs with full tool access (Graphify, file I/O, test runner)
5. **Learning** — Every decision recorded. Patterns discovered. Performance improves weekly.

---

## Quick Start

### Requirements

- Python 3.11 or 3.12 (Python 3.13 is not yet supported)
- [Ollama](https://ollama.com) installed and running locally
- `pip install -e .` (editable install, installs all runtime dependencies)

```bash
# 1. Clone and install
git clone <repo-url> anvil
cd anvil
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Pull a model
ollama pull dolphin-mistral:7b

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

# Pull models — Anvil selects the best one per task type
ollama pull dolphin-mistral:7b     # Best all-around model (start here)
ollama pull deepseek-r1:7b         # Reasoning and debugging
ollama pull dolphincoder:7b        # Code generation (fine-tuned)
ollama pull qwen3.5:9b             # Pattern discovery and analysis
ollama pull gemma4:e4b             # Architecture and complex tasks
ollama pull phi4-mini:latest       # Fast mode selection
ollama pull bge-m3:latest          # Semantic code search (embeddings, not chat)
```

**Available model summary:**

| Model | Size | Best For |
|-------|------|----------|
| `dolphin-mistral:7b` | 4.1 GB | General reasoning, council proposals, documentation |
| `deepseek-r1:7b` | 4.7 GB | Debugging, conflict resolution, synthesis |
| `dolphincoder:7b` | 4.2 GB | Code generation (fine-tuned for writing code) |
| `qwen3.5:9b` | 6.6 GB | Pattern discovery, analytical reflection |
| `gemma4:e4b` | 9.6 GB | Architecture analysis, complex multi-system tasks |
| `phi4-mini:latest` | 2.5 GB | Mode selection, simple classification (fast) |
| `bge-m3:latest` | 1.2 GB | Semantic code search via embeddings (not for chat) |
| `qwen2.5vl:7b` | 6.0 GB | Vision/screenshot analysis only (not for text tasks) |

You do not need all models. One model is enough to start.

Default task routing (configured in `config/models/default.yaml`):

| Task Type | Default Model | Why |
|-----------|---------------|-----|
| `code_generation` | `dolphincoder:7b` | Fine-tuned specifically for writing code |
| `code_review` | `dolphin-mistral:7b` | Best instruction-following for review |
| `debugging` | `deepseek-r1:7b` | Actual reasoning over root causes |
| `architecture_analysis` | `gemma4:e4b` | Largest model for complex structural analysis |
| `testing` | `dolphincoder:7b` | Code generation for test writing |
| `documentation` | `dolphin-mistral:7b` | General writing and explanation |
| `general_qa` | `dolphin-mistral:7b` | Best all-around model |

Override the model for any individual request by setting `"model"` in the
request body.

---

## Cloud Models

Anvil supports cloud-hosted models from OpenAI, Anthropic, Google, and OpenRouter
alongside local Ollama models. No config files needed — just export an API key.

### Enabling cloud providers

```bash
# OpenAI — gpt-4o, gpt-4o-mini, o1, o1-mini
export OPENAI_API_KEY=sk-...

# Anthropic — claude-3.5-sonnet, claude-3-opus, claude-3-haiku
export ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini — gemini-2.0-flash, gemini-1.5-pro
export GOOGLE_API_KEY=AIza...

# OpenRouter — 200+ models through a single key
# (openai/gpt-4o, anthropic/claude-3.5-sonnet, deepseek/deepseek-r1, …)
export OPENROUTER_API_KEY=sk-or-...
```

Restart Anvil and the models appear automatically:

```bash
curl http://localhost:8008/api/v1/models/
# {
#   "local":  [{"name": "dolphin-mistral:7b", "provider": "ollama", ...}],
#   "cloud":  [{"name": "gpt-4o", "provider": "openai", ...},
#              {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic", ...}],
#   "total": N
# }
```

### Using a cloud model in a request

```bash
curl -X POST http://localhost:8008/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/your/repo",
    "query":    "Review the auth module for security issues",
    "model":    "gpt-4o"
  }'
```

Anvil routes the request to OpenAI automatically. The CEO and Council work the
same way regardless of whether the model is local or cloud-hosted.

### Behaviour without API keys

If no API keys are set, Anvil works exactly as before — only local models appear
in the model list, no errors, no warnings. Cloud support is invisible until configured.

### Provider summary

| Provider | Env var | Models |
|----------|---------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, o1, o1-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | `GOOGLE_API_KEY` | gemini-2.0-flash, gemini-1.5-pro |
| OpenRouter | `OPENROUTER_API_KEY` | 200+ models via one key |

**OpenRouter is the easiest path.** One key gives access to every major model
from every provider — no separate accounts needed.

---

## Free Cloud Models (Automatic)

When `OPENROUTER_API_KEY` is set, Anvil automatically promotes five completely
free OpenRouter models to the front of its routing chains. No credits, no billing.

| Model | Params | Primary Role |
|-------|--------|--------------|
| `deepseek/deepseek-v4-flash:free` | 284B | Code generation, debugging, architecture |
| `openai/gpt-oss-120b:free` | 120B | Reasoning, security audit, pattern discovery |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | CEO reasoning, synthesis, council proposals |
| `google/gemma-3-27b-it:free` | 27B | Council voting, performance analysis |
| `mistralai/mistral-small-3.2-24b-instruct:free` | 24B | Mode selection, council critique |

### How It Works

1. **Try free cloud first** — each AI component calls its assigned free OpenRouter model
2. **Automatic fallback** — if a cloud call fails, the next model in the chain is tried
3. **Local always works** — the local Ollama model is the guaranteed last resort in every chain

### Setup

```bash
export OPENROUTER_API_KEY=sk-or-...   # free account at openrouter.ai
uvicorn app.main:app --port 8008
```

That's it. Anvil routes internally — no other changes needed.

### Offline Mode

No key set? Anvil works identically with local Ollama models. Remove the env var
at any time to return to fully local inference with no errors or restarts required.

---

## Adding New Models

Anvil auto-discovers any model available in Ollama. No config changes needed.

### Local Models

```bash
ollama pull <model-name>
# Restart Anvil — model auto-discovered
uvicorn app.main:app --port 8008
```

### Cloud Models (Optional)

Set the API key for any provider. Anvil auto-detects and makes models available:

```bash
export OPENAI_API_KEY=sk-...           # gpt-4o, o1, gpt-4o-mini
export ANTHROPIC_API_KEY=sk-ant-...    # claude-3.5-sonnet, claude-3-opus
export GOOGLE_API_KEY=...              # gemini-2.0-flash, gemini-1.5-pro
export OPENROUTER_API_KEY=sk-or-...    # 200+ models via single API key
```

Restart Anvil. Cloud models appear alongside local models automatically.
No API keys set? Anvil works with local models only. No errors.

### Verify

```bash
curl http://localhost:8008/api/v1/models/
```

Returns all available models — both local and cloud.

---

## Graphify Setup

Graphify performs static AST extraction and builds a knowledge graph of your
codebase. When available, Anvil feeds graph data — god-nodes, community
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

Anvil detects Graphify automatically at startup. If the CLI is not
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
When `model` is omitted Anvil selects the best available model
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
  default_model: dolphin-mistral:7b

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

Current coverage: **85.14%** across 1,103 tests. The minimum enforced threshold is
85% (configured in `pyproject.toml` under `--cov-fail-under=85`).

---

## Project Structure

```
anvil/
├── app/
│   ├── main.py                     # FastAPI app factory and lifecycle events
│   ├── api/v1/
│   │   ├── router.py               # Top-level API v1 router
│   │   └── endpoints/
│   │       ├── agent.py            # /agent/run, /agent/run/stream, /agent/stats
│   │       ├── health.py           # /system/status, /live, /ready, /info
│   │       └── models.py           # /models/ listing and health
│   ├── ai/                          # AI Organization — CEO, Council, Synthesizer
│   │   ├── ceo.py                   # CEO decides mode, learns, develops intuition
│   │   ├── council.py               # Council debate and voting system
│   │   ├── council_members.py       # 6 specialized AI experts
│   │   ├── synthesizer.py           # Combines proposals into unified plans
│   │   ├── mode_selector.py         # AI-driven mode selection
│   │   ├── intuition.py             # Pattern discovery via bge-m3
│   │   ├── relevance.py             # Finds relevant files for any task
│   │   ├── code_context_builder.py  # Builds focused 1,200-token context
│   │   ├── context_budget.py        # Hard token limits per category
│   │   └── model_routing.py         # Routes tasks to optimal models
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
│   │   ├── learning_service.py     # Feedback loop and pattern extraction
│   │   ├── cloud_registry.py       # Auto-discovers cloud providers
│   │   ├── onboarding_service.py   # Automatic repo analysis pipeline
│   │   ├── evolution_service.py    # Weekly self-improvement cycles
│   │   └── proactive_service.py    # Continuous repo monitoring
│   ├── integrations/
│   │   ├── ollama/                 # Ollama HTTP client + streaming
│   │   ├── graphify/               # Graph parser and client wrapper
│   │   ├── skillfile/              # Skillfile catalog client
│   │   ├── ide/                    # MCP server, Windsurf adapter
│   │   ├── cloud/                  # Cloud model providers (OpenAI, Anthropic, etc.)
│   │   └── code_indexer.py         # bge-m3 semantic code index
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
│   │   ├── repo_state.py           # Repository memory and health tracking
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
├── cli/
│   └── ae.py                       # CLI tool with interactive folder picker
├── vscode-extension/               # VSCode extension for Anvil
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
│   ├── code_indexes/              # bge-m3 function embeddings
│   ├── repo_states/               # Per-repo health history
│   ├── ai_decisions/              # CEO decision records and patterns
│   └── logs/                      # Structured JSON logs
├── docs/
│   ├── demo/end_to_end_demo.md    # Full walkthrough with real results
│   ├── api/                       # API reference docs
│   └── architecture/              # Architecture decision records
├── tests/                         # 1,103 tests, 85.14% coverage
├── prompts/                       # Agent system prompt templates
├── scripts/                       # Utility scripts (migrations, seed data)
└── pyproject.toml                 # Build config, dependencies, tool settings
```

---

## Roadmap

Anvil's core organization (CEO, Council, Synthesizer, Intuition Engine), proactive
monitoring, and multi-agent collaboration are all production-ready. Planned future
work:

**IDE Integration**
- Full MCP server (scaffolded in `app/integrations/ide/mcp_server.py`) for
  Windsurf and other MCP-compatible editors
- In-editor graph overlay: hover a function to see its connectivity score
  and community membership

**Performance and Scale**
- Redis-backed execution queue for high-concurrency environments
- Distributed worker mode using Celery or Ray for long-running analysis jobs

**Extended Model Support**
- Automatic benchmark harness that continuously measures model quality per task
  type and updates routing weights without human intervention

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
