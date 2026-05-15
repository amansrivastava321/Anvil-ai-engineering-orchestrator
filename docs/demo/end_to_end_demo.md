# End-to-End Demo: AI Engineering Orchestrator

This document walks through a complete demonstration of the AI Engineering Orchestrator
using the included ShopFlow sample repository. Every command, response, and result shown
here was produced from an actual run of the system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Starting the Server](#starting-the-server)
3. [Sample Repository Overview](#sample-repository-overview)
4. [Graphify Setup and Verification](#graphify-setup-and-verification)
5. [Workflow Demonstrations](#workflow-demonstrations)
   - [Debug Analysis](#workflow-1-debug-analysis)
   - [Test Generation](#workflow-2-test-generation)
   - [Architecture Analysis](#workflow-3-architecture-analysis)
6. [Real Results](#real-results)
7. [Artifact Verification](#artifact-verification)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Python Environment

The project requires Python 3.11 or 3.12 (Python 3.13 is not yet supported).

```bash
# Verify your Python version
python3 --version   # Should print 3.11.x or 3.12.x

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the project with dev extras
pip install -e ".[dev]"
```

### 2. Ollama Setup

Ollama provides local LLM inference. Install it from https://ollama.com then pull
at least one model before starting the server.

```bash
# Confirm Ollama is running
curl http://localhost:11434/api/tags

# Pull the recommended demo model (fits comfortably in 8 GB RAM)
ollama pull deepseek-r1:7b

# Optional — pull the full routing set used in production config
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull llama3.1:8b
ollama pull deepseek-coder:6.7b

# Verify the model is available
ollama list
```

The demo results in this document were produced with `deepseek-r1:7b`. Any model
listed above will work; reasoning quality and latency will vary.

### 3. Graphify (optional, recommended)

Graphify generates a structural knowledge graph of the codebase. The orchestrator
uses it to enrich agent context with graph-aware information (which functions are
most connected, which modules are critical paths, etc.).

```bash
pip install graphifyy
graphify install

# Verify the CLI is on PATH
graphify --version
```

If Graphify is not installed the server still starts and agents still run — they
fall back to assembling context directly from source files.

---

## Starting the Server

```bash
# From the project root (with .venv active)
uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
```

Expected startup output:

```
INFO     Application created name=AI Engineering Orchestrator version=1.0.0 environment=development
INFO     Graphify is installed
INFO     Ollama is connected
INFO     Application startup complete
INFO     Uvicorn running on http://0.0.0.0:8008
```

If you see `Graphify not installed` or `Ollama connection issue` in the log, see
the [Troubleshooting](#troubleshooting) section.

**Confirm the server is healthy:**

```bash
curl http://localhost:8008/health
# {"status": "ok", "timestamp": "2026-05-14T16:45:00.000000"}

curl http://localhost:8008/health/detailed
# Full JSON with Ollama status, model availability, and active executions
```

Interactive API docs are available at http://localhost:8008/docs when the server
is running in development mode.

---

## Sample Repository Overview

The demo uses `examples/sample_repo/` — a minimal Python e-commerce application
called **ShopFlow**. It is a self-contained Git repository with intentional bugs
and realistic test coverage so you can observe the agents catching real problems.

### File Tree

```
examples/sample_repo/
├── app/
│   ├── __init__.py
│   ├── auth.py          # JWT-style token auth (hash, verify, generate, validate)
│   ├── database.py      # In-memory store with CRUD helpers
│   ├── main.py          # HTTP server (stdlib BaseHTTPRequestHandler)
│   └── payment.py       # Order total + tax calculation — CONTAINS BUG
├── tests/
│   ├── __init__.py
│   ├── test_auth.py     # 14 tests, all passing
│   └── test_payment.py  # 10 tests, 1 FAILING (exposes the bug)
├── graphify-out/
│   ├── graph.json        # 93 nodes, 114 edges
│   └── GRAPH_REPORT.md   # Human-readable graph summary
└── requirements.txt
```

### The Bug in `app/payment.py`

The `calculate_total` function has the following signature:

```python
def calculate_total(items: list[dict], discount_percent: float | None) -> float:
```

When `discount_percent` is `None` (which is a valid caller action — "no discount"),
the function reaches:

```python
discount = subtotal * (discount_percent / 100)
```

This raises `TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`.

The failing test in `tests/test_payment.py` is `test_none_discount_treated_as_zero`,
which calls `calculate_total(ITEMS_BASIC, None)` and expects `30.00`.

### Running the Sample Repo Tests

```bash
cd examples/sample_repo
pip install -r requirements.txt
pytest tests/ -v

# Expected output:
# PASSED tests/test_payment.py::TestCalculateTotal::test_no_discount_returns_sum
# PASSED tests/test_payment.py::TestCalculateTotal::test_ten_percent_discount
# PASSED tests/test_payment.py::TestCalculateTotal::test_full_discount_returns_zero
# PASSED tests/test_payment.py::TestCalculateTotal::test_empty_items_returns_zero
# PASSED tests/test_payment.py::TestCalculateTotal::test_missing_price_key_raises
# FAILED tests/test_payment.py::TestCalculateTotal::test_none_discount_treated_as_zero
# PASSED tests/test_payment.py::TestApplyTax::test_default_rate
# ... (all other tests pass)
```

---

## Graphify Setup and Verification

Graphify performs static AST extraction and builds a knowledge graph of the codebase.
The sample repo already has a pre-generated graph in `graphify-out/`.

### Generate the Graph (or Regenerate)

```bash
cd examples/sample_repo
graphify run .

# Graphify writes output to graphify-out/
# graph.json       — full graph (nodes + edges)
# GRAPH_REPORT.md  — human-readable summary
```

### Verify the Output

```bash
# Check node and edge count
python3 -c "
import json, pathlib
g = json.loads(pathlib.Path('examples/sample_repo/graphify-out/graph.json').read_text())
print('Nodes:', len(g['nodes']))
print('Edges:', len(g['edges']))
"
# Nodes: 93
# Edges: 114
```

The graph report (`graphify-out/GRAPH_REPORT.md`) identifies the top god-nodes:

| Node | Type | Connections |
|------|------|-------------|
| `auth` | module | 24 |
| `payment` | module | 22 |
| `database` | module | 18 |
| `main` | module | 16 |
| `calculate_total()` | function | 8 |

The graph also identifies four communities — Payment Pipeline, Authentication,
Data Persistence, and Entry Point — which the architecture agent uses to reason
about module coupling and change impact.

---

## Workflow Demonstrations

The orchestrator exposes a single primary endpoint for running workflows:

```
POST /api/v1/agent/run
Content-Type: application/json
```

**Request schema:**

```json
{
  "workflow_type": "<type>",
  "repo_path": "<absolute-path-to-repo>",
  "query": "<natural-language-description>",
  "model": "<ollama-model-name>",
  "context": {
    "focus_files": ["<relative-path>"],
    "test_files": ["<relative-path>"]
  }
}
```

All examples below assume the server is running on `localhost:8008` and the sample
repo is at the path shown. Adjust to match your actual checkout location.

---

### Workflow 1: Debug Analysis

Ask the system to find the root cause of the failing test and propose a fix.

```bash
curl -X POST http://localhost:8008/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "debug_analysis",
    "repo_path": "/Users/aman/Documents/Projects/ai-engineering-orchestrator/examples/sample_repo",
    "query": "The test test_none_discount_treated_as_zero is failing. Find the root cause in app/payment.py and propose a fix.",
    "model": "deepseek-r1:7b",
    "context": {
      "focus_files": ["app/payment.py"],
      "test_files": ["tests/test_payment.py"]
    }
  }'
```

**Expected response structure:**

```json
{
  "status": "completed",
  "execution_id": "5c9337a0-2dce-45fb-9911-f47ba640e4c4",
  "workflow_type": "debug_analysis",
  "model_used": "deepseek-r1:7b",
  "response": "## Root Cause Analysis\n\nThe bug occurs because ...",
  "tokens_used": 698,
  "duration_seconds": 139,
  "artifacts": {
    "run_json": "data/artifacts/runs/5c9337a0-2dce-45fb-9911-f47ba640e4c4/run.json",
    "response_md": "data/artifacts/runs/5c9337a0-2dce-45fb-9911-f47ba640e4c4/response.md"
  }
}
```

---

### Workflow 2: Test Generation

Generate additional tests for the payment module covering edge cases.

```bash
curl -X POST http://localhost:8008/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "test_generation",
    "repo_path": "/Users/aman/Documents/Projects/ai-engineering-orchestrator/examples/sample_repo",
    "query": "Generate pytest tests for app/payment.py. Focus on: zero discount, large order values, and invalid price types.",
    "model": "deepseek-r1:7b",
    "context": {
      "focus_files": ["app/payment.py"],
      "test_files": ["tests/test_payment.py"]
    }
  }'
```

**Expected response structure:**

```json
{
  "status": "completed",
  "execution_id": "33615f5b-d4fe-4dd6-a349-615f08db8975",
  "workflow_type": "test_generation",
  "model_used": "deepseek-r1:7b",
  "response": "### Tests File Structure\n\n```python\ndef test_calculate_total_with_zero_discount(): ...",
  "duration_seconds": 91,
  "artifacts": {
    "run_json": "data/artifacts/runs/33615f5b-d4fe-4dd6-a349-615f08db8975/run.json"
  }
}
```

---

### Workflow 3: Architecture Analysis

Ask the system to describe the module structure and identify high-risk components.

```bash
curl -X POST http://localhost:8008/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "architecture_analysis",
    "repo_path": "/Users/aman/Documents/Projects/ai-engineering-orchestrator/examples/sample_repo",
    "query": "Describe the architecture of this application. Identify the most critical modules, cross-module dependencies, and any structural risks.",
    "model": "deepseek-r1:7b"
  }'
```

**Expected response structure:**

```json
{
  "status": "completed",
  "execution_id": "0b95ef68-6055-4c68-a4f5-434f65b0bbf0",
  "workflow_type": "architecture_analysis",
  "model_used": "deepseek-r1:7b",
  "response": "### System Architecture Analysis\n\nThe application is structured into three main modules ...",
  "duration_seconds": 63,
  "artifacts": {
    "run_json": "data/artifacts/runs/0b95ef68-6055-4c68-a4f5-434f65b0bbf0/run.json"
  }
}
```

---

### Streaming Variant

Any workflow can be streamed. Tokens are yielded as they are produced by the model —
useful for long-running analyses or when you want to display progress in a UI.

```bash
curl -X POST http://localhost:8008/api/v1/agent/run/stream \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "debug_analysis",
    "repo_path": "/Users/aman/Documents/Projects/ai-engineering-orchestrator/examples/sample_repo",
    "query": "Why is test_none_discount_treated_as_zero failing?",
    "model": "deepseek-r1:7b"
  }'
```

The response body is a plain text stream; each chunk is a token fragment from the model.

---

## Real Results

These are unedited outputs from actual runs on 2026-05-14.

### Debug Analysis — Actual Output

- **Execution ID:** `5c9337a0-2dce-45fb-9911-f47ba640e4c4`
- **Model:** `deepseek-r1:7b`
- **Duration:** 139 seconds
- **Status:** COMPLETED

**Root cause identified:**

> The bug occurs because `discount_percent` can be `None`, causing division by 100
> to raise a TypeError instead of treating it as 0%.
>
> When `discount_percent` is None, line `discount = subtotal * (discount_percent / 100)`
> raises a TypeError since you can't divide None by an integer.

**Proposed fix (from model output):**

Option A — change the default in the signature:

```python
def calculate_total(items: list[dict], discount_percent: float | None = 0) -> float:
```

Option B — add a guard at the top of the function body:

```python
discount_percent = discount_percent or 0
```

Both approaches were correctly identified. The model produced the full corrected
source file with the fix applied.

---

### Test Generation — Actual Output

- **Execution ID:** `33615f5b-d4fe-4dd6-a349-615f08db8975`
- **Model:** `deepseek-r1:7b`
- **Duration:** 91 seconds
- **Status:** COMPLETED

**Tests generated for:**
- Zero discount: `test_calculate_total_with_zero_discount`
- Large orders (1e6+ values): `test_calculate_total_with_large_order`
- Invalid price types (missing `price` key): `test_calculate_total_with_invalid_item`
- Full pipeline with tax: `test_process_payment_with_tax`
- Negative amount guard: `test_negative_amount_in_apply_tax`
- Parametrized None/0 discount: `@pytest.mark.parametrize`

---

### Architecture Analysis — Actual Output

- **Execution ID:** `0b95ef68-6055-4c68-a4f5-434f65b0bbf0`
- **Model:** `deepseek-r1:7b`
- **Duration:** 63 seconds
- **Status:** COMPLETED
- **Note:** Graphify CLI was not installed at the system level during this run;
  context was assembled from source files directly. When Graphify is available,
  the agent receives the full graph (93 nodes, 114 edges) with community and
  god-node data, significantly enriching the analysis.

**Key findings from model output:**

- Three main modules identified: payment pipeline, authentication, and API gateway layer
- Cross-community dependencies correctly mapped (`main.py` imports all three leaf modules)
- Risks surfaced: no centralized logging, caching absent from hot paths, MFA not
  implemented in the auth module
- Recommended adding Redis caching for frequently accessed records

---

## Artifact Verification

Every run stores two files under `data/artifacts/runs/<execution_id>/`:

| File | Contents |
|------|----------|
| `run.json` | Full metadata: run ID, workflow type, model, timestamps, raw response |
| `response.md` | Response content formatted as Markdown for human reading |

**List all stored runs:**

```bash
ls data/artifacts/runs/
# 0b95ef68-6055-4c68-a4f5-434f65b0bbf0
# 16f2d24c-0d6e-4b67-a754-8bc4b35a1b0e
# 33615f5b-d4fe-4dd6-a349-615f08db8975
# 5c9337a0-2dce-45fb-9911-f47ba640e4c4
```

**Inspect a specific run:**

```bash
# Pretty-print the metadata
python3 -m json.tool data/artifacts/runs/5c9337a0-2dce-45fb-9911-f47ba640e4c4/run.json

# Quick summary of all runs
for d in data/artifacts/runs/*/; do
  python3 -c "
import json, pathlib, sys
d = json.loads((pathlib.Path('$d') / 'run.json').read_text())
print(d['run_id'][:8], d['workflow_type'], d['model_used'], d['timestamp'][:16])
"
done
```

**Query execution stats via the API:**

```bash
curl http://localhost:8008/api/v1/agent/stats
curl http://localhost:8008/api/v1/system/stats
```

---

## Troubleshooting

### Server will not start — `ModuleNotFoundError`

You likely need to install the package in editable mode:

```bash
pip install -e ".[dev]"
```

### `Ollama connection issue` at startup

Confirm Ollama is running as a background process:

```bash
ollama serve &
curl http://localhost:11434/api/tags
```

If Ollama is on a non-default port, update `config/environments/development.yaml`:

```yaml
ollama:
  base_url: http://localhost:<your-port>
```

### `Model not found` error when running a workflow

The model name in your request must exactly match a model that Ollama has pulled:

```bash
ollama list                  # see what is available
ollama pull deepseek-r1:7b   # pull the demo model if missing
```

### `Graphify not installed` warning at startup

This is non-fatal. The server runs without Graphify; agents assemble context from
source files instead of graph data. To enable graph-aware context:

```bash
pip install graphifyy
graphify install
# Then restart the server
```

### Agent returns a 400 error — `repo_path not found`

The `repo_path` field must be an absolute filesystem path that the server process
can read. Use the full path:

```bash
# Wrong
"repo_path": "examples/sample_repo"

# Correct
"repo_path": "/Users/aman/Documents/Projects/ai-engineering-orchestrator/examples/sample_repo"
```

### Response is slow (> 3 minutes)

Inference time depends heavily on model size, hardware, and prompt length.
With `deepseek-r1:7b` on a MacBook Pro M2:

- Debug analysis: ~140 seconds
- Test generation: ~90 seconds
- Architecture analysis: ~60 seconds

If you need faster responses, switch to `qwen2.5-coder:7b` or `deepseek-coder:6.7b`.
Both are smaller and faster, with somewhat lower reasoning depth.

### Checking server logs

The server logs to stdout in JSON format. Pipe through `jq` for readability:

```bash
uvicorn app.main:app --port 8008 2>&1 | jq '.'
```

Alternatively, logs are also written to `logs/` in the project root.
