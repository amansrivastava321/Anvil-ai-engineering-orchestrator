# Memory and Learning — AI Engineering Orchestrator

## Overview

The orchestrator maintains five distinct types of memory, each serving a different cognitive role in the system's reasoning and improvement over time. Understanding which type of memory serves which purpose is essential to understanding how the system learns — and what it cannot yet do.

```
Memory Architecture
─────────────────────────────────────────────────────────────────

Type              Location                   Persistence    Scope
─────────────────────────────────────────────────────────────────
Structural        Graphify (graph)           Process-scoped Codebase
Episodic          ExecutionMemory (JSON)     Permanent      All runs
Semantic          PatternStore (JSON)        Permanent      Task types
Procedural        Skillfile skills           External       Domain
Evolutionary      LearningService (future)   Not built yet  System

─────────────────────────────────────────────────────────────────
```

---

## Memory Type 1: Structural Memory (Graphify)

**What it is**: The knowledge graph of the codebase — modules, dependencies, call paths, complexity scores.

**Where it lives**: Graphify extracts this per-request by running its CLI against the repo. There is no persistent structural memory; the graph is regenerated each time it is needed.

**How it's used**: `GraphifyParser` converts the graph output into text blocks that `ContextAssembler` injects at HIGH priority. The agent sees the codebase as a connected graph, not a bag of files.

**What it knows**:
- Which modules import which other modules
- Which functions call which other functions
- Cyclomatic complexity per module
- Entry points (HTTP handlers, CLI entrypoints, cron jobs)
- Test coverage per module (if a coverage report is available)

**Limitation**: The graph is ephemeral — once the request completes, the graph data is discarded. There is no cross-run accumulation of structural knowledge. If the same file is analyzed in 100 runs, the 100th run has no memory of the previous 99 graph states.

---

## Memory Type 2: Episodic Memory (ExecutionMemory)

**What it is**: A timestamped log of every agent run — which model was used, whether it succeeded, how long it took.

**Where it lives**: `data/memory/executions.json` — a plain JSON file, capped at the 1000 most recent entries.

**Implementation** (`app/memory/execution_memory.py`):

```python
class ExecutionMemory:
    def record(
        self,
        run_id: str,
        workflow: str,     # "audit", "debug", "direct_agent"
        model: str,        # "qwen2.5-coder:7b"
        task_type: str,    # "code_generation", "security_scan"
        success: bool,
        duration_ms: float,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None: ...
```

Each entry written:

```json
{
  "run_id": "20260514_142301_a3f9b1c2",
  "workflow": "debug",
  "model": "qwen2.5-coder:7b",
  "task_type": "debugging",
  "success": true,
  "duration_ms": 4312.7,
  "error": null,
  "metadata": { "agent": "code", "tokens": 1203 },
  "recorded_at": "2026-05-14T14:23:01.412Z"
}
```

**Query capabilities**:

```python
# All runs in the last 20 executions
memory.get_recent(limit=20)

# Success rate and average latency for a specific model
memory.get_model_stats("qwen2.5-coder:7b")
# → { "model": "...", "runs": 47, "success_rate": 0.915, "avg_duration_ms": 3812.4 }

# Stats for a specific workflow
memory.get_workflow_stats("audit")
```

**What it enables today**: After 50+ runs, an operator can query `get_model_stats()` to see which models are reliable and which are slow. This data is surfaced by `ModelService` but is not yet used to automatically switch models.

**What's missing for true learning**: ExecutionMemory records that a run succeeded or failed, but not *why*. There is no semantic understanding of what made the run successful — no annotation of the prompt, no comparison of response quality, no connection between execution outcomes and code quality improvements.

---

## Memory Type 3: Semantic Memory (PatternStore)

**What it is**: A collection of named patterns accumulated from execution history. Each pattern is a `(type, key)` pair with a list of observed outcomes.

**Where it lives**: `data/memory/patterns.json`

**Implementation** (`app/memory/pattern_store.py`):

```python
class PatternStore:
    def record_pattern(
        self,
        pattern_type: str,  # e.g., "fix_strategy", "model_preference", "error_class"
        pattern_key: str,   # e.g., "null_pointer_in_async", "qwen2.5-coder"
        context: dict,
        outcome: str,       # e.g., "success", "failure", "partial"
    ) -> None: ...

    def get_patterns(self, pattern_type: str | None = None) -> list[dict]: ...

    def get_best_model_for_task(self, task_type: str) -> str | None: ...
```

Stored pattern structure:

```json
{
  "fix_strategy:null_pointer_in_async": {
    "type": "fix_strategy",
    "key": "null_pointer_in_async",
    "outcomes": [
      {
        "outcome": "success",
        "context": { "model": "qwen2.5-coder:7b", "tokens": 892 },
        "recorded_at": "2026-05-12T09:14:22.000Z"
      }
    ],
    "first_seen": "2026-05-10T...",
    "last_seen": "2026-05-14T...",
    "count": 7
  }
}
```

**Current state of `get_best_model_for_task`**: The method exists but returns `None` — it is a documented placeholder:

```python
def get_best_model_for_task(self, task_type: str) -> str | None:
    """Return the model with the best success rate for a task type."""
    # Placeholder: real impl would query execution_memory
    return None
```

This is the primary gap between the current system and a genuinely learning system. The data exists in `ExecutionMemory`; the query logic to cross-reference it with patterns has not been implemented.

---

## Memory Type 4: Procedural Memory (Skillfile)

**What it is**: Structured, externally managed engineering skills that encode best practices, task-specific instructions, and domain knowledge in a form that small local models can reliably follow.

**Where it lives**: The Skillfile service (external process, accessed via `app/integrations/skillfile/client.py`)

**How it's used**: Before every agent run, `ContextAssembler` fetches the relevant skill for the agent type and injects it at CRITICAL priority — meaning it will always be included regardless of token budget pressure.

**Why this matters for small models**: A 7B parameter model does not have the same embedded knowledge as a 70B model. Skillfile compensates by explicitly providing structured instructions that the model follows. Instead of hoping the model knows pytest parametrize conventions, the skill tells it explicitly.

**Difference from other memory types**: Procedural memory is not learned from runs — it is authored by human experts and versioned externally. It is static guidance, not dynamic learning.

---

## Memory Type 5: Evolutionary Memory (LearningService — Future)

**What it is**: A planned service that would close the feedback loop between execution outcomes and strategy selection — the component that would make the system genuinely self-improving.

**Current status**: Not implemented. The architecture reserves a place for it (the `data/memory/` directory is designed to support it), but no `LearningService` class exists in the codebase today.

**Planned design**:

```python
class LearningService:
    """
    Analyzes execution history to optimize strategy selection.
    
    Planned capabilities:
    - Cross-reference ExecutionMemory outcomes with PatternStore patterns
    - Identify which model + prompt combination achieves highest success rate
      per task_type
    - Surface strategy recommendations to ModelService
    - Run weekly optimization cycles (see autonomous_audit_roadmap.md)
    """
    
    async def analyze_model_performance(self) -> dict[str, str]:
        """Return {task_type: best_model} based on execution history."""
        ...
    
    async def record_strategy_outcome(
        self,
        strategy_id: str,
        task_type: str,
        outcome: str,
        quality_score: float,
    ) -> None:
        """Record the outcome of a specific strategy attempt."""
        ...
    
    async def get_recommended_strategy(self, task_type: str) -> StrategyConfig:
        """Return the current best strategy for this task type."""
        ...
```

---

## ArtifactStore: Run-Level Persistence

The `ArtifactStore` is distinct from the memory layer — it persists the complete record of each run to the filesystem, not just summary statistics.

**Location**: `data/artifacts/runs/{run_id}/`

**Files written per run** (`app/artifacts/store.py`):

```
run_id/
├── run.json          All run metadata (model, duration, status, agent, workflow)
├── prompt.md         The exact prompt sent to the model
├── response.md       The model's raw response
├── context.json      The full AssembledContext that was used
├── logs.txt          Structured log lines from the run
├── test_results.json (optional) Pytest results if tests were executed
└── patches.diff      (optional) Unified diff if files were modified
```

**Run ID format**: `{YYYYMMDD_HHMMSS}_{uuid8}` — e.g., `20260514_142301_a3f9b1c2`

The artifact store is written asynchronously using `aiofiles` so it never blocks the main event loop.

**Relationship to memory**: ArtifactStore is the raw data source; `ExecutionMemory` is the summarized index. If a richer query is needed (e.g., "show me all prompts that produced failing tests"), the `data/artifacts/runs/` directory can be scanned directly — each `run.json` contains all the structured metadata needed for arbitrary analysis.

---

## The Feedback Loop Design

The intended feedback loop between runs is:

```
Run completes
     │
     ▼
ArtifactStore.save_run()      ← Full run record written to disk
     │
     ▼
ExecutionMemory.record()      ← Summary: model, duration, success/fail
     │
     ▼
PatternStore.record_pattern() ← Pattern annotation if a fix type was identified
     │
     ▼
[Future] LearningService      ← Analyzes accumulated data, updates strategy
     │
     ▼
ModelService reads strategy   ← Next run uses the better model/prompt
```

Today, the first three steps are implemented. The loop breaks at step 4 — data is collected but not yet consumed to change system behavior.

---

## What Data Is Captured Today vs. What's Missing

### Captured today

| Data | Where | Format |
|---|---|---|
| Run outcome (success/fail) | ExecutionMemory | JSON |
| Model used per run | ExecutionMemory | JSON |
| Duration per run | ExecutionMemory | JSON |
| Full prompt text | ArtifactStore | Markdown |
| Full response text | ArtifactStore | Markdown |
| Assembled context | ArtifactStore | JSON |
| Test results | ArtifactStore | JSON |
| Code diffs | ArtifactStore | Unified diff |
| Pattern occurrences | PatternStore | JSON |

### Missing for true learning

| Missing data | Why it matters |
|---|---|
| **Response quality score** | Without a quality metric, there is no signal to optimize toward |
| **Human feedback** | The system has no way to incorporate engineer judgment ("this fix was wrong") |
| **Before/after code quality** | Static analysis scores before and after a run would quantify improvement |
| **Model comparison data** | No A/B testing infrastructure to compare two models on the same prompt |
| **Failure root causes** | `error` field captures exception text, not the semantic reason for failure |
| **Downstream impact** | Does a suggested refactor reduce bugs in subsequent runs? Not tracked. |
| **Cross-run prompt evolution** | Prompts don't change between runs based on history — they are static |

The data infrastructure (JSON files, timestamped runs, artifact directories) is fully ready to support learning. The gap is the analytical layer that reads this data and acts on it.

---

## Memory Access Patterns

All memory components follow the same access pattern:

```python
# Initialization (called once at startup by Orchestrator)
execution_memory = ExecutionMemory(data_dir="data/memory")
pattern_store = PatternStore(data_dir="data/memory")
memory_store = MemoryStore(data_dir="data/memory")

# Write (called after every run)
execution_memory.record(run_id, workflow, model, task_type, success, duration_ms)

# Read (called by ModelService or future LearningService)
stats = execution_memory.get_model_stats("qwen2.5-coder:7b")
patterns = pattern_store.get_patterns(pattern_type="fix_strategy")
```

All writes are synchronous file I/O (JSON serialization + file write). This is intentional — memory writes are a low-priority side effect of run completion and should not be on the async critical path. Future optimization would batch writes and use SQLite instead of JSON for richer querying.
