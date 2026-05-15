# Autonomous Audit Roadmap — AI Engineering Orchestrator

## Current State (May 2026): Reactive, Single-Request Model

The system today is a **request-response engineering assistant**. It does not initiate work autonomously. Every action requires a human to send an HTTP request. The system:

- Receives a prompt and repo path via `POST /api/v1/agent/run`
- Assembles context, selects a model, runs an agent or workflow
- Returns a structured response and writes artifacts to disk
- Records the run in memory for future analysis

This is genuinely useful — the context assembly is sophisticated, the agents produce high-quality output, and the artifact trail is complete. But it is reactive. The codebase does not change unless someone asks.

**What is already built that enables the roadmap:**

| Capability | Component | How it enables future phases |
|---|---|---|
| Graph-aware context | `GraphifyParser` | Foundation for automated complexity scoring and risk detection |
| Structured agent framework | `BaseAgent`, `AgentFactory` | Ready to add Planner, Verifier, and Learning agents without framework changes |
| Execution memory | `ExecutionMemory`, `PatternStore` | Data layer for model performance tracking and strategy optimization |
| Full artifact persistence | `ArtifactStore` | Raw data source for quality scoring and outcome analysis |
| Async throughout | `asyncio`, `httpx` | Ready to handle continuous background monitoring without blocking the API |
| Circuit breakers | `CircuitBreaker` in `OllamaClient` | Required for autonomous operation — self-healing on Ollama restarts |
| Path security | `PathValidator`, `PathSecurityError` | Safe autonomous file writes within repo boundaries |
| 858 tests, 85.14% coverage | `tests/` | Confidence floor for autonomous behavior changes |

---

## Phase 1 (Q3 2026): Proactive Monitoring

**Goal**: The system watches the codebase and triggers analysis automatically on change, without human prompting.

### What gets built

**1.1 File Watcher Service**

```python
# app/monitoring/file_watcher.py
class FileWatcherService:
    """Watches a repo path and enqueues analysis jobs on file change."""

    def __init__(self, repo_path: str, orchestrator: OrchestratorService) -> None:
        self._repo_path = repo_path
        self._orchestrator = orchestrator
        self._queue: asyncio.Queue = asyncio.Queue()
        self._debounce_seconds = 2.0  # Avoid burst triggers on save

    async def start(self) -> None:
        """Start watchdog observer in background thread, drain queue in event loop."""
        ...

    async def _on_file_changed(self, path: str, event_type: str) -> None:
        """Enqueue lightweight scan job for changed file."""
        ...
```

Library: `watchdog` (pure Python, cross-platform inotify/FSEvents/kqueue wrapper)

**1.2 Git Hook Integration**

Install hooks at project setup time:

```bash
# .git/hooks/post-commit (installed by setup.sh)
#!/bin/bash
curl -s -X POST http://localhost:8000/api/v1/hooks/post-commit \
  -H "Content-Type: application/json" \
  -d "{\"commit\": \"$(git rev-parse HEAD)\", \"files\": $(git diff --name-only HEAD~1 HEAD | jq -R . | jq -s .)}"
```

The API endpoint enqueues a targeted analysis of modified files rather than a full scan.

**1.3 Test Monitor**

Runs `pytest --json-report` after each file change and compares coverage delta against the previous run. If coverage drops more than 2%, enqueues a `TestingAgent` job to identify the uncovered path.

**1.4 CI Webhook Receiver**

Receives `POST /api/v1/webhooks/github-actions` with GitHub Actions workflow completion events. On test failure, automatically triggers a `DebugWorkflow` against the failing test file.

### Technical requirements

- Python `watchdog>=4.0.0` added to dependencies
- New `app/monitoring/` module with `FileWatcherService`, `GitHookHandler`, `CIWebhookHandler`
- Async job queue (`asyncio.Queue`) to decouple event detection from analysis execution
- Background task runner integrated with FastAPI lifespan events
- Deduplication: don't enqueue the same file twice within a 2-second window

### Success metrics

- File change to analysis complete: < 30 seconds for single-file changes
- Zero false negatives: every commit triggers at least one analysis job
- No interference with test suite (monitoring runs in a separate process)

---

## Phase 2 (Q4 2026): Autonomous Detection

**Goal**: The system not only watches for changes but understands which changes are risky and prioritizes autonomous work accordingly.

### What gets built

**2.1 Anomaly Detector**

Compares graph snapshots before and after each commit to detect:

- **Complexity increase**: cyclomatic complexity of a function exceeds its 90-day moving average by more than 20%
- **Coverage drop**: test coverage for a module falls below 80%
- **New external dependency**: a new import appears that wasn't in the previous snapshot
- **Entry point modification**: a route handler or public API function is changed

```python
# app/detection/anomaly_detector.py
class AnomalyDetector:
    async def compare_snapshots(
        self,
        before: GraphSnapshot,
        after: GraphSnapshot,
    ) -> list[Anomaly]: ...

    async def score_anomaly(self, anomaly: Anomaly) -> RiskScore: ...
```

**2.2 Risk Scorer**

Assigns a numeric risk score (0–100) to each module based on:

- Change frequency: how often is this file modified? (from git log)
- Bug history: how many past debug runs targeted this file? (from ExecutionMemory)
- Criticality: is this file on a critical path? (from Graphify entry-point analysis)
- Complexity trend: is complexity increasing over time?

```python
# app/detection/risk_scorer.py
@dataclass
class ModuleRiskScore:
    module_path: str
    score: float          # 0.0 (low) to 100.0 (critical)
    factors: dict[str, float]  # contribution per factor
    recommended_action: str
```

**2.3 Priority Queue**

An ordered work queue where the orchestrator pulls from the highest-risk items first. The queue is persistent (written to `data/detection/queue.json`) so autonomous work survives a restart.

```python
# app/detection/work_queue.py
class AutonomousWorkQueue:
    async def enqueue(self, item: WorkItem) -> None: ...
    async def dequeue(self) -> WorkItem | None: ...  # Returns highest-risk item
    async def list_pending(self) -> list[WorkItem]: ...
```

### Technical requirements

- Graph snapshot persistence: Graphify output stored per-commit in `data/snapshots/{commit_hash}/`
- Git history reader: `git log --follow --name-status` parsed by a new `GitHistoryReader` utility
- SQLite upgrade for `ExecutionMemory` (JSON file insufficient for range queries needed by risk scorer)
- Work queue with priority ordering and deduplication by `(repo_path, module_path)`

### Success metrics

- Risk score correlation with actual bugs: >70% of files scored HIGH-CRITICAL in the prior week were the target of a debug run in the same week
- Priority queue drains at a rate matching Ollama's throughput (no growing backlog during normal operation)
- Anomaly false positive rate: <15% (most anomalies flagged are genuine concerns)

---

## Phase 3 (Q1 2027): Strategic Reasoning

**Goal**: The system understands business context well enough to prioritize autonomous work by impact, not just technical risk.

### What gets built

**3.1 Business Context Model**

A manually authored configuration file that maps code modules to business capabilities:

```yaml
# config/business_context.yaml
critical_paths:
  - module: "app/payments/"
    business_capability: "payment_processing"
    revenue_impact: "direct"
    sla_seconds: 2.0
    compliance: ["PCI-DSS"]

  - module: "app/auth/"
    business_capability: "user_authentication"
    revenue_impact: "indirect"
    sla_seconds: 1.0
    compliance: ["SOC2"]
```

**3.2 Impact Scorer**

Combines technical risk score with business impact:

```
final_priority = (technical_risk * 0.4) + (business_impact * 0.4) + (compliance_risk * 0.2)
```

**3.3 Planner Agent**

A new agent type (`app/agents/specialized/planner_agent.py`) that:
- Reads the priority queue
- Reads the business context model
- Decides which items to work on in the next execution window
- Decomposes complex items into sub-tasks for specialized agents

**3.4 Execution Scheduler**

Runs the Planner Agent on a configurable schedule (e.g., every hour during business hours, every 4 hours at night) and dispatches the resulting work plan to the execution layer.

### Technical requirements

- `config/business_context.yaml` schema and loader
- `PlannerAgent` with specialized system prompt for strategic reasoning
- Scheduler integrated with FastAPI lifespan (replaces the simple queue poller from Phase 2)
- API endpoint: `GET /api/v1/planner/plan` to see the current work plan

### Success metrics

- Autonomous work targets business-critical modules at 2x the rate of non-critical modules
- Mean time to detect and triage a new vulnerability in a critical path module: < 4 hours
- Operator intervention rate: operators override the planner's plan in < 20% of cycles

---

## Phase 4 (Q2 2027): Self-Evolution

**Goal**: The system improves its own strategies from accumulated outcome data. Prompts, model selections, and workflow configurations evolve based on what has worked.

### What gets built

**4.1 LearningService (now implemented)**

The placeholder from Phase 1 becomes real:

```python
class LearningService:
    async def run_weekly_optimization(self) -> OptimizationReport:
        """
        Analyze last 7 days of execution data.
        Cross-reference model choices with outcome quality.
        Update ModelService strategy table.
        """

    async def a_b_test_prompt(
        self,
        agent_type: str,
        variant_a: str,
        variant_b: str,
        sample_size: int = 20,
    ) -> PromptTestResult:
        """Run both prompt variants on real tasks and compare outcomes."""
```

**4.2 Strategy Configuration Store**

A mutable strategy table that `ModelService` reads from:

```json
// data/strategies/current.json
{
  "code_generation": {
    "model": "qwen2.5-coder:7b",
    "temperature": 0.2,
    "strategy_version": 14,
    "updated_at": "2027-04-03T...",
    "confidence": 0.87
  }
}
```

**4.3 Prompt A/B Testing Infrastructure**

The system runs two prompt variants in parallel on the same task class and records quality scores (test pass rate, static analysis score, human approval rate) to determine the winner.

**4.4 Model Performance Tracker**

Implements `get_best_model_for_task()` in `PatternStore` with real logic:

```python
def get_best_model_for_task(self, task_type: str) -> str | None:
    from app.memory.execution_memory import ExecutionMemory
    mem = ExecutionMemory()
    # Group by model, filter to task_type, rank by success_rate then avg_duration
    stats = mem.get_stats_by_task_type(task_type)
    if not stats:
        return None
    return max(stats, key=lambda s: (s["success_rate"], -s["avg_duration_ms"]))["model"]
```

### Technical requirements

- Quality scoring pipeline: automated static analysis (ruff, mypy) run post-response to score code quality
- SQLite migration for ExecutionMemory (JSON capped at 1000 entries is insufficient for 6+ months of data)
- Versioned strategy store with rollback capability
- `LearningService` as a scheduled background task (weekly at off-peak hours)

### Success metrics

- Model selection accuracy improves: success rate per task type increases by >5% over 90 days
- Prompt A/B tests converge within 20 samples in >80% of experiments
- Strategy confidence scores correlate with actual success rates (r > 0.8)

---

## Phase 5 (Q3 2027): CTO-Level Autonomy

**Goal**: The system can generate and execute engineering roadmaps without human decomposition. It decides what to build, when, and in what order — subject to human approval at defined checkpoints.

### What gets built

**5.1 Roadmap Generator**

Ingests the business context model, current technical risk scores, known vulnerabilities, test coverage gaps, and performance regressions to generate a prioritized engineering roadmap:

```
Q3 2027 Engineering Roadmap (generated by AI Engineering OS)
──────────────────────────────────────────────────────────────
WEEK 1-2: Reduce auth.py complexity (complexity score: 47, risk: HIGH)
          - Decompose validate_session() into 3 focused functions
          - Increase test coverage from 71% to >90%
          - Projected risk score after: 18 (LOW)

WEEK 3-4: Resolve CVE-2027-14521 in payments/ dependency
          - Upgrade stripe SDK 8.1.2 → 8.3.0
          - Regression test payment flows
          - Update integration tests for new API

WEEK 5-6: Performance regression in /api/v1/search (p99 degraded 340ms)
          - Profile query path
          - Implement result caching with TTL
          ...
```

**5.2 Architectural Decision Records**

The system generates ADRs (Architecture Decision Records) for proposed changes above a complexity threshold:

```
data/decisions/ADR-0047-auth-refactor.md
```

These are committed to the repo and reviewed by engineers before the autonomous execution window opens.

**5.3 Approval Workflow**

For roadmap items above a configurable risk threshold, the system creates a GitHub pull request with the proposed changes, waits for human approval, and merges only after approval. Below the threshold, changes are committed autonomously.

### Technical requirements

- GitHub API integration (`gh` CLI or `PyGithub`) for PR creation and status checking
- ADR template and generation logic in `DocumentationAgent`
- Configurable approval thresholds per module criticality
- Rollback automation: if a merged change causes test failures in CI, automatically revert and enqueue a debug run

### Success metrics

- Roadmap adoption rate: >60% of generated roadmap items are approved and executed within the target week
- Autonomous commit quality: CI pass rate on autonomous commits > 95%
- Mean time to full roadmap cycle (generate → execute → verify): < 2 weeks
- Engineer time saved per sprint: > 20% of engineering capacity redirected from maintenance to new features
