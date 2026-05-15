# Agent Framework — AI Engineering Orchestrator

## Overview

The agent framework is the execution core of the orchestrator. Agents are not LLM wrappers — they are **domain-specialized reasoning entities** with structured prompts, real tools (file access, test execution), observable step logs, and typed result schemas. They are designed to be composable: workflows chain agents together, and agents can invoke tools whose outputs feed back into the same reasoning loop.

---

## BaseAgent Design

`BaseAgent` (`app/agents/base_agent.py`) is an abstract class that all specialized agents extend. It defines the contract every agent must fulfill:

```python
class BaseAgent(ABC):
    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self._ollama = ollama_client or get_default_client()
        self._status: AgentStatus = AgentStatus.IDLE
        self._logger = get_logger(f"agent.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str: ...           # e.g. "code", "security"

    @property
    @abstractmethod
    def description(self) -> str: ...   # one-line description

    @property
    @abstractmethod
    def system_prompt(self) -> str: ... # the domain expertise prompt

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]: ...  # tools available during execution

    @abstractmethod
    async def _execute(self, task: AgentTask) -> str: ...
    # Domain-specific execution logic; returns raw response text

    async def run(self, task: AgentTask) -> AgentResult: ...
    # Wraps _execute with status tracking, timing, error handling

    async def run_streaming(self, task: AgentTask) -> AsyncIterator[str]: ...
    # Yields response tokens as they arrive from Ollama
```

The separation between `_execute` (abstract, domain logic) and `run` (concrete, infrastructure) is deliberate: subclasses write only the intelligence; the base class handles all observability, timing, and error normalization.

---

## Data Schemas

### AgentTask

The task submitted to any agent. All fields are typed; `context` is typed as `Any` to avoid circular imports with `AssembledContext`.

```python
class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str                          # The engineering question or instruction
    repo_path: str                       # Absolute path to the codebase
    context: Any | None = None           # AssembledContext from ContextAssembler
    model: str = "qwen2.5-coder:7b"     # Ollama model name
    temperature: float = 0.2            # Low temperature = deterministic output
    max_tokens: int = 4096
    stream: bool = False
    metadata: dict[str, Any] = {}
```

`temperature=0.2` is the default for all agents. Engineering tasks require deterministic output — high temperature produces creative but unreliable code. Only `DocumentationAgent` uses `temperature=0.3` for slightly more natural prose.

### AgentResult

The complete result of an agent execution. Returned by `run()`.

```python
@dataclass
class AgentResult:
    agent_name: str               # e.g. "code"
    execution_id: str             # UUID for this specific run
    status: AgentStatus           # IDLE / RUNNING / COMPLETED / FAILED
    response: str                 # The model's full response text
    steps: list[AgentStep]        # Reasoning steps recorded during execution
    tools_used: list[str]         # Names of tools that were invoked
    tokens_approx: int            # Approximate token count
    duration_ms: float            # Wall-clock time
    error: str | None             # If status is FAILED
    summary: str                  # One-line summary for logging
    files_read: list[str]         # Files accessed via FileReader tool
    files_written: list[str]      # Files modified via FileWriter tool
    tests_run: int                # Number of tests executed
    tests_passed: int
    confidence: float = 1.0       # Agent's self-reported confidence [0.0, 1.0]
    next_actions: list[str]       # Suggested follow-up tasks
    artifacts: list[dict]         # Additional structured outputs
```

### AgentStep

One recorded reasoning step within an agent's execution. Used for audit trails and debugging.

```python
@dataclass
class AgentStep:
    thought: str              # The agent's internal reasoning
    action: str | None        # What action was decided
    tool_used: str | None     # Tool name if a tool was invoked
    tool_input: dict | None   # Arguments passed to the tool
    observation: str | None   # Result from the tool
```

---

## BaseTool and ToolResult

Tools are the agent's interface to the real world. Each tool is a class implementing `BaseTool`:

```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...
```

```python
@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    duration_ms: float = 0.0
```

Current tools in `app/tools/`:

| Tool | Class | What it does |
|---|---|---|
| `file_reader` | `FileReader` | Reads file contents from `repo_path`; enforces path traversal guard |
| `file_writer` | `FileWriter` | Writes or patches files; creates backups before overwriting |
| `test_runner` | `TestRunner` | Runs `pytest` in a subprocess and captures JSON results |

All tools validate that their target path is within `repo_path` before acting. The `PathValidator` utility is invoked at the tool level as a second line of defense (the first being the API layer).

---

## Tool Use Pattern

Agents invoke tools during `_execute` by building a prompt that includes tool descriptions, observing the model's tool call requests, executing the tools, and feeding results back into the conversation. The pattern:

```
1. Build initial prompt
   ┌─────────────────────────────────────────────┐
   │ system_prompt                               │
   │ + context (graph, skills, files)            │
   │ + user_prompt                               │
   │ + available tool descriptions               │
   └─────────────────────────────────────────────┘
              │
              ▼
2. OllamaClient.generate() → response_text
              │
              ▼
3. Parse response for tool call markers
   e.g. <tool>file_reader</tool><path>app/auth.py</path>
              │
   ┌──────────▼──────────┐
   │  Tool call found?   │
   └──────────┬──────────┘
              │ YES
              ▼
4. Execute tool → ToolResult
5. Record as AgentStep
6. Append tool result to prompt
7. Re-invoke OllamaClient.generate()
              │
   ┌──────────▼──────────┐
   │  More tool calls?   │─── YES → go to step 3
   └──────────┬──────────┘
              │ NO
              ▼
8. Return final response text
```

The loop continues until the model produces a response with no tool call markers, or until a maximum step limit (default: 10) is reached to prevent infinite loops.

---

## The 6 Specialized Agents

### CodeAgent (`app/agents/specialized/code_agent.py`)

**Domain**: Python code generation, review, and refactoring

**System prompt focus**: Clean, type-safe, async Python 3.11+. No placeholders. TDD-first. Functions do one thing. Errors surface explicitly.

**Tools**: `FileReader`, `FileWriter`, `TestRunner`

**Unique behavior**: After generating or modifying code, CodeAgent automatically attempts to run the test suite with `TestRunner` and includes the result in its `AgentResult`. If tests fail, it records the failure in `errors[]` and sets `confidence` below 0.8.

**Model preference**: `qwen2.5-coder:7b` (fine-tuned for code generation; significantly outperforms general models on Python tasks)

---

### ArchitectureAgent (`app/agents/specialized/architecture_agent.py`)

**Domain**: System design, dependency analysis, architectural review

**System prompt focus**: SOLID principles, hexagonal architecture, identifying hidden coupling, naming module boundaries. Produces structured Markdown reports.

**Tools**: `FileReader` (reads all `__init__.py` and import blocks to reconstruct dependency graph manually if Graphify is unavailable)

**Unique behavior**: Works primarily with the Graphify graph context — it interprets graph structure rather than reading individual source files. Produces architecture diagrams in ASCII and structured findings with severity levels (INFO, WARNING, CRITICAL).

**Model preference**: `llama3.2` or `codellama` — general reasoning models outperform pure code models on architecture analysis

---

### TestingAgent (`app/agents/specialized/testing_agent.py`)

**Domain**: Test generation, coverage analysis, test strategy

**System prompt focus**: Pytest idioms, parametrize, fixtures, mocks, property-based testing with Hypothesis. Tests must be fast, isolated, deterministic.

**Tools**: `FileReader`, `FileWriter`, `TestRunner`

**Unique behavior**: Reads existing test structure before generating new tests to match conventions. After writing tests, runs them immediately with `TestRunner` and iterates up to 3 times to fix failures — the only agent with a built-in self-correction loop.

---

### DocumentationAgent (`app/agents/specialized/documentation_agent.py`)

**Domain**: Docstrings, API documentation, README, architecture docs

**System prompt focus**: Clear, precise prose. No jargon padding. Document the "why", not the "what". Follow Google/NumPy docstring conventions.

**Tools**: `FileReader`, `FileWriter`

**Unique behavior**: Uses `temperature=0.3` for slightly more natural prose. Reads existing documentation files before generating to avoid duplication and match voice.

---

### SecurityAgent (`app/agents/specialized/security_agent.py`)

**Domain**: Security vulnerability detection, OWASP compliance, secret detection

**System prompt focus**: SQL injection, XSS, path traversal, hardcoded secrets, unsafe deserialization, OWASP Top 10. Report with CVE references where applicable.

**Tools**: `FileReader`

**Unique behavior**: Read-only — `SecurityAgent` never writes to files. It produces structured findings (severity, file, line, description, remediation) as JSON within its response, which the orchestrator parses and stores in `AuditWorkflowResult.security_findings`.

**Model preference**: Any capable reasoning model; security scan quality depends more on context quality than model size

---

### PerformanceAgent (`app/agents/specialized/performance_agent.py`)

**Domain**: Performance profiling, complexity analysis, async correctness

**System prompt focus**: Big-O analysis, N+1 query detection, blocking calls in async context, memory allocation patterns, connection pool exhaustion.

**Tools**: `FileReader`, `TestRunner` (to run benchmarks)

**Unique behavior**: Flags blocking I/O in async functions (e.g., `time.sleep()`, synchronous file reads, synchronous DB calls) as HIGH severity findings. Identifies missing `await` on coroutines.

---

## AgentFactory: Registry Pattern

`AgentFactory` (`app/agents/agent_factory.py`) provides two things: a string registry mapping agent names to class paths, and a singleton cache that returns the same instance for repeated calls.

```python
_REGISTRY: dict[str, str] = {
    "code":          "app.agents.specialized.code_agent:CodeAgent",
    "architecture":  "app.agents.specialized.architecture_agent:ArchitectureAgent",
    "testing":       "app.agents.specialized.testing_agent:TestingAgent",
    "documentation": "app.agents.specialized.documentation_agent:DocumentationAgent",
    "security":      "app.agents.specialized.security_agent:SecurityAgent",
    "performance":   "app.agents.specialized.performance_agent:PerformanceAgent",
}
```

Dynamic imports (`importlib.import_module`) keep the registry declaration lightweight and prevent circular imports. The `_cache` dict on the factory instance ensures each agent is initialized once per factory lifetime — agent initialization connects to Ollama, so avoiding repeated initialization is meaningful.

The global factory singleton is accessed via `get_agent_factory()`:

```python
_factory: AgentFactory | None = None

def get_agent_factory(ollama_client: OllamaClient | None = None) -> AgentFactory:
    global _factory
    if _factory is None:
        _factory = AgentFactory(ollama_client=ollama_client)
    return _factory
```

---

## Creating a New Agent: Step-by-Step

**1. Create the module file**

```
app/agents/specialized/my_agent.py
```

**2. Define the system prompt**

```python
_SYSTEM_PROMPT = """You are an expert in [domain].

Your core discipline:
- [Principle 1]
- [Principle 2]
- [Principle 3]

When analyzing code:
- [What to look for]
- [What to flag]
"""
```

**3. Implement the agent class**

```python
from app.agents.base_agent import AgentTask, BaseAgent, BaseTool
from app.integrations.ollama.client import OllamaClient
from app.tools.file_system.file_reader import FileReader

class MyAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "my_agent"

    @property
    def description(self) -> str:
        return "Expert in [domain] for [purpose]"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    @property
    def tools(self) -> list[BaseTool]:
        return self._active_tools

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        super().__init__(ollama_client)
        self._active_tools: list[BaseTool] = []

    async def _execute(self, task: AgentTask) -> str:
        self._active_tools = [FileReader(task.repo_path)]

        # Build context-aware prompt
        context_text = ""
        if task.context:
            context_text = str(task.context)

        full_prompt = (
            f"{context_text}\n\n"
            f"Task: {task.prompt}"
        )

        response = await self._ollama.generate(
            model=task.model,
            prompt=full_prompt,
            system=self.system_prompt,
            temperature=task.temperature,
        )
        return response.get("response", "")
```

**4. Register in the factory**

Edit `app/agents/agent_factory.py`:

```python
_REGISTRY: dict[str, str] = {
    # ... existing agents ...
    "my_agent": "app.agents.specialized.my_agent:MyAgent",
}
```

**5. Write tests**

```
tests/test_agents/test_my_agent.py
```

Test `_execute` with a mocked `OllamaClient`. Test `run()` to verify `AgentResult.status == COMPLETED` on success and `== FAILED` on exception.

**6. Use the agent**

```python
factory = get_agent_factory()
agent = factory.get_agent("my_agent")
result = await agent.run(AgentTask(
    prompt="Analyze this codebase for [domain issues]",
    repo_path="/path/to/repo",
    model="qwen2.5-coder:7b",
))
print(result.response)
```

---

## Streaming Support

`run_streaming()` is an async generator that yields string chunks as tokens arrive from Ollama. The Ollama client's streaming endpoint sends server-sent events; the client's `stream_generate()` method yields each token chunk as it arrives.

```python
async def run_streaming(self, task: AgentTask) -> AsyncIterator[str]:
    self._status = AgentStatus.RUNNING
    start = time.monotonic()
    try:
        prompt = await self._build_prompt(task)
        async for chunk in self._ollama.stream_generate(
            model=task.model,
            prompt=prompt,
            system=self.system_prompt,
        ):
            yield chunk
        self._status = AgentStatus.COMPLETED
    except Exception as e:
        self._status = AgentStatus.FAILED
        self._logger.error("Streaming failed", error=str(e))
        yield f"\n\n[ERROR: {e}]"
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        self._logger.info("Streaming complete", duration_ms=duration_ms)
```

The API endpoint (`POST /api/v1/agent/stream`) wraps this generator in a `StreamingResponse`, forwarding each chunk to the client over HTTP chunked transfer encoding. Clients receive the response token-by-token without waiting for the full generation.

---

## Metrics Tracking

Every `run()` call initializes a `MetricsTracker` context that records:

```python
async with MetricsTracker(agent_name=self.name, task_id=task.task_id) as tracker:
    result = await self._execute(task)
    tracker.record_tokens(result.tokens_approx)
    tracker.record_tools_used(result.tools_used)
```

Prometheus counters and histograms updated per run:

- `agent_executions_total{agent, status}` — count of runs by agent and outcome
- `agent_execution_duration_seconds{agent}` — histogram of wall-clock time
- `agent_tokens_used_total{agent, model}` — total tokens consumed
- `successful_code_generations_total` — count of code-writing successes

Metrics are exposed on `GET /metrics` for Prometheus scraping.

---

## Error Handling

Errors are classified into two categories:

**Infrastructure errors** (propagate up, trigger retry):
- `OllamaClientError`: network or HTTP error talking to Ollama
- `ModelNotFoundError`: requested model not available; triggers model fallback
- `ModelTimeoutError`: inference took longer than the configured timeout
- `CircuitBreakerOpenError`: circuit is open; rejected immediately

**Agent-level errors** (swallowed, recorded in AgentResult):
- `tool_execution_error`: a tool call failed (e.g., file not found, test runner crash); recorded in `AgentResult.errors[]`, execution continues
- `parsing_error`: the model's response could not be parsed for tool calls; agent returns its response as-is
- `confidence_below_threshold`: agent sets `confidence < 0.5`; no exception, but `next_actions` includes a retry suggestion

The `run()` method always returns an `AgentResult`. It never raises. If `_execute()` raises an unhandled exception, `run()` catches it, sets `status=FAILED`, and populates `AgentResult.error`. The orchestrator inspects `result.status` to decide whether to retry or return an error response to the caller.

```python
async def run(self, task: AgentTask) -> AgentResult:
    execution_id = str(uuid.uuid4())
    self._status = AgentStatus.RUNNING
    start = time.monotonic()

    try:
        response = await self._execute(task)
        duration_ms = (time.monotonic() - start) * 1000
        self._status = AgentStatus.COMPLETED
        return AgentResult(
            agent_name=self.name,
            execution_id=execution_id,
            status=AgentStatus.COMPLETED,
            response=response,
            duration_ms=duration_ms,
        )
    except Exception as e:
        self._logger.error("Agent execution failed", error=str(e), exc_info=True)
        self._status = AgentStatus.FAILED
        return AgentResult(
            agent_name=self.name,
            execution_id=execution_id,
            status=AgentStatus.FAILED,
            response="",
            error=str(e),
            duration_ms=(time.monotonic() - start) * 1000,
        )
```
