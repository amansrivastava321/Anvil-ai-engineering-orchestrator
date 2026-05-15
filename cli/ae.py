"""ae — natural language CLI for the AI Engineering Orchestrator."""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

import click
import httpx
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8008"
DEFAULT_TIMEOUT = 300  # 5 minutes

WORKFLOW_KEYWORDS: dict[str, list[str]] = {
    "debug_analysis": ["debug", "fix", "bug", "error", "crash", "broken", "fail", "exception", "traceback"],
    "architecture_analysis": ["architecture", "structure", "design", "overview", "diagram", "system", "components"],
    "test_generation": ["test", "tests", "coverage", "unittest", "pytest"],
    "code_refactoring": ["refactor", "clean", "improve", "simplify", "restructure", "reorganize"],
    "documentation": ["document", "explain", "readme", "docs", "describe", "comment", "docstring"],
    "code_generation": ["write", "create", "generate", "implement", "build", "scaffold"],
    "code_review": ["review", "audit", "check", "analyze", "inspect", "evaluate"],
}

# Subcommand names so main() can detect them and avoid misrouting
_KNOWN_SUBCOMMANDS = {"status", "models", "watch", "dashboard", "run", "--help", "-h", "--version"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_workflow(prompt: str) -> str:
    """Infer workflow type from prompt using whole-word keyword matching."""
    lower = prompt.lower()
    for workflow, keywords in WORKFLOW_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return workflow
    return "general_qa"


def get_base_url() -> str:
    return os.environ.get("AE_BASE_URL", DEFAULT_BASE_URL)


def get_repo_path(repo: Optional[str]) -> str:
    path = Path(repo) if repo else Path.cwd()
    if not path.exists():
        console.print(f"[red]Repository not found at {path}[/red]")
        sys.exit(1)
    return str(path.resolve())


def _client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=get_base_url(), timeout=timeout)


def _handle_connection_error() -> None:
    console.print(
        Panel(
            "[red]Cannot connect to the AI Engineering Orchestrator.[/red]\n\n"
            "Start the server with:\n"
            "[bold cyan]uvicorn app.main:app --port 8008[/bold cyan]",
            title="Connection Error",
            border_style="red",
        )
    )
    sys.exit(1)


def _format_duration(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ── Result rendering ──────────────────────────────────────────────────────────

def render_response(data: dict) -> None:
    status = data.get("status", "unknown")
    error = data.get("error")

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("Model", f"[cyan]{data.get('model_used', 'unknown')}[/cyan]")
    meta.add_row("Workflow", data.get("workflow_type", "—").replace("_", " ").title())
    meta.add_row("Duration", _format_duration(data.get("duration_ms", 0)))
    meta.add_row("Tokens", str(data.get("tokens_used", 0)))
    if data.get("graphify_available"):
        meta.add_row("Graphify", "[green]loaded[/green]")
    files = data.get("files_analyzed", [])
    if files:
        meta.add_row("Files", str(len(files)))

    border = "red" if error or status == "failed" else "green"
    title = f"[bold]{'[red]FAILED[/red]' if error else '[green]DONE[/green]'}[/bold]  {data.get('execution_id', '')[:8]}"
    console.print()
    console.print(meta)

    if error:
        console.print(Panel(f"[red]{error}[/red]", title="Error", border_style="red"))
        return

    response_text = data.get("response") or ""
    if response_text:
        console.print(Panel(Markdown(response_text), title=title, border_style=border, padding=(1, 2)))

    for w in data.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/yellow]")


# ── Sync / streaming runners ─────────────────────────────────────────────────

def _run_sync(payload: dict) -> None:
    workflow_type = payload.get("workflow_type", "")
    steps = [
        f"Detecting workflow: [cyan]{workflow_type.replace('_', ' ').title()}[/cyan]",
        "Assembling context…",
        "Running agent…",
        "Formatting response…",
    ]
    step_idx = 0

    try:
        with _client() as client:
            with console.status(steps[0], spinner="dots") as status_ctx:
                start = time.monotonic()

                def _advance() -> None:
                    nonlocal step_idx
                    step_idx = min(step_idx + 1, len(steps) - 1)
                    status_ctx.update(steps[step_idx])

                result: dict = {}
                exc_holder: list = []

                def _call() -> None:
                    try:
                        r = client.post("/api/v1/agent/run", json=payload)
                        result["data"] = r.json()
                        result["status_code"] = r.status_code
                    except Exception as e:
                        exc_holder.append(e)

                _advance()
                t = threading.Thread(target=_call, daemon=True)
                t.start()
                while t.is_alive():
                    t.join(timeout=2.0)
                    _advance()

            if exc_holder:
                raise exc_holder[0]

            elapsed = (time.monotonic() - start) * 1000
            data = result.get("data", {})
            if not isinstance(data, dict):
                data = {"error": str(data)}
            data.setdefault("duration_ms", elapsed)

            if result.get("status_code", 200) >= 400:
                err = data.get("error") or data.get("detail") or "Unknown error"
                console.print(Panel(f"[red]{err}[/red]", title="Request Failed", border_style="red"))
                sys.exit(1)

            render_response(data)

    except httpx.ConnectError:
        _handle_connection_error()
    except httpx.TimeoutException:
        console.print("[red]Request timed out after 5 minutes.[/red]")
        sys.exit(1)


def _run_streaming(payload: dict) -> None:
    workflow_type = payload.get("workflow_type", "")
    try:
        with _client() as client:
            console.print(f"[dim]Streaming · {workflow_type.replace('_', ' ').title()}[/dim]\n")
            with client.stream("POST", "/api/v1/agent/run/stream", json=payload) as resp:
                if resp.status_code >= 400:
                    console.print(f"[red]Error {resp.status_code}[/red]")
                    sys.exit(1)
                for chunk in resp.iter_text():
                    console.print(chunk, end="")
            console.print()
    except httpx.ConnectError:
        _handle_connection_error()
    except httpx.TimeoutException:
        console.print("\n[red]Stream timed out.[/red]")
        sys.exit(1)


# ── Interactive mode ──────────────────────────────────────────────────────────

def _get_known_repos() -> List[dict]:
    """Fetch known repos from server; returns [] on any error or connection failure."""
    try:
        with _client(timeout=5) as client:
            resp = client.get("/api/v1/repo/list")
            if resp.status_code == 200:
                return resp.json().get("repos", [])
    except Exception:
        pass
    return []


def _pick_folder() -> Optional[str]:
    """Interactive terminal directory browser. Returns selected path or None."""
    try:
        import readline  # noqa: F401 — enables arrow-key editing in input()
    except ImportError:
        pass

    current = Path.cwd()

    while True:
        console.print(f"\n[bold cyan]  {current}[/bold cyan]")

        try:
            entries = sorted(
                [e for e in current.iterdir() if e.is_dir() and not e.name.startswith(".")],
                key=lambda e: e.name.lower(),
            )
        except PermissionError:
            console.print("[red]  Permission denied — going up[/red]")
            current = current.parent
            continue

        grid = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        grid.add_column(style="dim", justify="right", width=3)
        grid.add_column(style="cyan")
        grid.add_row("0", "[dim].. (parent directory)[/dim]")
        for i, entry in enumerate(entries[:40], 1):
            grid.add_row(str(i), entry.name + "/")

        console.print(grid)
        console.print("[dim]  number → navigate  ·  s → select this folder  ·  q → cancel[/dim]")

        try:
            choice = console.input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "q":
            return None
        if choice == "s":
            return str(current)
        if choice == "0":
            current = current.parent
            continue
        try:
            idx = int(choice)
            if 1 <= idx <= len(entries):
                current = entries[idx - 1]
            else:
                console.print("[yellow]  Out of range[/yellow]")
        except ValueError:
            console.print("[yellow]  Enter a number, 's', or 'q'[/yellow]")


def _select_repo_from_list(repos: List[dict]) -> Optional[str]:
    """Numbered menu of known repos. Returns selected path or None."""
    grid = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    grid.add_column(style="dim", justify="right", width=3)
    grid.add_column(style="cyan bold")
    grid.add_column(style="dim")

    for i, repo in enumerate(repos, 1):
        name = Path(repo.get("path", "")).name or repo.get("name", "unknown")
        grid.add_row(str(i), name, repo.get("path", ""))

    grid.add_row("b", "[yellow]Browse…[/yellow]", "pick a folder from the filesystem")
    grid.add_row("q", "[red]Quit[/red]", "")
    console.print(grid)

    while True:
        try:
            choice = console.input("\n  Select > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "q":
            return None
        if choice == "b":
            return _pick_folder()
        try:
            idx = int(choice)
            if 1 <= idx <= len(repos):
                return repos[idx - 1].get("path")
            console.print("[yellow]  Out of range[/yellow]")
        except ValueError:
            console.print("[yellow]  Enter a number, 'b', or 'q'[/yellow]")


def _prompt_and_run(repo_path: str) -> None:
    """Ask the user what to do with the selected repo, then execute."""
    repo_name = Path(repo_path).name
    console.print(
        f"\n[bold]Repository:[/bold] [cyan]{repo_name}[/cyan]  [dim]{repo_path}[/dim]\n"
    )

    presets = [
        ("1", "Review code for issues",    "code_review"),
        ("2", "Explain the architecture",  "architecture_analysis"),
        ("3", "Generate missing tests",     "test_generation"),
        ("4", "Find and fix bugs",          "debug_analysis"),
        ("5", "Improve documentation",      "documentation"),
    ]

    grid = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    grid.add_column(style="dim", justify="right", width=3)
    grid.add_column(style="cyan")
    for key, label, _ in presets:
        grid.add_row(key, label)
    grid.add_row("c", "[bold]Custom prompt…[/bold]")
    console.print(grid)

    try:
        choice = console.input("\n  Action > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice in ("c", ""):
        try:
            prompt_text = console.input("  Describe the task: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not prompt_text:
            return
        workflow_type = detect_workflow(prompt_text)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                _, prompt_text, workflow_type = presets[idx]
            else:
                console.print("[yellow]Invalid selection[/yellow]")
                return
        except ValueError:
            console.print("[yellow]Invalid selection[/yellow]")
            return

    try:
        stream_yn = console.input("  Stream output? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        stream_yn = "n"
    stream = stream_yn == "y"

    console.print()
    payload = {
        "repo_path": repo_path,
        "prompt": prompt_text,
        "workflow_type": workflow_type,
        "mode": "streaming" if stream else "sync",
    }
    if stream:
        _run_streaming(payload)
    else:
        _run_sync(payload)


def _interactive_main() -> None:
    """No-args entry: pick a repo, then pick an action."""
    console.print(
        Panel(
            "[bold cyan]AI Engineering Orchestrator[/bold cyan]\n"
            "[dim]Select a repository to get started[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    repos = _get_known_repos()

    if repos:
        console.print(f"\n[bold]Known repositories[/bold]  [dim]({len(repos)} found)[/dim]\n")
        repo_path = _select_repo_from_list(repos)
    else:
        console.print("\n[dim]No known repositories — browsing filesystem…[/dim]\n")
        repo_path = _pick_folder()

    if repo_path:
        _prompt_and_run(repo_path)


# ── CLI group and subcommands ─────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """ae — talk to your AI Engineering OS in plain English.

    \b
    Run a task:
      ae "fix the payment timeout bug"
      ae --stream "write tests for payment.py"
      ae --repo /path "explain the architecture"

    \b
    Other commands:
      ae status      System health and stats
      ae models      List available models
      ae watch       Start monitoring a repository
      ae dashboard   Open web dashboard
    """


@cli.command(name="run")
@click.argument("prompt", nargs=-1, required=True)
@click.option("--repo", "-r", default=None, help="Path to repository (default: CWD)")
@click.option("--workflow", "-w", default=None, help="Force workflow type")
@click.option("--model", "-m", default=None, help="Preferred model name")
@click.option("--stream", "-s", is_flag=True, help="Stream output tokens as they arrive")
def run_cmd(prompt: tuple, repo: Optional[str], workflow: Optional[str],
            model: Optional[str], stream: bool) -> None:
    """Execute a natural language engineering task."""
    prompt_text = " ".join(prompt)
    repo_path = get_repo_path(repo)
    workflow_type = workflow or detect_workflow(prompt_text)

    payload = {
        "repo_path": repo_path,
        "prompt": prompt_text,
        "workflow_type": workflow_type,
        "mode": "streaming" if stream else "sync",
    }
    if model:
        payload["preferred_model"] = model

    if stream:
        _run_streaming(payload)
    else:
        _run_sync(payload)


@cli.command()
def status() -> None:
    """Show system health, active agents, and execution stats."""
    try:
        with _client(timeout=10) as client:
            with console.status("Fetching system status…", spinner="dots"):
                health = client.get("/api/v1/system/status").json()
                stats = client.get("/api/v1/agent/stats").json()
                monitor = client.get("/api/v1/monitor/status").json()
                evolution = client.get("/api/v1/evolution/status").json()
    except httpx.ConnectError:
        _handle_connection_error()
        return

    status_icon = {True: "[green]●[/green]", False: "[red]●[/red]"}

    htable = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    htable.add_column("Component", style="bold")
    htable.add_column("Status")
    htable.add_column("Detail", style="dim")

    ollama = health.get("ollama", {})
    htable.add_row("Ollama", status_icon[ollama.get("status") == "healthy"],
                   f"{ollama.get('model_count', 0)} models")
    graphify = health.get("graphify", {})
    htable.add_row("Graphify", status_icon[bool(graphify.get("available"))],
                   f"{graphify.get('repos_analyzed', 0)} repos analyzed")
    skills = health.get("skills", {})
    htable.add_row("Skills", status_icon[bool(skills.get("loaded"))],
                   f"{skills.get('count', 0)} loaded")
    watching = monitor.get("watching", [])
    htable.add_row("Monitor", status_icon[bool(watching)], f"watching {len(watching)} repo(s)")
    evo_cycles = evolution.get("total_cycles_run", 0)
    evo_applied = evolution.get("total_improvements_applied", 0)
    htable.add_row("Evolution", "[cyan]●[/cyan]",
                   f"{evo_cycles} cycles · {evo_applied} improvements applied")

    console.print(Panel(htable, title="[bold]System Health[/bold]", border_style="cyan"))

    stable = Table(box=box.SIMPLE, padding=(0, 1), show_header=False)
    stable.add_column("Metric", style="bold")
    stable.add_column(style="cyan")

    total = stats.get("total_executions", 0)
    success = stats.get("successful_executions", 0)
    rate = f"{success / total * 100:.1f}%" if total else "—"
    stable.add_row("Total executions", str(total))
    stable.add_row("Success rate", rate)
    stable.add_row("Active now", str(stats.get("active_executions", 0)))
    avg_ms = stats.get("avg_duration_ms", 0)
    stable.add_row("Avg duration", _format_duration(avg_ms) if avg_ms else "—")

    console.print(Panel(stable, title="[bold]Execution Stats[/bold]", border_style="cyan"))


@cli.command()
def models() -> None:
    """List available models with tier and capability info."""
    try:
        with _client(timeout=15) as client:
            with console.status("Fetching models…", spinner="dots"):
                data = client.get("/api/v1/models/").json()
    except httpx.ConnectError:
        _handle_connection_error()
        return

    model_list = data.get("models", [])
    if not model_list:
        console.print("[yellow]No models returned. Is Ollama running?[/yellow]")
        return

    table = Table(title=f"Available Models ({data.get('total', len(model_list))})", box=box.ROUNDED)
    table.add_column("Model", style="cyan bold")
    table.add_column("Tier", style="magenta")
    table.add_column("Size", style="dim")
    table.add_column("Best For", style="dim")
    table.add_column("Weight", justify="right", style="dim")

    for m in model_list:
        table.add_row(
            m.get("name", ""),
            m.get("tier", "—"),
            m.get("size", "—"),
            ", ".join(m.get("task_types", [])) or "—",
            f"{m.get('weight', 1.0):.2f}",
        )

    console.print(table)


@cli.command()
@click.argument("repo", required=False)
@click.option("--interval", "-i", default=60, show_default=True, help="Poll interval in seconds")
@click.option("--no-auto-debug", is_flag=True, help="Disable auto-diagnosis on test failures")
def watch(repo: Optional[str], interval: int, no_auto_debug: bool) -> None:
    """Start proactive monitoring of a repository (default: CWD)."""
    repo_path = get_repo_path(repo)
    payload = {
        "repo_paths": [repo_path],
        "poll_interval": interval,
        "auto_debug": not no_auto_debug,
    }
    try:
        with _client(timeout=15) as client:
            with console.status(f"Starting monitor for [cyan]{repo_path}[/cyan]…", spinner="dots"):
                resp = client.post("/api/v1/monitor/start", json=payload)
                data = resp.json()
    except httpx.ConnectError:
        _handle_connection_error()
        return

    if resp.status_code >= 400:
        console.print(f"[red]Failed: {data.get('error') or data.get('detail')}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[green]Now watching[/green] [cyan]{repo_path}[/cyan]\n"
            f"Poll interval: [bold]{interval}s[/bold]  ·  "
            f"Auto-debug: [bold]{'yes' if not no_auto_debug else 'no'}[/bold]",
            title="Monitor Started",
            border_style="green",
        )
    )


@cli.command()
@click.option("--port", default=8008, show_default=True, help="Server port")
def dashboard(port: int) -> None:
    """Open the web dashboard in the default browser."""
    url = f"http://localhost:{port}/dashboard"
    console.print(f"Opening dashboard at [link={url}]{url}[/link]")
    webbrowser.open(url)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: no args → interactive mode; otherwise route to the CLI."""
    args = sys.argv[1:]
    if not args:
        _interactive_main()
        return
    positional = [a for a in args if not a.startswith("-")]
    if positional and positional[0] not in _KNOWN_SUBCOMMANDS:
        sys.argv.insert(1, "run")
    cli()


if __name__ == "__main__":
    main()
