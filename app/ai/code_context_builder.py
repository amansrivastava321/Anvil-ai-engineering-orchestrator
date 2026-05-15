"""Code context builder — assembles ~1,200-token focused context for any problem.

Pipeline:
  1. Semantic search (bge-m3 via CodeIndexer)   → top 5 most relevant functions
  2. Graphify expansion                          → direct dependencies of those functions
  3. AST compression                             → full body for suspects, signatures for deps
  4. Assembly                                    → structured text under token budget

Falls back gracefully when no code index exists (skips step 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4


def _tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class FocusedContext:
    code_snippets: List[str]        # formatted code blocks
    dependency_graph: str           # who calls what (text)
    graph_summary: str              # module structure from Graphify
    total_tokens: int               # total token estimate


class CodeContextBuilder:
    """Builds minimal, focused context for AI models from a code repository."""

    async def build_context(
        self,
        problem: str,
        repo_path: str,
        max_tokens: int = 1200,
    ) -> FocusedContext:
        """Full pipeline: search → expand → compress → assemble."""
        from app.tools.file_system.file_chunker import FileChunker

        graph_query = self._load_graph(repo_path)
        graph_summary = self._get_graph_summary(graph_query) if graph_query else ""

        # Step 1: semantic search (function level)
        search_results = await self._semantic_search(problem, repo_path)

        if not search_results:
            # No index yet — return empty context (caller will fall back)
            return FocusedContext(
                code_snippets=[],
                dependency_graph="",
                graph_summary=graph_summary,
                total_tokens=_tokens(graph_summary),
            )

        # Step 2: Graphify expansion — add deps as signature-only entries
        expanded = await self._graphify_expand(search_results, graph_query)

        # Step 3: compress and read code
        chunker = FileChunker()
        snippets, dep_graph = await self._compress_and_read(
            problem, search_results, expanded, chunker, repo_path, max_tokens
        )

        total = _tokens("\n".join(snippets)) + _tokens(dep_graph) + _tokens(graph_summary)

        logger.info(
            "Focused context assembled",
            tokens=total,
            snippets=len(snippets),
            search_hits=len(search_results),
        )

        return FocusedContext(
            code_snippets=snippets,
            dependency_graph=dep_graph,
            graph_summary=graph_summary,
            total_tokens=total,
        )

    # ── Pipeline steps ────────────────────────────────────────────────────────

    async def _semantic_search(self, problem: str, repo_path: str):
        """Return top-5 SearchResult via CodeIndexer (empty list if no index)."""
        from app.integrations.code_indexer import CodeIndexer

        indexer = CodeIndexer()
        if not indexer.has_index(repo_path):
            return []
        try:
            return await indexer.search(problem, repo_path, top_k=5)
        except Exception as e:
            logger.debug("Semantic search failed", error=str(e))
            return []

    async def _graphify_expand(
        self, search_results: list, graph_query: Optional[Any]
    ) -> Dict[str, str]:
        """For each found file, get its dependencies from Graphify.

        Returns {dep_file_path: reason_string} for files not already in search_results.
        """
        if graph_query is None:
            return {}

        primary_files = {r.file_path for r in search_results}
        deps: Dict[str, str] = {}

        for result in search_results:
            try:
                dep_files = graph_query.get_dependencies(result.file_path)
                for dep in dep_files:
                    if dep not in primary_files and dep not in deps:
                        deps[dep] = f"dependency of {result.function_name}()"
            except Exception:
                pass

        return deps

    async def _compress_and_read(
        self,
        problem: str,
        search_results: list,
        expanded_deps: Dict[str, str],
        chunker: Any,
        repo_path: str,
        max_tokens: int,
    ):
        """Build snippets list and dependency graph text.

        Compression tiers:
          #0 (top match)  — full function body
          #1-4 (matches)  — skeleton (signatures + docstrings)
          deps            — one-line reference only
        """
        snippets: List[str] = []
        dep_lines: List[str] = []
        used_tokens = 0
        budget = max_tokens

        for i, result in enumerate(search_results):
            if used_tokens >= budget:
                break
            content = self._read_file(result.file_path, repo_path)
            if content is None:
                continue

            if i == 0:
                # Primary suspect — full function body
                code = chunker.extract_function(content, result.function_name)
                label = "primary suspect"
            else:
                # Secondary matches — skeleton only
                code = chunker.extract_skeleton(result.file_path, content)
                label = f"similarity {result.similarity:.2f}"

            snippet_tokens = _tokens(code)
            if used_tokens + snippet_tokens > budget:
                # Truncate to fit
                max_chars = (budget - used_tokens) * _CHARS_PER_TOKEN
                code = code[:max_chars] + "\n# ... [truncated]"
                snippet_tokens = _tokens(code)

            snippets.append(
                f"# {result.file_path}  [{label}]\n"
                f"# similarity: {result.similarity:.2f} | lines {result.start_line}-{result.end_line}\n"
                + code
            )
            used_tokens += snippet_tokens

            # Build dependency graph lines for this result
            try:
                graph_query = self._load_graph(repo_path)
                if graph_query:
                    deps = graph_query.get_dependencies(result.file_path)
                    for dep in deps[:3]:
                        dep_lines.append(f"  {result.function_name}() → {Path(dep).name}")
            except Exception:
                pass

        # Add compressed dependency file references
        dep_remaining = max(0, budget - used_tokens - 100)
        dep_snippet_tokens = 0
        for dep_file, reason in list(expanded_deps.items())[:5]:
            ref = f"# {dep_file}  [{reason}]"
            if dep_snippet_tokens + _tokens(ref) > dep_remaining:
                break
            snippets.append(ref)
            dep_snippet_tokens += _tokens(ref)

        dep_graph = "Dependencies (via Graphify):\n" + "\n".join(dep_lines) if dep_lines else ""
        return snippets, dep_graph

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_graph(self, repo_path: str) -> Optional[Any]:
        """Load GraphQuery for repo_path, or None if graph.json not found."""
        try:
            from app.integrations.graphify.query import get_graph_query

            graph_json = Path(repo_path) / "graphify-out" / "graph.json"
            if not graph_json.exists():
                return None
            gq = get_graph_query(graph_json)
            gq.load()
            return gq
        except Exception:
            return None

    def _get_graph_summary(self, graph_query: Any) -> str:
        try:
            return graph_query.get_repo_summary()
        except Exception:
            return ""

    def _read_file(self, file_path: str, repo_path: str) -> Optional[str]:
        """Read file content; try absolute path then relative to repo_path."""
        for p in (Path(file_path), Path(repo_path) / file_path):
            try:
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        return None
