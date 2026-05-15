"""Code indexer — function-level semantic index using bge-m3.

Walks a repository, extracts every function and class, creates bge-m3
embeddings for each, and stores them for fast similarity search.

Storage layout:
  data/code_indexes/{repo_hash}.json   ← metadata (file paths, names, lines)
  data/code_indexes/{repo_hash}.npy    ← embedding vectors (float32 numpy array)

One entry per function/class. Search is O(n) cosine similarity over the vector array.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.monitoring.logging import get_logger

logger = get_logger(__name__)

_INDEX_DIR = Path("data/code_indexes")

_SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cs"}

_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


@dataclass
class SearchResult:
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    similarity: float
    func_type: str  # "function" | "class"
    class_context: Optional[str] = None


class CodeIndexer:
    """Indexes all functions/classes in a repository for bge-m3 semantic search."""

    def __init__(self, index_dir: Path = _INDEX_DIR) -> None:
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def has_index(self, repo_path: str) -> bool:
        meta, vecs = self._index_paths(repo_path)
        return meta.exists() and vecs.exists()

    async def index_repository(
        self,
        repo_path: str,
        graph_query: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Extract all functions, embed them, persist index. Returns stats dict."""
        from app.integrations.ollama.embeddings import get_embedding

        meta_path, vecs_path = self._index_paths(repo_path)

        source_files = self._discover_files(repo_path, graph_query)
        logger.info("Indexing repository", files=len(source_files), repo=repo_path)

        all_funcs: List[Dict] = []
        for file_path in source_files:
            try:
                p = Path(file_path)
                if not p.exists():
                    p = Path(repo_path) / file_path
                if not p.exists():
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                funcs = self._extract_functions(str(p.resolve()), content)
                all_funcs.extend(funcs)
            except Exception as e:
                logger.debug("Skipping file during indexing", path=file_path, error=str(e))

        if not all_funcs:
            return {"total_functions": 0, "total_files": len(source_files), "indexed": False}

        # Build embeddings — one bge-m3 call per function
        embeddings: List[List[float]] = []
        fallback_dim = 1024  # bge-m3 output dimension
        for func in all_funcs:
            text = self._create_text_representation(func)
            try:
                emb = await get_embedding(text)
                embeddings.append(emb)
                if not fallback_dim and emb:
                    fallback_dim = len(emb)
            except Exception as e:
                logger.debug("Embedding failed for function", name=func.get("name"), error=str(e))
                embeddings.append([0.0] * fallback_dim)

        # Persist
        meta_path.write_text(json.dumps(all_funcs, indent=2, ensure_ascii=False))
        arr = np.array(embeddings, dtype=np.float32)
        np.save(str(vecs_path), arr)

        logger.info(
            "Repository indexed",
            functions=len(all_funcs),
            files=len(source_files),
            index=str(meta_path),
        )
        return {
            "total_functions": len(all_funcs),
            "total_files": len(source_files),
            "indexed": True,
        }

    async def search(
        self, query: str, repo_path: str, top_k: int = 5
    ) -> List[SearchResult]:
        """Semantic search over indexed functions. Returns [] if no index exists."""
        from app.integrations.ollama.embeddings import get_embedding

        meta_path, vecs_path = self._index_paths(repo_path)
        if not meta_path.exists() or not vecs_path.exists():
            return []

        try:
            meta = json.loads(meta_path.read_text())
            vecs = np.load(str(vecs_path))
        except Exception as e:
            logger.warning("Index load failed", repo=repo_path, error=str(e))
            return []

        if not meta or vecs.shape[0] == 0:
            return []

        query_emb = np.array(await get_embedding(query), dtype=np.float32)

        # Batch cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vecs / norms
        q_norm = query_emb / (float(np.linalg.norm(query_emb)) or 1.0)
        sims = normalized @ q_norm

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if int(idx) >= len(meta):
                continue
            m = meta[int(idx)]
            results.append(SearchResult(
                file_path=m.get("file_path", ""),
                function_name=m.get("name", ""),
                start_line=m.get("line_start", 1),
                end_line=m.get("line_end", 1),
                signature=m.get("signature", ""),
                docstring=m.get("docstring", ""),
                similarity=float(sims[int(idx)]),
                func_type=m.get("type", "function"),
                class_context=m.get("class_context"),
            ))
        return results

    # ── File discovery ────────────────────────────────────────────────────────

    def _discover_files(self, repo_path: str, graph_query: Optional[Any]) -> List[str]:
        """Get source files from Graphify if available, else filesystem walk."""
        if graph_query is not None:
            try:
                files = graph_query.get_all_source_files()
                if files:
                    return files
            except Exception:
                pass

        # Filesystem walk fallback
        root = Path(repo_path)
        files: List[str] = []
        for p in root.rglob("*"):
            if any(part in _IGNORE_DIRS for part in p.parts):
                continue
            if p.suffix.lower() in _SOURCE_EXTENSIONS:
                files.append(str(p))
        return files[:500]  # cap to avoid massive repos

    # ── Extraction ────────────────────────────────────────────────────────────

    def _extract_functions(self, file_path: str, content: str) -> List[Dict]:
        """Extract all functions and classes from a source file."""
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return self._extract_python(file_path, content)
        return self._extract_regex(file_path, content)

    def _extract_python(self, file_path: str, content: str) -> List[Dict]:
        results: List[Dict] = []
        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    results.append(self._func_dict(node, file_path, class_ctx=None))
                elif isinstance(node, ast.ClassDef):
                    results.append(self._class_dict(node, file_path))
                    for item in ast.iter_child_nodes(node):
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            results.append(self._func_dict(item, file_path, class_ctx=node.name))
        except SyntaxError:
            pass
        return results

    def _func_dict(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str, class_ctx: Optional[str]
    ) -> Dict:
        args = [a.arg for a in node.args.args]
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {node.name}({', '.join(args)})"
        if class_ctx:
            sig = f"[{class_ctx}] {sig}"
        doc = ast.get_docstring(node) or ""
        return {
            "file_path": file_path,
            "name": node.name,
            "type": "function",
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno + 5),
            "signature": sig,
            "docstring": doc.splitlines()[0] if doc else "",
            "class_context": class_ctx,
        }

    def _class_dict(self, node: ast.ClassDef, file_path: str) -> Dict:
        doc = ast.get_docstring(node) or ""
        return {
            "file_path": file_path,
            "name": node.name,
            "type": "class",
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno + 5),
            "signature": f"class {node.name}",
            "docstring": doc.splitlines()[0] if doc else "",
            "class_context": None,
        }

    _DEF_RE = re.compile(
        r"^[ \t]*((?:export\s+)?(?:async\s+)?(?:function|func|fn|def|sub)\s+(\w+)\s*\([^)]{0,120}\)[^{;\n]*)",
        re.MULTILINE,
    )

    def _extract_regex(self, file_path: str, content: str) -> List[Dict]:
        results: List[Dict] = []
        lines = content.count("\n") + 1
        for m in self._DEF_RE.finditer(content):
            name = m.group(2)
            start = content[: m.start()].count("\n") + 1
            results.append({
                "file_path": file_path,
                "name": name,
                "type": "function",
                "line_start": start,
                "line_end": min(start + 40, lines),
                "signature": m.group(1).strip(),
                "docstring": "",
                "class_context": None,
            })
        return results[:60]  # cap per file

    def _create_text_representation(self, func: Dict) -> str:
        """Searchable text embedding the signature + first docstring line."""
        parts = []
        if func.get("class_context"):
            parts.append(f"class {func['class_context']}:")
        parts.append(func.get("signature", func["name"]))
        doc = func.get("docstring", "")
        if doc:
            parts.append(f'    """{doc}"""')
        return "\n".join(parts)

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _repo_hash(self, repo_path: str) -> str:
        return hashlib.sha1(str(Path(repo_path).resolve()).encode()).hexdigest()[:16]

    def _index_paths(self, repo_path: str) -> tuple[Path, Path]:
        h = self._repo_hash(repo_path)
        return self._index_dir / f"{h}.json", self._index_dir / f"{h}.npy"
