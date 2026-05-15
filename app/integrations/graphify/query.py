"""
Lightweight query layer over graph.json produced by graphify.

Reads the graph once, builds an in-memory index, and answers structured
questions in ~300 tokens instead of reading the raw 50 000-token codebase.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── purpose keywords used by get_files_by_purpose ────────────────────────────
_PURPOSE_KEYWORDS: Dict[str, List[str]] = {
    "security": ["auth", "security", "encrypt", "validate", "sanitize", "token", "password", "permission", "rbac"],
    "performance": ["cache", "queue", "pool", "async", "rate", "timeout", "buffer", "batch"],
    "testing": ["test", "mock", "fixture", "assert", "spec", "coverage"],
    "api": ["router", "endpoint", "route", "handler", "controller", "rest", "http", "fastapi", "flask"],
    "data": ["model", "schema", "database", "repo", "repository", "migration", "orm", "store"],
    "monitoring": ["log", "metric", "trace", "monitor", "alert", "health", "status"],
    "config": ["config", "setting", "env", "environment"],
}

# ── extension → language ──────────────────────────────────────────────────────
_EXT_LANG: Dict[str, str] = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".rb": "Ruby",
}


class GraphQuery:
    """Query interface over a graphify graph.json file."""

    def __init__(self, graph_path: str | Path) -> None:
        self._path = Path(graph_path)
        self._graph: Dict[str, Any] = {}
        self._nodes: List[Dict] = []
        self._links: List[Dict] = []
        self._node_map: Dict[str, Dict] = {}          # id → node
        self._file_nodes: Dict[str, List[Dict]] = defaultdict(list)  # source_file → [nodes]
        self._out_edges: Dict[str, List[Dict]] = defaultdict(list)   # source → [links]
        self._in_edges: Dict[str, List[Dict]] = defaultdict(list)    # target → [links]
        self._degree: Dict[str, int] = {}
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            raise FileNotFoundError(f"graph.json not found at {self._path}")

        with open(self._path, encoding="utf-8") as f:
            self._graph = json.load(f)

        self._nodes = self._graph.get("nodes", [])
        self._links = self._graph.get("links", [])

        for n in self._nodes:
            nid = n.get("id", "")
            self._node_map[nid] = n
            src = n.get("source_file", "")
            if src:
                self._file_nodes[src].append(n)

        src_cnt: Counter = Counter()
        tgt_cnt: Counter = Counter()
        for lnk in self._links:
            s = lnk.get("source", "")
            t = lnk.get("target", "")
            if s:
                self._out_edges[s].append(lnk)
                src_cnt[s] += 1
            if t:
                self._in_edges[t].append(lnk)
                tgt_cnt[t] += 1

        for nid in self._node_map:
            self._degree[nid] = src_cnt.get(nid, 0) + tgt_cnt.get(nid, 0)

        self._loaded = True

    # ── Public API ────────────────────────────────────────────────────────────

    def get_repo_summary(self, repo_path: Optional[str] = None) -> str:
        """Return a ~300-token plain-text summary of the repository.

        If repo_path is given, filters nodes whose source_file starts with it.
        """
        self.load()

        all_files = self._collect_source_files(repo_path)
        lang_counts = self._detect_languages(all_files)
        god_nodes = self._top_nodes(repo_path, n=5)
        communities = self._community_count(repo_path)
        entry_points = self._find_entry_points(repo_path)

        lines = [
            f"Files: {len(all_files)}  Nodes: {len(self._nodes)}  Edges: {len(self._links)}",
            f"Languages: {', '.join(f'{v} {k}' for k, v in lang_counts.most_common(5))}",
            f"Communities (modules): {communities}",
            "",
            "God nodes (highest connectivity):",
        ]
        for label, deg, src_file in god_nodes:
            lines.append(f"  • {label} ({src_file}) — {deg} edges")

        if entry_points:
            lines.append("")
            lines.append("Entry points:")
            for ep in entry_points[:5]:
                lines.append(f"  • {ep}")

        return "\n".join(lines)

    def get_files_by_purpose(self, purpose: str, repo_path: Optional[str] = None) -> List[str]:
        """Return source files likely serving the given purpose.

        purpose: one of security, performance, testing, api, data, monitoring, config
        """
        self.load()
        keywords = _PURPOSE_KEYWORDS.get(purpose.lower(), [purpose.lower()])
        matched: Dict[str, int] = {}

        for node in self._nodes:
            src = node.get("source_file", "")
            if repo_path and not src.startswith(repo_path):
                continue
            label = (node.get("label") or "").lower()
            score = sum(1 for kw in keywords if kw in label or kw in src.lower())
            if score > 0 and src:
                matched[src] = matched.get(src, 0) + score

        return sorted(matched, key=lambda f: -matched[f])

    def get_dependencies(self, file_path: str) -> List[str]:
        """Return all files that file_path imports or calls (flat, deduped)."""
        self.load()
        node_ids = {n["id"] for n in self._file_nodes.get(file_path, [])}
        seen: Dict[str, bool] = {}
        for nid in node_ids:
            for lnk in self._out_edges.get(nid, []):
                tgt = self._node_map.get(lnk.get("target", ""), {})
                tgt_file = tgt.get("source_file", "")
                if tgt_file and tgt_file != file_path:
                    seen[tgt_file] = True
        return list(seen)

    def get_dependents(self, file_path: str) -> List[str]:
        """Return source files that import/call into file_path (inbound edges)."""
        self.load()
        node_ids = {n["id"] for n in self._file_nodes.get(file_path, [])}
        dependents: Dict[str, int] = {}
        for nid in node_ids:
            for lnk in self._in_edges.get(nid, []):
                src_node = self._node_map.get(lnk.get("source", ""), {})
                src_file = src_node.get("source_file", "")
                if src_file and src_file != file_path:
                    dependents[src_file] = dependents.get(src_file, 0) + 1
        return sorted(dependents, key=lambda f: -dependents[f])

    def get_module_boundaries(self, repo_path: Optional[str] = None) -> Dict[int, List[str]]:
        """Return community_id → [source_files] grouping (logical module boundaries)."""
        self.load()
        communities: Dict[int, Dict[str, int]] = defaultdict(dict)

        for node in self._nodes:
            src = node.get("source_file", "")
            if repo_path and not src.startswith(repo_path):
                continue
            if not src:
                continue
            c = node.get("community", -1)
            communities[c][src] = communities[c].get(src, 0) + 1

        return {c: sorted(files.keys()) for c, files in communities.items()}

    def get_risk_analysis(self, repo_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return highest-risk files (most connected + security-sensitive labels)."""
        self.load()
        file_degree: Dict[str, int] = defaultdict(int)
        file_security: Dict[str, int] = defaultdict(int)
        security_kw = _PURPOSE_KEYWORDS["security"]

        for node in self._nodes:
            src = node.get("source_file", "")
            if repo_path and not src.startswith(repo_path):
                continue
            if not src:
                continue
            nid = node.get("id", "")
            deg = self._degree.get(nid, 0)
            file_degree[src] += deg
            label = (node.get("label") or "").lower()
            sec = sum(1 for kw in security_kw if kw in label or kw in src.lower())
            file_security[src] += sec

        all_files = set(file_degree.keys()) | set(file_security.keys())
        risk_items = []
        for f in all_files:
            deg = file_degree.get(f, 0)
            sec = file_security.get(f, 0)
            risk_score = deg + sec * 20  # security hits count more
            risk_items.append({
                "file": f,
                "degree": deg,
                "security_signals": sec,
                "risk_score": risk_score,
                "risk_level": "high" if risk_score > 200 else "medium" if risk_score > 80 else "low",
            })

        return sorted(risk_items, key=lambda x: -x["risk_score"])[:20]

    def get_all_source_files(self, repo_path: Optional[str] = None) -> List[str]:
        """Return all unique source files in the graph."""
        self.load()
        return list(self._collect_source_files(repo_path))

    @property
    def total_files(self) -> int:
        """Total unique source files in this graph."""
        self.load()
        return len(self._collect_source_files(None))

    def get_god_nodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Most connected components — changes here ripple furthest."""
        self.load()
        top = self._top_nodes(None, n=limit)
        return [
            {
                "label": label,
                "source_file": src_file,
                "degree": deg,
                "impact": "critical" if deg > 100 else "high" if deg > 50 else "medium",
            }
            for label, deg, src_file in top
        ]

    def get_recently_changed(self, repo_path: str, days: int = 7) -> List[str]:
        """Files changed in the last N days (git log). Returns graph-known paths only."""
        import subprocess
        self.load()
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "log",
                 f"--since={days} days ago", "--name-only", "--format="],
                capture_output=True, text=True, timeout=15,
            )
            seen: Dict[str, bool] = {}
            for rel in result.stdout.splitlines():
                rel = rel.strip()
                if not rel:
                    continue
                # Try absolute path first, then relative path as-is
                abs_p = str(Path(repo_path) / rel)
                key = abs_p if abs_p in self._file_nodes else (rel if rel in self._file_nodes else None)
                if key:
                    seen[key] = True
            return list(seen)
        except Exception:
            return []

    def get_hotspots(self, repo_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Files most frequently modified in git history — likely where bugs live."""
        import subprocess
        self.load()
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "log", "--format=", "--name-only"],
                capture_output=True, text=True, timeout=30,
            )
            counts: Counter = Counter()
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    counts[line] += 1
            hotspots = []
            for rel_path, count in counts.most_common(limit * 3):
                abs_p = str(Path(repo_path) / rel_path)
                matched = (
                    abs_p if abs_p in self._file_nodes
                    else rel_path if rel_path in self._file_nodes
                    else None
                )
                if matched:
                    hotspots.append({
                        "file": matched,
                        "change_count": count,
                        "risk": "high" if count > 20 else "medium" if count > 10 else "low",
                    })
                if len(hotspots) >= limit:
                    break
            return hotspots
        except Exception:
            return []

    async def get_related_files(
        self, task_description: str, repo_path: Optional[str] = None
    ) -> List[str]:
        """Semantic file search via bge-m3. Falls back to keyword overlap."""
        self.load()
        try:
            from app.integrations.ollama.embeddings import cosine_similarity, get_embedding

            task_emb = await get_embedding(task_description)

            # Aggregate labels per file; use stored embeddings where available
            file_labels: Dict[str, List[str]] = defaultdict(list)
            file_emb: Dict[str, List[float]] = {}

            for node in self._nodes:
                src = node.get("source_file", "")
                if repo_path and not src.startswith(repo_path):
                    continue
                if not src:
                    continue
                label = node.get("label", "")
                if label:
                    file_labels[src].append(label)
                stored = node.get("embedding")
                if stored and src not in file_emb:
                    file_emb[src] = stored

            scores: Dict[str, float] = {}
            task_words = {w for w in task_description.lower().split() if len(w) > 3}

            for src_file, labels in file_labels.items():
                if src_file in file_emb:
                    scores[src_file] = cosine_similarity(task_emb, file_emb[src_file])
                else:
                    # Keyword overlap as cheap proxy
                    text = " ".join(labels).lower() + " " + src_file.lower()
                    overlap = sum(1 for w in task_words if w in text)
                    scores[src_file] = min(0.5, overlap / max(len(task_words), 1) * 0.5)

            return sorted(scores, key=lambda f: -scores[f])[:10]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Semantic file search failed: %s", e)
            return []

    def get_impact_radius(self, file_path: str) -> Dict[str, Any]:
        """Everything that could break if file_path changes."""
        self.load()
        direct = self.get_dependents(file_path)

        # 2-hop traversal (capped to avoid O(n²))
        indirect: Dict[str, int] = {}
        for dep in direct[:15]:
            for second in self.get_dependents(dep):
                if second != file_path and second not in direct:
                    indirect[second] = indirect.get(second, 0) + 1

        god_set = {g["source_file"] for g in self.get_god_nodes(10)}
        affected_gods = [f for f in direct if f in god_set]

        impact = (
            "critical" if affected_gods or len(direct) > 10
            else "high" if len(direct) > 5
            else "medium" if direct
            else "low"
        )

        return {
            "file": file_path,
            "direct_dependents": direct[:10],
            "indirect_dependents": sorted(indirect, key=lambda f: -indirect[f])[:10],
            "affected_god_nodes": affected_gods,
            "impact_level": impact,
            "total_affected": len(direct) + len(indirect),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_source_files(self, repo_path: Optional[str]) -> List[str]:
        seen: Dict[str, bool] = {}
        for node in self._nodes:
            src = node.get("source_file", "")
            if src and (not repo_path or src.startswith(repo_path)):
                seen[src] = True
        return list(seen)

    def _detect_languages(self, files: List[str]) -> Counter:
        lang: Counter = Counter()
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in _EXT_LANG:
                lang[_EXT_LANG[ext]] += 1
        return lang

    def _top_nodes(
        self, repo_path: Optional[str], n: int = 5
    ) -> List[Tuple[str, int, str]]:
        """Return top-n (label, degree, source_file) tuples by degree."""
        candidates = []
        for node in self._nodes:
            src = node.get("source_file", "")
            if repo_path and not src.startswith(repo_path):
                continue
            nid = node.get("id", "")
            deg = self._degree.get(nid, 0)
            label = node.get("label") or nid
            candidates.append((label, deg, src))
        return sorted(candidates, key=lambda x: -x[1])[:n]

    def _community_count(self, repo_path: Optional[str]) -> int:
        communities = set()
        for node in self._nodes:
            src = node.get("source_file", "")
            if repo_path and not src.startswith(repo_path):
                continue
            c = node.get("community")
            if c is not None:
                communities.add(c)
        return len(communities)

    def _find_entry_points(self, repo_path: Optional[str]) -> List[str]:
        """Files with high out-degree but low in-degree (nobody imports them = entry points)."""
        self.load()
        file_in: Dict[str, int] = defaultdict(int)
        file_out: Dict[str, int] = defaultdict(int)

        for lnk in self._links:
            src_node = self._node_map.get(lnk.get("source", ""), {})
            tgt_node = self._node_map.get(lnk.get("target", ""), {})
            src_file = src_node.get("source_file", "")
            tgt_file = tgt_node.get("source_file", "")
            if repo_path:
                if src_file and not src_file.startswith(repo_path):
                    src_file = ""
                if tgt_file and not tgt_file.startswith(repo_path):
                    tgt_file = ""
            if src_file:
                file_out[src_file] += 1
            if tgt_file:
                file_in[tgt_file] += 1

        all_files = set(file_out.keys()) | set(file_in.keys())
        scored = []
        for f in all_files:
            out = file_out.get(f, 0)
            inp = file_in.get(f, 0)
            if out > 0 and inp == 0:
                scored.append((f, out))
        return [f for f, _ in sorted(scored, key=lambda x: -x[1])]


# ── Process-wide instances (keyed by graph path) ─────────────────────────────

_instances: Dict[str, GraphQuery] = {}


def get_graph_query(graph_path: str | Path) -> GraphQuery:
    """Return a cached GraphQuery for the given graph.json path."""
    key = str(graph_path)
    if key not in _instances:
        _instances[key] = GraphQuery(graph_path)
    return _instances[key]
