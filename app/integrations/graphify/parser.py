"""
Graphify integration wrapper for the open-source graphifyy package.

This module wraps the graphify CLI tool and parses its standard outputs:
- graph.json (queryable NetworkX graph)
- GRAPH_REPORT.md (human-readable audit)
- graph.html (interactive visualization)

Uses the actual graphifyy package: https://github.com/safi-shamsi/graphify
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class GraphifyWrapper:
    """Wraps the graphify CLI for use within the orchestrator."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.output_dir = self.repo_path / "graphify-out"
        
    def is_installed(self) -> bool:
        """Check if graphify CLI is available."""
        try:
            subprocess.run(["graphify", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def has_output(self) -> bool:
        """Check if graphify has already been run on this repo."""
        return (self.output_dir / "graph.json").exists()
    
    async def run_graphify(self, force: bool = False) -> Dict[str, Any]:
        """Execute graphify on the repository."""
        if not force and self.has_output():
            logger.info("Graphify output already exists, skipping build")
            return await self.load_output()
        
        logger.info(f"Running graphify on {self.repo_path}")
        
        # Run graphify as subprocess
        process = await asyncio.create_subprocess_exec(
            "graphify", str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Graphify failed: {stderr.decode()}")
        
        return await self.load_output()
    
    async def load_output(self) -> Dict[str, Any]:
        """Load existing graphify output."""
        graph_file = self.output_dir / "graph.json"
        report_file = self.output_dir / "GRAPH_REPORT.md"
        
        result = {"available": False}
        
        if graph_file.exists():
            with open(graph_file) as f:
                result["graph"] = json.load(f)
            result["available"] = True
        
        if report_file.exists():
            result["report"] = report_file.read_text()
        
        return result
    
    async def query_graph(self, query: str) -> List[Dict]:
        """Run a graphify query command."""
        process = await asyncio.create_subprocess_exec(
            "graphify", "query", query,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        return json.loads(stdout.decode()) if stdout else []

class GraphifyParser:
    """High-level graphify parser — takes repo_path per call rather than at init.

    Used by ContextAssembler to parse repositories on demand without
    creating a new GraphifyWrapper per orchestrator instance.
    """

    async def parse_repository(self, repo_path: str) -> Dict[str, Any]:
        """Parse a repository and return its graphify output dict."""
        wrapper = GraphifyWrapper(repo_path)
        try:
            return await wrapper.run_graphify()
        except Exception as exc:
            logger.warning("Graphify parse failed", repo_path=repo_path, error=str(exc))
            return {"available": False}

    async def get_affected_modules(
        self, repo_path: str, target_files: list[str]
    ) -> dict[str, Any]:
        """Return modules affected by changes to target_files."""
        try:
            result = await self.parse_repository(repo_path)
            if not result.get("available", True):
                return {"available": False, "affected": []}
            # Extract modules that import the target files
            affected = []
            modules = result.get("modules", {})
            target_set = set(target_files)
            for mod_name, mod_data in modules.items():
                deps = mod_data.get("imports", []) + mod_data.get("dependencies", [])
                if any(t in dep for t in target_set for dep in deps):
                    affected.append(mod_name)
            return {"available": True, "affected": affected, "target_files": target_files}
        except Exception as exc:
            return {"available": False, "error": str(exc), "affected": []}

    async def get_module_dependencies(
        self, repo_path: str, module_name: str
    ) -> dict[str, Any]:
        """Return dependency graph for a specific module."""
        try:
            result = await self.parse_repository(repo_path)
            if not result.get("available", True):
                return {"available": False, "module": module_name, "dependencies": []}
            modules = result.get("modules", {})
            mod_data = modules.get(module_name, {})
            return {
                "available": True,
                "module": module_name,
                "imports": mod_data.get("imports", []),
                "dependencies": mod_data.get("dependencies", []),
                "dependents": [
                    m for m, d in modules.items()
                    if module_name in d.get("imports", []) + d.get("dependencies", [])
                ],
            }
        except Exception as exc:
            return {"available": False, "error": str(exc), "module": module_name}

    def clear_cache(self) -> None:
        """Clear any cached parse results."""
        # GraphifyWrapper runs fresh each time; no in-process cache to clear
        pass

    def get_stats(self) -> dict[str, Any]:
        """Return parser statistics."""
        return {
            "parser": "GraphifyParser",
            "backend": "graphify-cli",
            "cache": "none",
        }

    async def close(self) -> None:
        """No persistent resources to release."""


_default_parser: Optional[GraphifyParser] = None


def get_default_parser() -> GraphifyParser:
    """Return the process-wide GraphifyParser singleton."""
    global _default_parser
    if _default_parser is None:
        _default_parser = GraphifyParser()
    return _default_parser
