"""
Automatic repository onboarding service — with persistent memory.

Pipeline for NEW repos:
  1. Run graphify (builds graph.json)
  2. Query graph for structural intelligence (~300 tokens)
  3. CEO AI analysis using graph summary
  4. Save HealthSnapshot to RepoStateStore
  5. Return full OnboardingReport

For KNOWN repos:
  - Returns current state + detected changes immediately (no AI, ~0 tokens)
  - Quick-scan runs only on changed files (~500 tokens vs 3,000)
  - Delta report shows only what changed since last analysis
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.monitoring.logging import get_logger
from app.core.repo_state import (
    HealthSnapshot,
    RepoState,
    RepoStateStore,
    get_repo_state_store,
)
from app.integrations.graphify.parser import GraphifyWrapper
from app.integrations.graphify.query import GraphQuery, get_graph_query

logger = get_logger(__name__)


# ── Report models ─────────────────────────────────────────────────────────────


class IssueItem(BaseModel):
    title: str
    description: str
    file_path: Optional[str] = None
    line: Optional[int] = None
    fix_suggestion: Optional[str] = None
    estimated_time: Optional[str] = None
    priority: str = "medium"  # critical | high | medium | low


class OnboardingReport(BaseModel):
    repo_path: str
    project_name: str
    project_type: str
    language: str
    framework: Optional[str] = None
    health_score: int
    status: str = "new_analysis"   # new_analysis | re_analysis
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    graph_available: bool = False
    total_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    communities: int = 0
    god_nodes: List[str] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    risk_files: List[str] = Field(default_factory=list)

    executive_summary: str = ""
    action_plan: str = ""

    critical: List[IssueItem] = Field(default_factory=list)
    high: List[IssueItem] = Field(default_factory=list)
    medium: List[IssueItem] = Field(default_factory=list)
    low: List[IssueItem] = Field(default_factory=list)

    # Delta fields (populated on re-analysis)
    issues_fixed_since: int = 0
    new_issues_since: int = 0
    health_delta: int = 0         # positive = improved

    duration_ms: float = 0.0


class QuickScanReport(BaseModel):
    repo_path: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    changed_files: List[str] = Field(default_factory=list)
    new_commits: int = 0
    impact_summary: str = ""
    affected_modules: List[str] = Field(default_factory=list)
    risk_level: str = "low"
    recommendations: List[str] = Field(default_factory=list)
    # Delta vs last analysis
    new_issues: int = 0
    fixed_issues: int = 0
    health_before: Optional[int] = None
    health_after: Optional[int] = None
    duration_ms: float = 0.0


# ── Service ───────────────────────────────────────────────────────────────────


class OnboardingService:
    """Orchestrates the repository onboarding and incremental scan pipelines."""

    def __init__(self) -> None:
        self._repo_store: RepoStateStore = get_repo_state_store()
        # In-memory cache for full reports (state is persisted; reports are not)
        self._reports: Dict[str, OnboardingReport] = {}

    # ── Smart entry point ─────────────────────────────────────────────────────

    async def analyze_repository(
        self, repo_path: str, force_full: bool = False
    ) -> Dict[str, Any]:
        """Smart analysis: checks state first, runs full pipeline only when needed.

        Returns dict with `status` key:
        - "new_analysis"        → first time, full report included
        - "previously_analyzed" → known repo, returns state + detected changes
        """
        path = Path(repo_path).resolve()
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        existing = self._repo_store.get_state(str(path))

        if existing and not force_full:
            changed_files = await self._repo_store.get_changed_files(str(path))
            new_commits = await self._repo_store.get_new_commit_count(str(path))
            return {
                "status": "previously_analyzed",
                "repo_name": existing.repo_name,
                "repo_path": existing.repo_path,
                "last_analyzed": existing.last_analyzed.isoformat(),
                "health_score": existing.current_health.health_score,
                "health_trend": existing.trend,
                "trend_arrow": existing.trend_arrow,
                "improvement_rate": existing.improvement_rate,
                "analysis_count": existing.analysis_count,
                "issues_found": existing.total_issues(),
                "critical_issues": existing.current_health.critical_issues,
                "high_issues": existing.current_health.high_issues,
                "issues_fixed_since": existing.current_health.fixed_issues_since_previous,
                "new_commits_detected": new_commits,
                "files_changed": changed_files,
                "options": ["re_analyze_full", "quick_scan", "view_last_report"],
                "last_report_id": self._report_id(str(path)),
            }

        # New repo or forced: run full pipeline
        report = await self._full_onboarding(str(path))
        return {"status": report.status, "report": report.model_dump(mode="json")}

    # ── Full onboarding ───────────────────────────────────────────────────────

    async def onboard_repository(
        self, repo_path: str, force_rebuild: bool = False
    ) -> OnboardingReport:
        """Full onboarding pipeline. Saves state. Returns a structured report."""
        path = Path(repo_path).resolve()
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        return await self._full_onboarding(str(path), force=force_rebuild)

    async def _full_onboarding(
        self, repo_path: str, force: bool = False
    ) -> OnboardingReport:
        start = time.monotonic()
        path = Path(repo_path)

        logger.info("Starting full onboarding", repo=repo_path)

        # Step 1 — Build or load graphify graph
        graph_data = await self._build_graph(repo_path, force=force)
        graph_available = graph_data.get("available", False)

        # Step 2 — Query graph
        graph_summary, god_nodes, entry_points, risk_files, total_files, \
            total_nodes, total_edges, communities = await self._query_graph(
                repo_path, graph_available
            )

        # Step 3 — CEO analysis
        exec_summary, action_plan, issues = await self._ceo_analysis(
            repo_path, graph_summary, risk_files
        )

        # Step 4 — Project metadata
        project_name, project_type, language, framework = self._detect_project_meta(repo_path)

        # Step 5 — Health score
        health_score = self._compute_health_score(total_files, communities, risk_files, issues)

        # Step 6 — Get git HEAD and save state
        commit_hash = await self._repo_store.get_current_commit(repo_path)
        snapshot = HealthSnapshot(
            health_score=health_score,
            critical_issues=sum(1 for i in issues if i.priority == "critical"),
            high_issues=sum(1 for i in issues if i.priority == "high"),
            medium_issues=sum(1 for i in issues if i.priority == "medium"),
            low_issues=sum(1 for i in issues if i.priority == "low"),
            total_files=total_files,
            total_modules=communities,
            commit_hash=commit_hash,
        )

        existing = self._repo_store.get_state(repo_path)
        prev_score = existing.current_health.health_score if existing else health_score
        saved_state = self._repo_store.save_state(repo_path, snapshot)

        duration_ms = (time.monotonic() - start) * 1000

        report = OnboardingReport(
            repo_path=repo_path,
            project_name=project_name,
            project_type=project_type,
            language=language,
            framework=framework,
            health_score=health_score,
            status="re_analysis" if existing else "new_analysis",
            graph_available=graph_available,
            total_files=total_files,
            total_nodes=total_nodes,
            total_edges=total_edges,
            communities=communities,
            god_nodes=god_nodes,
            entry_points=entry_points,
            risk_files=risk_files,
            executive_summary=exec_summary,
            action_plan=action_plan,
            critical=[i for i in issues if i.priority == "critical"],
            high=[i for i in issues if i.priority == "high"],
            medium=[i for i in issues if i.priority == "medium"],
            low=[i for i in issues if i.priority == "low"],
            issues_fixed_since=snapshot.fixed_issues_since_previous,
            new_issues_since=snapshot.new_issues_since_previous,
            health_delta=health_score - prev_score,
            duration_ms=duration_ms,
        )

        self._reports[repo_path] = report
        logger.info(
            "Onboarding complete",
            repo=project_name,
            health=health_score,
            delta=report.health_delta,
            duration_ms=f"{duration_ms:.0f}",
        )
        return report

    # ── Quick scan ─────────────────────────────────────────────────────────────

    async def quick_scan(self, repo_path: str) -> QuickScanReport:
        """Incremental scan: only changed files since last analysis."""
        start = time.monotonic()
        path = Path(repo_path).resolve()

        existing = self._repo_store.get_state(str(path))
        health_before = existing.current_health.health_score if existing else None

        # Files changed since last analyzed commit
        changed_files = await self._repo_store.get_changed_files(str(path))
        new_commits = await self._repo_store.get_new_commit_count(str(path))

        affected_modules: List[str] = []
        risk_level = "low"

        graph_json = path / "graphify-out" / "graph.json"
        if graph_json.exists() and changed_files:
            try:
                gq = get_graph_query(graph_json)
                gq.load()
                affected_set: set = set()
                for f in changed_files:
                    affected_set.update(gq.get_dependents(f)[:3])
                affected_modules = list(affected_set)[:10]

                risk_files_all = [r["file"] for r in gq.get_risk_analysis()[:10]]
                risky = [f for f in changed_files if f in risk_files_all]
                if len(risky) > 2 or len(changed_files) > 15:
                    risk_level = "high"
                elif risky or len(changed_files) > 5:
                    risk_level = "medium"
            except Exception as e:
                logger.warning("Graph query failed during quick scan", error=str(e))

        impact_summary, recommendations = await self._quick_scan_analysis(
            changed_files, affected_modules, risk_level
        )

        # Simple heuristic health delta for quick scan
        health_delta = 0
        if risk_level == "high":
            health_delta = -3
        elif risk_level == "low" and changed_files:
            health_delta = 1
        health_after = (health_before + health_delta) if health_before is not None else None

        # Update state snapshot if we have one
        new_issues = 0
        fixed_issues = 0
        if existing and changed_files:
            commit_hash = await self._repo_store.get_current_commit(str(path))
            snap = HealthSnapshot(
                health_score=health_after or existing.current_health.health_score,
                critical_issues=existing.current_health.critical_issues,
                high_issues=existing.current_health.high_issues,
                medium_issues=existing.current_health.medium_issues,
                low_issues=existing.current_health.low_issues,
                total_files=existing.current_health.total_files,
                total_modules=existing.current_health.total_modules,
                commit_hash=commit_hash,
                files_changed_since_previous=changed_files,
            )
            saved = self._repo_store.save_state(str(path), snap)
            new_issues = snap.new_issues_since_previous
            fixed_issues = snap.fixed_issues_since_previous

        duration_ms = (time.monotonic() - start) * 1000
        return QuickScanReport(
            repo_path=str(path),
            changed_files=changed_files,
            new_commits=new_commits,
            impact_summary=impact_summary,
            affected_modules=affected_modules,
            risk_level=risk_level,
            recommendations=recommendations,
            new_issues=new_issues,
            fixed_issues=fixed_issues,
            health_before=health_before,
            health_after=health_after,
            duration_ms=duration_ms,
        )

    # ── State accessors ───────────────────────────────────────────────────────

    def get_state(self, repo_path: str) -> Optional[RepoState]:
        return self._repo_store.get_state(str(Path(repo_path).resolve()))

    def list_repos(self) -> List[Dict[str, Any]]:
        """Return all repos as rich summary cards."""
        states = self._repo_store.get_all_states()
        cards = []
        for s in states:
            cards.append({
                "repo_name": s.repo_name,
                "repo_path": s.repo_path,
                "health_score": s.current_health.health_score,
                "trend": s.trend,
                "trend_arrow": s.trend_arrow,
                "improvement_rate": s.improvement_rate,
                "last_analyzed": s.last_analyzed.isoformat(),
                "days_since_analysis": s.days_since_analysis(),
                "analysis_count": s.analysis_count,
                "critical_issues": s.current_health.critical_issues,
                "high_issues": s.current_health.high_issues,
                "total_issues": s.total_issues(),
                "needs_attention": s.needs_attention,
                "graph_available": (
                    Path(s.repo_path) / "graphify-out" / "graph.json"
                ).exists(),
                "total_files": s.current_health.total_files,
            })
        return cards

    def get_report(self, repo_path: str) -> Optional[OnboardingReport]:
        return self._reports.get(str(Path(repo_path).resolve()))

    def get_weekly_summary(self) -> Dict[str, Any]:
        return self._repo_store.weekly_summary()

    # ── Pipeline helpers ──────────────────────────────────────────────────────

    async def _build_graph(self, repo_path: str, force: bool = False) -> Dict[str, Any]:
        try:
            wrapper = GraphifyWrapper(repo_path)
            return await wrapper.run_graphify(force=force)
        except Exception as e:
            logger.warning("Graphify build failed", error=str(e))
            return {"available": False}

    async def _query_graph(
        self, repo_path: str, graph_available: bool
    ) -> Tuple:
        if not graph_available:
            return "Graph analysis unavailable.", [], [], [], 0, 0, 0, 0

        graph_json = Path(repo_path) / "graphify-out" / "graph.json"
        try:
            gq = get_graph_query(graph_json)
            gq.load()
            summary = gq.get_repo_summary()
            all_files = gq.get_all_source_files()
            total_files = len(all_files)
            total_nodes = len(gq._nodes)
            total_edges = len(gq._links)
            communities = gq._community_count(None)
            god_nodes = [
                f"{lbl} ({src})"
                for lbl, _, src in gq._top_nodes(None, n=5)
            ]
            entry_points = gq._find_entry_points(None)[:5]
            risk_files = [
                r["file"]
                for r in gq.get_risk_analysis()[:5]
                if r["risk_level"] in ("high", "medium")
            ]
            return summary, god_nodes, entry_points, risk_files, \
                total_files, total_nodes, total_edges, communities
        except Exception as e:
            logger.warning("Graph query failed", error=str(e))
            return "Graph analysis unavailable.", [], [], [], 0, 0, 0, 0

    async def _ceo_analysis(
        self,
        repo_path: str,
        graph_summary: str,
        risk_files: List[str],
    ) -> Tuple[str, str, List[IssueItem]]:
        try:
            from app.ai.ceo import get_ceo

            ceo = get_ceo()
            decision = await ceo.receive_problem(
                description=(
                    f"Perform a comprehensive audit of the repository at {repo_path}. "
                    "Using the graph summary provided in context, identify: "
                    "1) Critical security issues, 2) High-priority architectural problems, "
                    "3) Performance bottlenecks, 4) Testing gaps, "
                    "5) A prioritised action plan. Be specific — include file names and concrete fixes."
                ),
                context={
                    "repo_path": repo_path,
                    "graph_summary": graph_summary,
                    "risk_files": risk_files[:5],
                    "workflow_type": "onboarding",
                },
                risk_level="high",
            )
            raw = decision.final_plan
            parts = raw.strip().split("\n\n", 1)
            return (
                parts[0] if parts else raw,
                parts[1] if len(parts) > 1 else "",
                self._parse_issues_from_plan(raw, risk_files),
            )
        except Exception as e:
            logger.warning("CEO analysis failed", error=str(e))
            return (
                f"Repository at {repo_path} was analysed.\n{graph_summary}",
                "Manual review recommended.",
                [],
            )

    async def _quick_scan_analysis(
        self,
        changed_files: List[str],
        affected_modules: List[str],
        risk_level: str,
    ) -> Tuple[str, List[str]]:
        if not changed_files:
            return "No changed files detected.", ["Run git status to verify your working directory."]
        try:
            from app.ai.ceo import get_ceo

            ceo = get_ceo()
            decision = await ceo.receive_problem(
                description=(
                    f"Quick scan: {len(changed_files)} files changed. "
                    f"Risk: {risk_level}. Changed: {', '.join(changed_files[:10])}. "
                    f"Affected modules: {', '.join(affected_modules[:5])}. "
                    "Give a 2-sentence impact summary and 3 specific recommendations."
                ),
                context={"quick_scan": True, "risk_level": risk_level},
                risk_level=risk_level,
            )
            lines = decision.final_plan.strip().split("\n")
            impact = " ".join(lines[:2]) if lines else decision.final_plan
            recs = [l.lstrip("•-123456789. ") for l in lines[2:] if l.strip()][:5]
            return impact, recs or ["Review all changed files before merging."]
        except Exception as e:
            logger.warning("Quick scan CEO analysis failed", error=str(e))
            return (
                f"{len(changed_files)} files changed with {risk_level} risk.",
                ["Review changed files carefully.", "Run tests before merging."],
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _detect_project_meta(self, repo_path: str) -> Tuple[str, str, str, Optional[str]]:
        p = Path(repo_path)
        name = p.name
        py_files = list(p.rglob("*.py"))
        ts_files = list(p.rglob("*.ts"))
        go_files = list(p.rglob("*.go"))
        rs_files = list(p.rglob("*.rs"))

        if len(py_files) >= len(ts_files) and len(py_files) >= len(go_files):
            language = "Python"
        elif ts_files:
            language = "TypeScript"
        elif go_files:
            language = "Go"
        elif rs_files:
            language = "Rust"
        else:
            language = "Unknown"

        requirements = p / "requirements.txt"
        pkg_json = p / "package.json"
        framework: Optional[str] = None
        project_type = "Software project"

        if (p / "app" / "main.py").exists() or (
            requirements.exists()
            and "fastapi" in requirements.read_text(errors="ignore").lower()
        ):
            framework, project_type = "FastAPI", "FastAPI web service"
        elif pkg_json.exists():
            try:
                deps = {
                    **json.loads(pkg_json.read_text()).get("dependencies", {}),
                    **json.loads(pkg_json.read_text()).get("devDependencies", {}),
                }
                if "next" in deps:
                    framework, project_type = "Next.js", "Next.js application"
                elif "react" in deps:
                    framework, project_type = "React", "React application"
                elif "express" in deps:
                    framework, project_type = "Express", "Express API"
                else:
                    project_type = "Node.js application"
            except Exception:
                project_type = "Node.js application"
        elif (p / "go.mod").exists():
            project_type = "Go service"
        elif (p / "Cargo.toml").exists():
            project_type = "Rust project"
        elif language == "Python":
            project_type = "Python project"

        return name, project_type, language, framework

    def _compute_health_score(
        self,
        total_files: int,
        communities: int,
        risk_files: List[str],
        issues: List[IssueItem],
    ) -> int:
        score = 80
        score -= sum(1 for i in issues if i.priority == "critical") * 15
        score -= sum(1 for i in issues if i.priority == "high") * 5
        if total_files > 0 and communities > 0 and communities / total_files > 0.1:
            score += 5
        score -= min(len(risk_files) * 2, 10)
        return max(0, min(100, score))

    def _parse_issues_from_plan(
        self, plan_text: str, risk_files: List[str]
    ) -> List[IssueItem]:
        issues: List[IssueItem] = []
        for f in risk_files[:3]:
            issues.append(IssueItem(
                title=f"High-connectivity file: {Path(f).name}",
                description=f"{f} has many dependencies — changes here have wide blast radius.",
                file_path=f,
                fix_suggestion="Consider splitting into smaller modules or adding interface boundaries.",
                priority="medium",
            ))
        lower = plan_text.lower()
        if any(kw in lower for kw in ["sql injection", "xss", "auth bypass", "hardcoded secret", "exposed credential"]):
            issues.append(IssueItem(
                title="Potential security vulnerability detected",
                description="CEO analysis flagged security concerns.",
                fix_suggestion="Review flagged files with a security audit.",
                priority="critical",
            ))
        if any(kw in lower for kw in ["no tests", "missing tests", "test coverage"]):
            issues.append(IssueItem(
                title="Insufficient test coverage",
                description="Test coverage appears low.",
                fix_suggestion="Add unit tests for critical business logic paths.",
                estimated_time="2–4 hours",
                priority="high",
            ))
        return issues

    def _report_id(self, repo_path: str) -> Optional[str]:
        """Return a deterministic ID if a cached report exists."""
        report = self._reports.get(repo_path)
        if report:
            import hashlib
            return "rpt_" + hashlib.sha1(repo_path.encode()).hexdigest()[:8]
        return None


# ── Singleton ──────────────────────────────────────────────────────────────────

_onboarding_service: Optional[OnboardingService] = None


def get_onboarding_service() -> OnboardingService:
    global _onboarding_service
    if _onboarding_service is None:
        _onboarding_service = OnboardingService()
    return _onboarding_service
