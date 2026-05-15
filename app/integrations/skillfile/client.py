"""
Integration with skillfile - declarative AI skill management.

This module wraps the skillfile CLI to:
- Load community and custom skills into agent context
- Manage skill lifecycle (install, update, pin, diff)
- Inject skills into context assembly automatically
- Track skill versions and customizations
- Search and discover skills from community registries
- Validate skill compatibility and security scores
- Cache skills for fast context injection

skillfile is an open-source tool (MIT) that manages AI skills declaratively.
GitHub: https://github.com/eljulians/skillfile
"""

import asyncio
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re

import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

from app.core.config.settings import settings
from app.core.monitoring.logging import get_logger, log_function_call
from app.utils.validators import PathValidator, InputSanitizer

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class SkillType(str, Enum):
    """Types of skills."""
    SKILL = "skill"       # Instructional skill/prompt
    AGENT = "agent"       # Autonomous agent definition
    TEMPLATE = "template" # Code template
    CHECKLIST = "checklist" # Verification checklist


class SkillSource(str, Enum):
    """Sources of skills."""
    GITHUB = "github"     # From GitHub repository
    LOCAL = "local"       # Local file in repository
    URL = "url"           # From a direct URL
    COMMUNITY = "community" # From skill registries


class SkillPlatform(str, Enum):
    """AI coding platforms that skills deploy to."""
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    CURSOR = "cursor"
    WINDSURF = "windsurf"
    GEMINI_CLI = "gemini-cli"
    COPILOT = "copilot"
    OPENCODE = "opencode"
    FACTORY = "factory"
    ANTIGRAVITY = "antigravity"
    JUNIE = "junie"


@dataclass
class SkillEntry:
    """Represents a single skill entry from Skillfile."""
    name: str
    source: SkillSource
    skill_type: SkillType
    path: str                          # GitHub path or local file path
    repo: Optional[str] = None         # GitHub repo (owner/repo)
    ref: Optional[str] = None          # Git ref (branch/tag/commit)
    url: Optional[str] = None          # Direct URL for URL source
    pinned: bool = False               # Whether skill has local patches
    patch_path: Optional[Path] = None  # Path to patch file
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def identifier(self) -> str:
        """Unique identifier for this skill."""
        if self.source == SkillSource.GITHUB:
            return f"github:{self.repo}:{self.path}"
        elif self.source == SkillSource.LOCAL:
            return f"local:{self.path}"
        elif self.source == SkillSource.URL:
            return f"url:{self.url}"
        return f"unknown:{self.name}"


@dataclass
class SkillContent:
    """Loaded skill content with metadata."""
    entry: SkillEntry
    content: str
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: str = ""
    size_bytes: int = 0
    security_score: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode()
            ).hexdigest()[:16]
        if not self.size_bytes:
            self.size_bytes = len(self.content.encode())
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count."""
        return self.size_bytes // 4


@dataclass
class SkillSearchResult:
    """Result from searching community skill registries."""
    name: str
    description: str
    source: str                       # Registry source
    url: str
    author: str
    stars: int = 0
    downloads: int = 0
    security_score: int = 0           # 0-100
    tags: List[str] = field(default_factory=list)
    compatible_platforms: List[str] = field(default_factory=list)
    updated_at: Optional[datetime] = None


# ============================================================================
# Skillfile Client
# ============================================================================

class SkillfileError(Exception):
    """Base exception for skillfile errors."""
    pass


class SkillNotFoundError(SkillfileError):
    """Skill not found in registry."""
    pass


class SkillfileNotInstalledError(SkillfileError):
    """skillfile CLI not installed."""
    pass


class SkillfileClient:
    """
    Client for the skillfile CLI tool.
    
    Manages AI skills declaratively - install, update, pin, search.
    """
    
    # Default platforms to deploy to
    DEFAULT_PLATFORMS = [SkillPlatform.CLAUDE_CODE]
    
    # Skill categories for our orchestrator
    AGENT_SKILL_CATEGORIES = {
        "code_review": ["code-review", "review-checklist", "pr-review"],
        "code_generation": ["code-generation", "boilerplate", "scaffold"],
        "testing": ["testing", "test-generation", "tdd", "unit-test"],
        "architecture": ["architecture", "design-patterns", "system-design"],
        "debugging": ["debugging", "troubleshooting", "root-cause"],
        "documentation": ["documentation", "api-docs", "readme"],
        "refactoring": ["refactoring", "clean-code", "code-smells"],
        "security": ["security", "vulnerability", "owasp", "sast"],
    }
    
    def __init__(
        self,
        repo_path: Optional[Path] = None,
        platforms: Optional[List[SkillPlatform]] = None,
    ):
        """
        Initialize skillfile client.
        
        Args:
            repo_path: Path to repository (for local skills)
            platforms: AI platforms to deploy skills to
        """
        self.repo_path = repo_path or Path.cwd()
        self.platforms = platforms or self.DEFAULT_PLATFORMS
        
        # Check if skillfile is installed
        self._installed = self._check_installation()
        
        # Cache for loaded skills
        self._skill_cache: Dict[str, SkillContent] = {}
        self._cache_ttl = timedelta(minutes=30)
        
        # Search result cache
        self._search_cache: Dict[str, Tuple[List[SkillSearchResult], datetime]] = {}
        self._search_cache_ttl = timedelta(hours=1)
        
        # Track statistics
        self._skills_loaded = 0
        self._skills_injected = 0
        
        if self._installed:
            logger.info(
                "skillfile client initialized",
                repo_path=str(self.repo_path),
                platforms=[p.value for p in self.platforms],
            )
        else:
            logger.warning(
                "skillfile CLI not found. Install with: "
                "curl -fsSL https://github.com/eljulians/skillfile/releases/latest/download/install.sh | sh"
            )
    
    # ========================================
    # Installation & Setup
    # ========================================
    
    def _check_installation(self) -> bool:
        """Check if skillfile CLI is installed."""
        try:
            result = subprocess.run(
                ["skillfile", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @property
    def is_installed(self) -> bool:
        """Check if skillfile is available."""
        return self._installed
    
    async def ensure_installed(self) -> bool:
        """Ensure skillfile is installed, with helpful error if not."""
        if self._installed:
            return True
        
        raise SkillfileNotInstalledError(
            "skillfile is not installed. Install it with:\n"
            "  curl -fsSL https://github.com/eljulians/skillfile/releases/latest/download/install.sh | sh\n"
            "Or with cargo:\n"
            "  cargo install skillfile"
        )
    
    async def init_project(
        self,
        platforms: Optional[List[SkillPlatform]] = None,
    ) -> bool:
        """
        Initialize skillfile in the project.
        
        Args:
            platforms: Platforms to configure
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        platforms = platforms or self.platforms
        
        try:
            # Run skillfile init
            process = await asyncio.create_subprocess_exec(
                "skillfile", "init",
                "--platform", ",".join(p.value for p in platforms),
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("skillfile init failed", stderr=stderr.decode())
                return False
            
            logger.info(
                "skillfile initialized",
                platforms=[p.value for p in platforms],
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize skillfile", error=str(e))
            return False
    
    # ========================================
    # Skill Management
    # ========================================
    
    async def add_skill(
        self,
        name: str,
        source: SkillSource,
        path: str,
        repo: Optional[str] = None,
        url: Optional[str] = None,
        skill_type: SkillType = SkillType.SKILL,
    ) -> bool:
        """
        Add a skill to the Skillfile.
        
        Args:
            name: Name for the skill
            source: Source type (github, local, url)
            path: Path to skill file
            repo: GitHub repository (owner/repo)
            url: Direct URL
            skill_type: Type of skill
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        args = ["skillfile", "add"]
        
        if source == SkillSource.GITHUB:
            args.extend(["github", skill_type.value, repo or "", path])
        elif source == SkillSource.LOCAL:
            args.extend(["local", skill_type.value, path])
        elif source == SkillSource.URL:
            args.extend(["url", skill_type.value, name, url or path])
        else:
            raise ValueError(f"Unknown skill source: {source}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(
                    "Failed to add skill",
                    name=name,
                    error=stderr.decode(),
                )
                return False
            
            logger.info(
                "Skill added successfully",
                name=name,
                source=source.value,
            )
            return True
            
        except Exception as e:
            logger.error("Failed to add skill", name=name, error=str(e))
            return False
    
    async def install_skills(self, update: bool = False) -> bool:
        """
        Install all skills from Skillfile.
        
        Args:
            update: Update to latest upstream versions
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        args = ["skillfile", "install"]
        if update:
            args.append("--update")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("skillfile install failed", error=stderr.decode())
                return False
            
            # Clear cache after install
            self._skill_cache.clear()
            
            logger.info("Skills installed successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to install skills", error=str(e))
            return False
    
    async def pin_skill(self, name: str) -> bool:
        """
        Pin a skill with local customizations.
        
        Args:
            name: Skill name to pin
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        try:
            process = await asyncio.create_subprocess_exec(
                "skillfile", "pin", name,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("Failed to pin skill", name=name, error=stderr.decode())
                return False
            
            # Clear cached version
            for key in list(self._skill_cache.keys()):
                if name in key:
                    del self._skill_cache[key]
            
            logger.info("Skill pinned successfully", name=name)
            return True
            
        except Exception as e:
            logger.error("Failed to pin skill", name=name, error=str(e))
            return False
    
    async def remove_skill(self, name: str) -> bool:
        """
        Remove a skill from Skillfile.
        
        Args:
            name: Skill name to remove
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        try:
            process = await asyncio.create_subprocess_exec(
                "skillfile", "remove", name,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("Failed to remove skill", name=name, error=stderr.decode())
                return False
            
            # Remove from cache
            for key in list(self._skill_cache.keys()):
                if name in key:
                    del self._skill_cache[key]
            
            logger.info("Skill removed successfully", name=name)
            return True
            
        except Exception as e:
            logger.error("Failed to remove skill", name=name, error=str(e))
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get status of all skills.
        
        Returns:
            Status dictionary
        """
        await self.ensure_installed()
        
        try:
            process = await asyncio.create_subprocess_exec(
                "skillfile", "status",
                "--check-upstream",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {"error": stderr.decode()}
            
            # Parse status output
            return self._parse_status_output(stdout.decode())
            
        except Exception as e:
            logger.error("Failed to get skill status", error=str(e))
            return {"error": str(e)}
    
    # ========================================
    # Skill Discovery & Search
    # ========================================
    
    async def search_skills(
        self,
        query: str,
        category: Optional[str] = None,
        min_score: int = 0,
        registry: Optional[str] = None,
        limit: int = 20,
    ) -> List[SkillSearchResult]:
        """
        Search community skill registries.
        
        Args:
            query: Search query
            category: Filter by category
            min_score: Minimum security score (0-100)
            registry: Target specific registry
            limit: Maximum results
            
        Returns:
            List of search results
        """
        await self.ensure_installed()
        
        # Check cache
        cache_key = f"{query}:{category}:{min_score}:{registry}"
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if datetime.utcnow() - timestamp < self._search_cache_ttl:
                return results[:limit]
        
        # Build search command
        args = ["skillfile", "search", query]
        
        if min_score > 0:
            args.extend(["--min-score", str(min_score)])
        
        if registry:
            args.extend(["--registry", registry])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("Skill search failed", error=stderr.decode())
                return []
            
            # Parse search results
            results = self._parse_search_results(stdout.decode())
            
            # Filter by category if specified
            if category:
                category_tags = self.AGENT_SKILL_CATEGORIES.get(category, [category])
                results = [
                    r for r in results
                    if any(tag in r.tags for tag in category_tags)
                ]
            
            # Cache results
            self._search_cache[cache_key] = (results, datetime.utcnow())
            
            logger.info(
                "Skills search completed",
                query=query,
                results=len(results),
            )
            
            return results[:limit]
            
        except Exception as e:
            logger.error("Failed to search skills", query=query, error=str(e))
            return []
    
    async def discover_skills_for_task(
        self,
        task_type: str,
        repo_context: Optional[Dict[str, Any]] = None,
    ) -> List[SkillSearchResult]:
        """
        Discover relevant skills for a specific task.
        
        Args:
            task_type: Type of task (code_review, testing, etc.)
            repo_context: Repository context for better matching
            
        Returns:
            Relevant skills
        """
        # Map task to search categories
        categories = self.AGENT_SKILL_CATEGORIES.get(task_type, [task_type])
        
        all_results = []
        
        for category in categories:
            # Build contextual query
            query_parts = [category]
            
            if repo_context:
                if repo_context.get("primary_language"):
                    query_parts.append(repo_context["primary_language"])
                if repo_context.get("framework"):
                    query_parts.append(repo_context["framework"])
            
            query = " ".join(query_parts)
            
            results = await self.search_skills(
                query=query,
                category=task_type,
                min_score=70,  # Only high-quality skills
                limit=10,
            )
            
            all_results.extend(results)
        
        # Deduplicate by name
        seen = set()
        unique_results = []
        for result in all_results:
            if result.name not in seen:
                seen.add(result.name)
                unique_results.append(result)
        
        return unique_results[:10]
    
    # ========================================
    # Skill Loading for Context Injection
    # ========================================
    
    async def load_skills(
        self,
        task_type: str,
        force_reload: bool = False,
    ) -> List[SkillContent]:
        """
        Load relevant skills for a task type.
        
        Args:
            task_type: Type of task
            force_reload: Force reload from disk
            
        Returns:
            Loaded skill contents
        """
        if not self._installed:
            logger.debug("skillfile not installed, skipping skill loading")
            return []
        
        # Check cache
        cache_key = f"skills:{task_type}"
        if not force_reload and cache_key in self._skill_cache:
            cached = self._skill_cache[cache_key]
            if datetime.utcnow() - cached.loaded_at < self._cache_ttl:
                return [cached]
        
        # Find deployed skills for this task
        skill_paths = self._find_deployed_skills(task_type)
        
        loaded_skills = []
        
        for skill_path in skill_paths:
            try:
                content = await self._read_skill_file(skill_path)
                if content:
                    skill_content = SkillContent(
                        entry=SkillEntry(
                            name=skill_path.stem,
                            source=SkillSource.LOCAL,
                            path=str(skill_path),
                            skill_type=SkillType.SKILL,
                        ),
                        content=content,
                    )
                    loaded_skills.append(skill_content)
                    
            except Exception as e:
                logger.warning(
                    "Failed to load skill",
                    path=str(skill_path),
                    error=str(e),
                )
        
        # Cache if skills found
        if loaded_skills:
            # Cache the first (most relevant) skill
            self._skill_cache[cache_key] = loaded_skills[0]
            self._skills_loaded += len(loaded_skills)
        
        logger.debug(
            "Skills loaded for task",
            task_type=task_type,
            count=len(loaded_skills),
        )
        
        return loaded_skills
    
    async def inject_skills_into_context(
        self,
        task_type: str,
        existing_context: str,
        max_skill_tokens: int = 1000,
    ) -> str:
        """
        Inject relevant skills into context.
        
        Args:
            task_type: Type of task
            existing_context: Existing context text
            max_skill_tokens: Maximum tokens for skills
            
        Returns:
            Enhanced context with skills injected
        """
        skills = await self.load_skills(task_type)
        
        if not skills:
            return existing_context
        
        # Build skill injection text
        skill_parts = []
        current_tokens = 0
        
        for skill in skills:
            skill_tokens = skill.token_estimate
            
            if current_tokens + skill_tokens > max_skill_tokens:
                # Truncate skill to fit
                available = max_skill_tokens - current_tokens
                if available > 100:  # Minimum useful skill size
                    truncated = skill.content[:available * 4]
                    skill_parts.append(truncated)
                break
            
            skill_parts.append(skill.content)
            current_tokens += skill_tokens
        
        if not skill_parts:
            return existing_context
        
        # Build injection wrapper
        injection = (
            "## SPECIALIZED INSTRUCTIONS (Injected Skills)\n\n"
            "The following expert instructions are provided to improve response quality:\n\n"
            + "\n\n---\n\n".join(skill_parts)
            + "\n\n## END OF INJECTED SKILLS\n\n"
            + "---\n\n"
        )
        
        # Inject after system prompt but before user request
        enhanced_context = injection + existing_context
        
        self._skills_injected += 1
        
        logger.debug(
            "Skills injected into context",
            task_type=task_type,
            skills_count=len(skill_parts),
            tokens=current_tokens,
        )
        
        return enhanced_context
    
    # ========================================
    # Skill Customization
    # ========================================
    
    async def customize_skill(
        self,
        name: str,
        custom_content: str,
        pin: bool = True,
    ) -> bool:
        """
        Customize a skill with project-specific content.
        
        Args:
            name: Skill name
            custom_content: Custom skill content
            pin: Whether to pin after customization
            
        Returns:
            True if successful
        """
        await self.ensure_installed()
        
        # Find the deployed skill file
        skill_path = None
        for platform in self.platforms:
            possible_path = self.repo_path / f".{platform.value}" / "skills" / name / "SKILL.md"
            if possible_path.exists():
                skill_path = possible_path
                break
        
        if not skill_path:
            raise SkillNotFoundError(f"Skill '{name}' not found in deployed platforms")
        
        try:
            # Write custom content
            import aiofiles
            async with aiofiles.open(skill_path, "w") as f:
                await f.write(custom_content)
            
            # Pin if requested
            if pin:
                await self.pin_skill(name)
            
            # Clear cache
            for key in list(self._skill_cache.keys()):
                if name in key:
                    del self._skill_cache[key]
            
            logger.info("Skill customized successfully", name=name)
            return True
            
        except Exception as e:
            logger.error("Failed to customize skill", name=name, error=str(e))
            return False
    
    async def get_skill_diff(self, name: str) -> Optional[str]:
        """
        Get diff between local and upstream version.
        
        Args:
            name: Skill name
            
        Returns:
            Diff text or None
        """
        await self.ensure_installed()
        
        try:
            process = await asyncio.create_subprocess_exec(
                "skillfile", "diff", name,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return None
            
            return stdout.decode()
            
        except Exception as e:
            logger.error("Failed to get skill diff", name=name, error=str(e))
            return None
    
    # ========================================
    # Private Helpers
    # ========================================
    
    def _find_deployed_skills(self, task_type: str) -> List[Path]:
        """Find deployed skill files for a task type."""
        skill_paths = []
        categories = self.AGENT_SKILL_CATEGORIES.get(task_type, [task_type])
        
        for platform in self.platforms:
            skills_dir = self.repo_path / f".{platform.value}" / "skills"
            
            if not skills_dir.exists():
                continue
            
            # Search for matching skills
            for category in categories:
                # Direct match
                for skill_dir in skills_dir.iterdir():
                    if not skill_dir.is_dir():
                        continue
                    
                    skill_file = skill_dir / "SKILL.md"
                    if not skill_file.exists():
                        continue
                    
                    # Check if skill name matches category
                    if category.lower() in skill_dir.name.lower():
                        skill_paths.append(skill_file)
                    
                    # Check metadata tags
                    meta_file = skill_dir / "metadata.yaml"
                    if meta_file.exists():
                        try:
                            with open(meta_file) as f:
                                metadata = yaml.safe_load(f)
                            if metadata and "tags" in metadata:
                                if any(c.lower() in t.lower() for t in metadata["tags"] for c in categories):
                                    skill_paths.append(skill_file)
                        except Exception:
                            pass
        
        return skill_paths
    
    async def _read_skill_file(self, path: Path) -> Optional[str]:
        """Read a skill file."""
        import aiofiles
        
        if not path.exists():
            return None
        
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()
    
    def _parse_status_output(self, output: str) -> Dict[str, Any]:
        """Parse skillfile status output."""
        # Basic parsing - extract key information
        result = {
            "skills": [],
            "total": 0,
            "pinned": 0,
            "outdated": 0,
        }
        
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            
            skill_info = {}
            
            if "pinned" in line.lower():
                skill_info["pinned"] = True
                result["pinned"] += 1
            
            if "outdated" in line.lower() or "update available" in line.lower():
                skill_info["outdated"] = True
                result["outdated"] += 1
            
            if skill_info:
                result["skills"].append(skill_info)
                result["total"] += 1
        
        return result
    
    def _parse_search_results(self, output: str) -> List[SkillSearchResult]:
        """Parse skillfile search output."""
        results = []
        
        # Try to parse as JSON first
        try:
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    results.append(SkillSearchResult(
                        name=item.get("name", ""),
                        description=item.get("description", ""),
                        source=item.get("source", ""),
                        url=item.get("url", ""),
                        author=item.get("author", ""),
                        stars=item.get("stars", 0),
                        security_score=item.get("security_score", 0),
                        tags=item.get("tags", []),
                    ))
                return results
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse line-by-line
        current: Dict[str, Any] = {}
        for line in output.splitlines():
            line = line.strip()
            if not line:
                if current.get("name"):
                    results.append(self._build_search_result(current))
                    current = {}
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key == "name":
                    current["name"] = value
                elif key == "description":
                    current["description"] = value
                elif key == "source":
                    current["source"] = value
                elif key == "author":
                    current["author"] = value
                elif key in ("stars", "score", "security_score"):
                    try:
                        current["security_score" if "security" in key else "stars"] = int(value)
                    except ValueError:
                        pass
                elif key == "tags":
                    current["tags"] = [t.strip() for t in value.split(",")]
                elif key == "url":
                    current["url"] = value

        if current.get("name"):
            results.append(self._build_search_result(current))

        return results

    def _build_search_result(self, current: Dict[str, Any]) -> "SkillSearchResult":
        """Build a SkillSearchResult with defaults for missing required fields."""
        return SkillSearchResult(
            name=current.get("name", ""),
            description=current.get("description", ""),
            source=current.get("source", "unknown"),
            url=current.get("url", ""),
            author=current.get("author", "unknown"),
            stars=current.get("stars", 0),
            security_score=current.get("security_score", 0),
            tags=current.get("tags", []),
        )
    
    # ========================================
    # Statistics & Health
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get skillfile integration statistics."""
        return {
            "installed": self._installed,
            "skills_loaded": self._skills_loaded,
            "skills_injected": self._skills_injected,
            "cached_skills": len(self._skill_cache),
            "cached_searches": len(self._search_cache),
            "platforms": [p.value for p in self.platforms],
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for skillfile integration."""
        health = {
            "installed": self._installed,
            "status": "healthy" if self._installed else "not_installed",
        }
        
        if self._installed:
            try:
                status = await self.get_status()
                health["skills_count"] = status.get("total", 0)
                health["pinned"] = status.get("pinned", 0)
                health["outdated"] = status.get("outdated", 0)
            except Exception as e:
                health["status"] = "error"
                health["error"] = str(e)
        
        return health
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._skill_cache.clear()
        self._search_cache.clear()
        logger.info("skillfile caches cleared")
    
    async def close(self) -> None:
        """Clean shutdown."""
        self.clear_cache()
        logger.info("skillfile client shut down")


# ============================================================================
# Factory
# ============================================================================

_default_skillfile: Optional[SkillfileClient] = None


def get_skillfile_client() -> SkillfileClient:
    """Get or create the default skillfile client."""
    global _default_skillfile
    if _default_skillfile is None:
        _default_skillfile = SkillfileClient()
    return _default_skillfile


logger.info("skillfile integration module initialized successfully")