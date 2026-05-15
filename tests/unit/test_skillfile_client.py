"""Tests for app.integrations.skillfile.client — data models and SkillfileClient."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integrations.skillfile.client import (
    SkillContent,
    SkillEntry,
    SkillfileClient,
    SkillfileNotInstalledError,
    SkillPlatform,
    SkillSearchResult,
    SkillSource,
    SkillType,
    get_skillfile_client,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_client(installed: bool = False, **kwargs) -> SkillfileClient:
    """Return a SkillfileClient without a real skillfile binary."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0 if installed else 1)
        return SkillfileClient(**kwargs)


# ============================================================================
# SkillType enum
# ============================================================================

def test_skill_type_values():
    assert SkillType.SKILL == "skill"
    assert SkillType.AGENT == "agent"
    assert SkillType.TEMPLATE == "template"
    assert SkillType.CHECKLIST == "checklist"


def test_skill_type_is_str():
    assert isinstance(SkillType.SKILL, str)


# ============================================================================
# SkillSource enum
# ============================================================================

def test_skill_source_values():
    assert SkillSource.GITHUB == "github"
    assert SkillSource.LOCAL == "local"
    assert SkillSource.URL == "url"
    assert SkillSource.COMMUNITY == "community"


# ============================================================================
# SkillPlatform enum
# ============================================================================

def test_skill_platform_values():
    assert SkillPlatform.CLAUDE_CODE == "claude-code"
    assert SkillPlatform.CURSOR == "cursor"
    assert SkillPlatform.COPILOT == "copilot"


def test_skill_platform_all_members():
    members = {p.value for p in SkillPlatform}
    assert "claude-code" in members
    assert "windsurf" in members
    assert "gemini-cli" in members


# ============================================================================
# SkillEntry dataclass
# ============================================================================

def test_skill_entry_identifier_github():
    entry = SkillEntry(
        name="my-skill",
        source=SkillSource.GITHUB,
        skill_type=SkillType.SKILL,
        path="skills/foo.md",
        repo="owner/repo",
    )
    assert entry.identifier == "github:owner/repo:skills/foo.md"


def test_skill_entry_identifier_local():
    entry = SkillEntry(
        name="local-skill",
        source=SkillSource.LOCAL,
        skill_type=SkillType.SKILL,
        path="./local.md",
    )
    assert entry.identifier == "local:./local.md"


def test_skill_entry_identifier_url():
    entry = SkillEntry(
        name="url-skill",
        source=SkillSource.URL,
        skill_type=SkillType.SKILL,
        path="",
        url="https://example.com/skill.md",
    )
    assert entry.identifier == "url:https://example.com/skill.md"


def test_skill_entry_identifier_unknown_source():
    entry = SkillEntry(
        name="community-skill",
        source=SkillSource.COMMUNITY,
        skill_type=SkillType.SKILL,
        path="some/path",
    )
    assert entry.identifier == "unknown:community-skill"


def test_skill_entry_defaults():
    entry = SkillEntry(
        name="x",
        source=SkillSource.LOCAL,
        skill_type=SkillType.SKILL,
        path="x.md",
    )
    assert entry.repo is None
    assert entry.ref is None
    assert entry.url is None
    assert entry.pinned is False
    assert entry.patch_path is None
    assert entry.metadata == {}


# ============================================================================
# SkillContent dataclass
# ============================================================================

def test_skill_content_post_init_sets_hash():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="hello world")
    expected_hash = hashlib.sha256(b"hello world").hexdigest()[:16]
    assert sc.content_hash == expected_hash


def test_skill_content_post_init_sets_size_bytes():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="hello")
    assert sc.size_bytes == len(b"hello")


def test_skill_content_post_init_preserves_provided_hash():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="hello", content_hash="custom_hash")
    assert sc.content_hash == "custom_hash"


def test_skill_content_post_init_preserves_provided_size():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="hello", size_bytes=999)
    assert sc.size_bytes == 999


def test_skill_content_token_estimate():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="x" * 400)
    # size_bytes == 400, token_estimate == 400 // 4 == 100
    assert sc.token_estimate == 100


def test_skill_content_token_estimate_zero():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="hi")
    assert sc.token_estimate == len("hi".encode()) // 4


def test_skill_content_loaded_at_is_datetime():
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    sc = SkillContent(entry=entry, content="abc")
    assert isinstance(sc.loaded_at, datetime)


# ============================================================================
# SkillSearchResult dataclass
# ============================================================================

def test_skill_search_result_defaults():
    r = SkillSearchResult(
        name="test-skill",
        description="A test skill",
        source="github",
        url="https://example.com",
        author="foo",
    )
    assert r.stars == 0
    assert r.downloads == 0
    assert r.security_score == 0
    assert r.tags == []
    assert r.compatible_platforms == []
    assert r.updated_at is None


def test_skill_search_result_with_values():
    r = SkillSearchResult(
        name="test-skill",
        description="desc",
        source="community",
        url="https://example.com/skill",
        author="author",
        stars=42,
        security_score=85,
        tags=["python", "testing"],
    )
    assert r.stars == 42
    assert r.security_score == 85
    assert "python" in r.tags


# ============================================================================
# SkillfileClient.__init__
# ============================================================================

def test_client_init_not_installed():
    client = _make_client(installed=False)
    assert client.is_installed is False


def test_client_init_installed():
    client = _make_client(installed=True)
    assert client.is_installed is True


def test_client_init_default_repo_path():
    client = _make_client()
    assert client.repo_path == Path.cwd()


def test_client_init_custom_repo_path(tmp_path):
    client = _make_client(repo_path=tmp_path)
    assert client.repo_path == tmp_path


def test_client_init_default_platforms():
    client = _make_client()
    assert client.platforms == [SkillPlatform.CLAUDE_CODE]


def test_client_init_custom_platforms():
    client = _make_client(platforms=[SkillPlatform.CURSOR, SkillPlatform.COPILOT])
    assert SkillPlatform.CURSOR in client.platforms
    assert SkillPlatform.COPILOT in client.platforms


def test_client_init_caches_empty():
    client = _make_client()
    assert client._skill_cache == {}
    assert client._search_cache == {}


def test_client_init_stats_zero():
    client = _make_client()
    assert client._skills_loaded == 0
    assert client._skills_injected == 0


# ============================================================================
# SkillfileClient._check_installation
# ============================================================================

def test_check_installation_returns_true_on_zero_returncode():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        client = SkillfileClient()
    assert client._installed is True


def test_check_installation_returns_false_on_nonzero_returncode():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        client = SkillfileClient()
    assert client._installed is False


def test_check_installation_returns_false_on_file_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        client = SkillfileClient()
    assert client._installed is False


def test_check_installation_returns_false_on_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="skillfile", timeout=5)):
        client = SkillfileClient()
    assert client._installed is False


def test_check_installation_calls_version_flag():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        SkillfileClient()
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args == ["skillfile", "--version"]


# ============================================================================
# SkillfileClient.is_installed property
# ============================================================================

def test_is_installed_property_false():
    client = _make_client(installed=False)
    # Must access as property, not call as method
    result = client.is_installed
    assert result is False


def test_is_installed_property_true():
    client = _make_client(installed=True)
    result = client.is_installed
    assert result is True


# ============================================================================
# SkillfileClient.ensure_installed
# ============================================================================

@pytest.mark.asyncio
async def test_ensure_installed_raises_when_not_installed():
    client = _make_client(installed=False)
    with pytest.raises(SkillfileNotInstalledError):
        await client.ensure_installed()


@pytest.mark.asyncio
async def test_ensure_installed_returns_true_when_installed():
    client = _make_client(installed=True)
    result = await client.ensure_installed()
    assert result is True


@pytest.mark.asyncio
async def test_ensure_installed_error_message_mentions_install():
    client = _make_client(installed=False)
    with pytest.raises(SkillfileNotInstalledError, match="skillfile"):
        await client.ensure_installed()


# ============================================================================
# SkillfileClient.get_stats
# ============================================================================

def test_get_stats_returns_dict_with_required_keys():
    client = _make_client(installed=False)
    stats = client.get_stats()
    assert "installed" in stats
    assert "skills_loaded" in stats
    assert "skills_injected" in stats
    assert "cached_skills" in stats
    assert "cached_searches" in stats
    assert "platforms" in stats


def test_get_stats_installed_false():
    client = _make_client(installed=False)
    assert client.get_stats()["installed"] is False


def test_get_stats_installed_true():
    client = _make_client(installed=True)
    assert client.get_stats()["installed"] is True


def test_get_stats_platforms_list():
    client = _make_client(installed=False)
    stats = client.get_stats()
    assert isinstance(stats["platforms"], list)
    assert "claude-code" in stats["platforms"]


def test_get_stats_cached_skills_reflects_cache():
    client = _make_client(installed=False)
    assert client.get_stats()["cached_skills"] == 0
    # Manually add something to the cache
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    client._skill_cache["key"] = SkillContent(entry=entry, content="x")
    assert client.get_stats()["cached_skills"] == 1


# ============================================================================
# SkillfileClient.health_check
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_not_installed():
    client = _make_client(installed=False)
    result = await client.health_check()
    assert result["installed"] is False
    assert result["status"] == "not_installed"


@pytest.mark.asyncio
async def test_health_check_installed_calls_get_status():
    client = _make_client(installed=True)
    mock_status = {"total": 3, "pinned": 1, "outdated": 0}
    client.get_status = AsyncMock(return_value=mock_status)
    result = await client.health_check()
    assert result["installed"] is True
    assert result["status"] == "healthy"
    assert result["skills_count"] == 3
    assert result["pinned"] == 1
    assert result["outdated"] == 0


@pytest.mark.asyncio
async def test_health_check_installed_get_status_raises():
    client = _make_client(installed=True)
    client.get_status = AsyncMock(side_effect=RuntimeError("boom"))
    result = await client.health_check()
    assert result["status"] == "error"
    assert "error" in result


# ============================================================================
# SkillfileClient.clear_cache
# ============================================================================

def test_clear_cache_clears_skill_cache():
    client = _make_client(installed=False)
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    client._skill_cache["key"] = SkillContent(entry=entry, content="x")
    client.clear_cache()
    assert client._skill_cache == {}


def test_clear_cache_clears_search_cache():
    client = _make_client(installed=False)
    client._search_cache["query"] = ([], datetime.utcnow())
    client.clear_cache()
    assert client._search_cache == {}


def test_clear_cache_idempotent_on_empty():
    client = _make_client(installed=False)
    client.clear_cache()
    client.clear_cache()
    assert client._skill_cache == {}
    assert client._search_cache == {}


# ============================================================================
# SkillfileClient.close
# ============================================================================

@pytest.mark.asyncio
async def test_close_clears_caches():
    client = _make_client(installed=False)
    entry = SkillEntry(
        name="s", source=SkillSource.LOCAL, skill_type=SkillType.SKILL, path="s.md"
    )
    client._skill_cache["k"] = SkillContent(entry=entry, content="data")
    client._search_cache["q"] = ([], datetime.utcnow())
    await client.close()
    assert client._skill_cache == {}
    assert client._search_cache == {}


# ============================================================================
# SkillfileClient._parse_status_output
# ============================================================================

def test_parse_status_output_empty_string():
    client = _make_client()
    result = client._parse_status_output("")
    assert result == {"skills": [], "total": 0, "pinned": 0, "outdated": 0}


def test_parse_status_output_pinned_line():
    client = _make_client()
    output = "my-skill [pinned] v1.0.0"
    result = client._parse_status_output(output)
    assert result["pinned"] == 1
    assert result["total"] == 1
    assert result["skills"][0]["pinned"] is True


def test_parse_status_output_outdated_line():
    client = _make_client()
    output = "another-skill [outdated] v0.9.0"
    result = client._parse_status_output(output)
    assert result["outdated"] == 1
    assert result["total"] == 1
    assert result["skills"][0]["outdated"] is True


def test_parse_status_output_update_available_line():
    client = _make_client()
    output = "some-skill - update available"
    result = client._parse_status_output(output)
    assert result["outdated"] == 1


def test_parse_status_output_multiple_lines():
    client = _make_client()
    output = (
        "skill-a [pinned] v1.0\n"
        "skill-b [outdated] v0.8\n"
        "skill-c [pinned] v2.1\n"
    )
    result = client._parse_status_output(output)
    assert result["pinned"] == 2
    assert result["outdated"] == 1
    assert result["total"] == 3


def test_parse_status_output_blank_lines_ignored():
    client = _make_client()
    output = "\n\nskill-x [pinned]\n\n"
    result = client._parse_status_output(output)
    assert result["total"] == 1
    assert result["pinned"] == 1


def test_parse_status_output_no_matching_keywords():
    client = _make_client()
    output = "random line with no keywords"
    result = client._parse_status_output(output)
    assert result["total"] == 0
    assert result["skills"] == []


# ============================================================================
# SkillfileClient._parse_search_results
# ============================================================================

def test_parse_search_results_empty_string():
    client = _make_client()
    results = client._parse_search_results("")
    assert results == []


def test_parse_search_results_valid_json_list():
    client = _make_client()
    data = [
        {
            "name": "test-skill",
            "description": "A test skill",
            "source": "github",
            "url": "https://github.com/owner/repo",
            "author": "foo",
            "stars": 10,
            "security_score": 90,
            "tags": ["python"],
        }
    ]
    output = json.dumps(data)
    results = client._parse_search_results(output)
    assert len(results) == 1
    r = results[0]
    assert r.name == "test-skill"
    assert r.description == "A test skill"
    assert r.source == "github"
    assert r.url == "https://github.com/owner/repo"
    assert r.author == "foo"
    assert r.stars == 10
    assert r.security_score == 90
    assert r.tags == ["python"]


def test_parse_search_results_json_list_multiple():
    client = _make_client()
    data = [
        {"name": "skill-a", "description": "d", "source": "s", "url": "u", "author": "a"},
        {"name": "skill-b", "description": "d", "source": "s", "url": "u", "author": "b"},
    ]
    results = client._parse_search_results(json.dumps(data))
    assert len(results) == 2
    assert results[0].name == "skill-a"
    assert results[1].name == "skill-b"


def test_parse_search_results_json_missing_fields_uses_defaults():
    client = _make_client()
    data = [{"name": "minimal"}]
    results = client._parse_search_results(json.dumps(data))
    assert len(results) == 1
    r = results[0]
    assert r.name == "minimal"
    assert r.stars == 0
    assert r.security_score == 0
    assert r.tags == []


def test_parse_search_results_invalid_json_fallback_multiline():
    client = _make_client()
    output = (
        "name: code-review\n"
        "description: Reviews code\n"
        "url: https://example.com/code-review\n"
    )
    results = client._parse_search_results(output)
    assert len(results) == 1
    r = results[0]
    assert r.name == "code-review"
    assert r.description == "Reviews code"
    assert r.url == "https://example.com/code-review"
    assert r.source == "unknown"  # default when not provided
    assert r.author == "unknown"  # default when not provided


def test_parse_search_results_multiline_with_stars():
    client = _make_client()
    output = (
        "name: star-skill\n"
        "description: Has stars\n"
        "url: https://example.com\n"
        "stars: 42\n"
    )
    results = client._parse_search_results(output)
    assert len(results) == 1
    assert results[0].stars == 42


def test_parse_search_results_multiline_with_tags():
    client = _make_client()
    output = (
        "name: tagged-skill\n"
        "description: Has tags\n"
        "url: https://example.com\n"
        "tags: python, testing, tdd\n"
    )
    results = client._parse_search_results(output)
    assert len(results) == 1
    tags = results[0].tags
    assert "python" in tags
    assert "testing" in tags
    assert "tdd" in tags


def test_parse_search_results_multiline_multiple_blocks():
    client = _make_client()
    output = (
        "name: skill-one\n"
        "description: First skill\n"
        "url: https://example.com/one\n"
        "\n"
        "name: skill-two\n"
        "description: Second skill\n"
        "url: https://example.com/two\n"
    )
    results = client._parse_search_results(output)
    assert len(results) == 2
    assert results[0].name == "skill-one"
    assert results[1].name == "skill-two"


def test_parse_search_results_multiline_with_source_and_author():
    client = _make_client()
    output = (
        "name: full-skill\n"
        "description: Has all fields\n"
        "source: github\n"
        "url: https://example.com\n"
        "author: contributor\n"
    )
    results = client._parse_search_results(output)
    assert len(results) == 1
    assert results[0].source == "github"
    assert results[0].author == "contributor"


# ============================================================================
# get_skillfile_client singleton
# ============================================================================

def test_get_skillfile_client_returns_client_instance():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        # Reset the module-level singleton so our patch applies
        import app.integrations.skillfile.client as sfmod
        sfmod._default_skillfile = None
        client = get_skillfile_client()
    assert isinstance(client, SkillfileClient)


def test_get_skillfile_client_returns_same_instance():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        import app.integrations.skillfile.client as sfmod
        sfmod._default_skillfile = None
        c1 = get_skillfile_client()
        c2 = get_skillfile_client()
    assert c1 is c2


# ============================================================================
# SkillfileClient constants
# ============================================================================

def test_default_platforms_constant():
    assert SkillfileClient.DEFAULT_PLATFORMS == [SkillPlatform.CLAUDE_CODE]


def test_agent_skill_categories_has_expected_keys():
    keys = set(SkillfileClient.AGENT_SKILL_CATEGORIES.keys())
    expected = {"code_review", "code_generation", "testing", "architecture",
                "debugging", "documentation", "refactoring", "security"}
    assert expected.issubset(keys)
