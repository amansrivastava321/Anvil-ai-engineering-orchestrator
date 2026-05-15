"""Context budget manager — hard token limits for every part of the CEO's context.

Token counting uses the 4 chars ≈ 1 token approximation. Good enough for
budget enforcement without adding a tokenizer dependency.
"""
from __future__ import annotations

from typing import Dict

_CHARS_PER_TOKEN = 4  # rough approximation


def _tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class ContextBudget:
    """Track and enforce per-category token budgets.

    Leave 1 000 tokens headroom for the model to think and respond.
    """

    MAX_TOTAL_TOKENS: int = 7_000
    MAX_GRAPH_SUMMARY: int = 300
    MAX_FILE_CONTENTS: int = 2_500
    MAX_MEMORY_PATTERNS: int = 300
    MAX_INSTRUCTIONS: int = 300
    MAX_USER_PROMPT: int = 2_000

    _CATEGORY_LIMITS: Dict[str, int] = {
        "graph_summary": MAX_GRAPH_SUMMARY,
        "file_contents": MAX_FILE_CONTENTS,
        "memory_patterns": MAX_MEMORY_PATTERNS,
        "instructions": MAX_INSTRUCTIONS,
        "user_prompt": MAX_USER_PROMPT,
    }

    def __init__(self) -> None:
        self._used: Dict[str, int] = {cat: 0 for cat in self._CATEGORY_LIMITS}

    # ── Public API ────────────────────────────────────────────────────────────

    def allocate(self, category: str, tokens: int) -> bool:
        """Try to allocate tokens in category. Returns False if any limit is exceeded."""
        cat_limit = self._CATEGORY_LIMITS.get(category, self.MAX_TOTAL_TOKENS)
        cat_used = self._used.get(category, 0)
        total_used = sum(self._used.values())

        if cat_used + tokens > cat_limit:
            return False
        if total_used + tokens > self.MAX_TOTAL_TOKENS:
            return False

        self._used[category] = cat_used + tokens
        return True

    def allocate_text(self, category: str, text: str) -> bool:
        """Convenience wrapper — convert text to token estimate before allocating."""
        return self.allocate(category, _tokens(text))

    def get_remaining(self) -> int:
        """Total tokens still available across all categories."""
        return max(0, self.MAX_TOTAL_TOKENS - sum(self._used.values()))

    def get_category_remaining(self, category: str) -> int:
        """Tokens remaining in a specific category."""
        limit = self._CATEGORY_LIMITS.get(category, self.MAX_TOTAL_TOKENS)
        used = self._used.get(category, 0)
        return max(0, limit - used)

    def truncate_to_fit(self, category: str, text: str) -> str:
        """Truncate text so it fits within the category's remaining budget.

        Keeps: imports, class/function signatures, docstrings.
        Drops: long comment blocks, blank lines at the end.
        """
        remaining_chars = self.get_category_remaining(category) * _CHARS_PER_TOKEN
        total_remaining_chars = self.get_remaining() * _CHARS_PER_TOKEN
        max_chars = min(remaining_chars, total_remaining_chars)

        if len(text) <= max_chars:
            return text

        # Smart truncation: try to end on a complete line
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.7:
            truncated = truncated[:last_newline]

        return truncated + "\n# ... [truncated to fit context budget]"

    def get_budget_report(self) -> str:
        """One-line budget summary for logging."""
        parts = []
        for cat, used in self._used.items():
            limit = self._CATEGORY_LIMITS.get(cat, self.MAX_TOTAL_TOKENS)
            short = cat.replace("_", " ").title()[:8]
            parts.append(f"{short}: {used}/{limit}")
        parts.append(f"Remaining: {self.get_remaining()}")
        return " | ".join(parts)

    def reset(self) -> None:
        self._used = {cat: 0 for cat in self._CATEGORY_LIMITS}
