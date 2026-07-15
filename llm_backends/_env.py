"""Subprocess environment hygiene and provider env-var canon for llm-backends.

Stage 2 of the extraction plan (StoryDaemon docs/LLM_BACKENDS_INVENTORY.md):
the key-stripping helper is ported from the analyzer's hardened CLI backends
(llm_creative_writing-analyser cli_backends/, the 2026-06-10 billing fix) by
way of NovelWriter's interim hotfix (core/generation/llm_interface/_env.py);
the ANTHROPIC_API_KEY / CLAUDE_API_KEY canon implements assumption A6.

The billing gotcha, and why stripping defaults ON (assumption A4): consuming
apps typically run load_dotenv(), so provider API keys from .env sit in
os.environ. The agent CLIs (codex, claude, gemini) treat an environment API
key as OUTRANKING their own configured subscription login, so a CLI backend
that inherits the full environment silently bills the metered API key instead
of the subscription the user thinks is paying. Stripping the key from the
child environment restores the CLI's own auth. The failure mode of stripping
is a visible auth error; the failure mode of not stripping is silent money.
"""

import os
import warnings
from typing import Optional


def subprocess_env_without(*keys: str) -> dict:
    """A COPY of os.environ with the named variables removed.

    os.environ itself is never mutated; only the child process sees the
    difference.
    """
    env = os.environ.copy()
    for key in keys:
        env.pop(key, None)
    return env


# One-time flag for the CLAUDE_API_KEY deprecation warning: warn once per
# process, not once per request.
_claude_api_key_warned = False


def anthropic_api_key() -> Optional[str]:
    """Return the Anthropic API key from the environment.

    ANTHROPIC_API_KEY is canonical (assumption A6: it is what the analyzer and
    Prompt-Injection-Testing already read, and what the Anthropic SDK
    documents). CLAUDE_API_KEY, the StoryDaemon/NovelWriter spelling, still
    works as a fallback but draws a one-time DeprecationWarning. When both are
    set, the canonical variable wins.
    """
    global _claude_api_key_warned
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    legacy = os.environ.get("CLAUDE_API_KEY")
    if legacy and not _claude_api_key_warned:
        warnings.warn(
            "CLAUDE_API_KEY is deprecated; set ANTHROPIC_API_KEY instead. "
            "llm-backends reads ANTHROPIC_API_KEY first, and the "
            "CLAUDE_API_KEY fallback will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        _claude_api_key_warned = True
    return legacy
