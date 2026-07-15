"""llm-backends: shared multi-provider LLM backend layer.

Extracted from StoryDaemon novel_agent/tools/ @ 9032e63f7508 (stage 1 of the
extraction plan in StoryDaemon docs/LLM_BACKENDS_INVENTORY.md). Stage 2 merged
the analyzer's hardening (llm_creative_writing-analyser: CLI key-stripping
default ON, the codex bubblewrap/user-namespace workaround, the hardened
OpenRouter client, the "openrouter:<upstream-id>" prefix passthrough, sampling
param omission for Fable 5 / Opus 4.8), NovelWriter's is_available() probes +
check_cli_availability(), the unified superset model registry with legacy
alias resolution (assumption A6), and the ANTHROPIC_API_KEY canon (with a
deprecated CLAUDE_API_KEY fallback).

Importing this package must always work with zero provider SDKs installed
(lazy-import contract, inventory doc section 7.2): the CLI backends are
stdlib-only, and the API layer degrades its SDK imports to None until a
provider is actually called.

Both `send_prompt` families keep their submodule homes to avoid ambiguity:
- `llm_backends.multi_provider_llm.send_prompt(...)` routes by model key
  through the registry (stateless).
- `llm_backends.llm_interface.send_prompt(...)` uses the module-level
  singleton set up by `initialize_llm(...)`.
"""

from .agent_cwd import neutral_cwd
from .claude_cli_interface import ClaudeCliInterface
from .codex_interface import CodexInterface
from .gemini_cli_interface import GeminiCliInterface
from .llm_interface import (
    LLMClient,
    check_cli_availability,
    initialize_llm,
    is_initialized,
)
from .multi_provider_llm import (
    MODEL_ALIASES,
    MultiProviderInterface,
    get_supported_models,
    resolve_model,
)

__version__ = "0.1.1"

# Canonical default API model. Consuming apps should read this instead of
# hardcoding their own fallback literals (e.g. StoryDaemon's cli/main.py /
# cli/commands/*.py "gpt-5.5" fallbacks), so a registry default bump is a
# single-place change (inventory doc section 7.4, step 2).
DEFAULT_API_MODEL = "gpt-5.5"

__all__ = [
    "ClaudeCliInterface",
    "CodexInterface",
    "DEFAULT_API_MODEL",
    "GeminiCliInterface",
    "LLMClient",
    "MODEL_ALIASES",
    "MultiProviderInterface",
    "check_cli_availability",
    "get_supported_models",
    "initialize_llm",
    "is_initialized",
    "neutral_cwd",
    "resolve_model",
    "__version__",
]
