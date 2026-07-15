"""llm-backends: shared multi-provider LLM backend layer.

Extracted from StoryDaemon novel_agent/tools/ @ 9032e63f7508 (stage 1 of the
extraction plan in StoryDaemon docs/LLM_BACKENDS_INVENTORY.md). Stage 1 is the
StoryDaemon base only; the analyzer's CLI hardening, NovelWriter's
is_available() probes, and the unified alias registry land in stage 2.

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
from .llm_interface import LLMClient, initialize_llm, is_initialized
from .multi_provider_llm import MultiProviderInterface, get_supported_models

__version__ = "0.0.1"

__all__ = [
    "ClaudeCliInterface",
    "CodexInterface",
    "GeminiCliInterface",
    "LLMClient",
    "MultiProviderInterface",
    "get_supported_models",
    "initialize_llm",
    "is_initialized",
    "neutral_cwd",
    "__version__",
]
