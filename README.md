# llm-backends

One model registry and one dispatch layer over every LLM backend the
`send_prompt` family of projects uses: six API providers and three
CLI-agent backends, stdlib-only at the core, provider SDKs as optional
extras.

## Backends

### API backends

Each needs its API key in the environment and the matching SDK extra
installed. Model keys shown are the registry primaries; legacy spellings
resolve through `MODEL_ALIASES`.

- **OpenAI**: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.2`.
  Key: `OPENAI_API_KEY`. Extra: `[openai]`.
- **Anthropic Claude**: `claude-fable-5`, `claude-opus-4-8`,
  `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`.
  Fable 5 and Opus 4.8 reject sampling params; the package omits
  `temperature` for them automatically. Key: `ANTHROPIC_API_KEY`
  (legacy `CLAUDE_API_KEY` still works, with a one-time
  `DeprecationWarning`). Extra: `[anthropic]`.
- **Google Gemini**: `gemini-3.1-pro-preview`, `gemini-3.1-flash-preview`,
  `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`,
  `gemini-2.5-flash`. Key: `GEMINI_API_KEY`. Extra: `[gemini]`.
- **OpenRouter** (hosted router over many upstream models): key
  `openrouter` (upstream chosen by `OPENROUTER_MODEL`), convenience keys
  `openrouter-deepseek` and `openrouter-haiku`, or
  `openrouter:<any-upstream-id>` to route any model with no registry
  change. Ships a hardened client (retries and timeout tuned for
  concurrent fan-out). Key: `OPENROUTER_API_KEY`. Extra: `[openai]`
  (OpenAI-compatible).
- **Venice** (venice.ai, OpenAI-compatible host of open-weight models,
  including uncensored variants): key `venice` (model chosen by
  `VENICE_MODEL`, e.g. `venice-uncensored`). Venice's own injected
  system prompt is disabled on every request so the caller's prompts
  fully govern. Key: `VENICE_API_KEY`. Extra: `[openai]`.
- **Self-hosted**: key `hosted-llm`, any OpenAI-compatible endpoint.
  Env: `HOSTED_LLM_URL`, `HOSTED_LLM_PORT`, `HOSTED_LLM_API_KEY`,
  `HOSTED_LLM_MODEL`. Extra: `[openai]`.

### CLI-agent backends

No API key and no SDK: these shell out to locally installed agent CLIs,
which authenticate via their own logins (`~/.codex`, `~/.claude`, the
gemini CLI's auth).

- **Codex CLI**: `CodexInterface` / `initialize_llm(backend="codex")`.
  Detects hardened-Linux user-namespace restrictions (Ubuntu 23.10+)
  and transparently works around them via an identity-mapped namespace.
- **Claude Code CLI**: `ClaudeCliInterface` /
  `initialize_llm(backend="claude-cli")`, with opus / sonnet / haiku /
  fable model selection.
- **Gemini CLI**: `GeminiCliInterface` /
  `initialize_llm(backend="gemini-cli")`.

All three run from a neutral empty working directory (so the agent
generates text instead of acting on your repo) and **strip the
provider's API keys from the subprocess environment by default**, so the
CLI's subscription login pays rather than a metered key inherited from
`.env`. Pass `strip_provider_keys=False` to opt out. Probe availability
with `is_available()` per interface or `check_cli_availability()`.

## Install

```bash
# Core (stdlib-only; the CLI backends and the whole import surface work
# with zero third-party packages installed):
pip install git+https://github.com/EdwardAThomson/llm-backends@v0.1.1

# With provider SDKs as needed:
pip install "llm-backends[openai]"     # OpenAI / hosted-llm / OpenRouter / Venice
pip install "llm-backends[anthropic]"  # Claude API
pip install "llm-backends[gemini]"     # Gemini API
pip install "llm-backends[all]"
```

## Usage

```python
# Explicit instance (preferred for library consumers):
from llm_backends import MultiProviderInterface
llm = MultiProviderInterface(model="gpt-5.5", timeout=300)
text = llm.generate("Write a sentence about the sea.", max_tokens=200)
text, finish_reason = llm.generate_with_meta("...", max_tokens=200)

# Backend dispatch + module-level convenience singleton:
from llm_backends import llm_interface
llm_interface.initialize_llm(backend="api", model="claude-sonnet-4-6", timeout=300)
text = llm_interface.send_prompt("...", max_tokens=200)

# Model naming helpers:
from llm_backends import get_supported_models, resolve_model
get_supported_models()                 # the registry primaries
resolve_model("claude-4.5")            # legacy alias -> "claude-sonnet-4-5"

# CLI-agent backends:
from llm_backends import CodexInterface, check_cli_availability
check_cli_availability()               # {"codex": bool, "gemini-cli": bool, "claude-cli": bool}
text = CodexInterface().generate("...", timeout=120)
```

Environment variables are read lazily and this package never loads
`.env` itself: apps own their own dotenv loading.

## Contract notes

- **Lazy-import contract:** `import llm_backends` and the entire
  CLI-backend path must work in an environment with no provider SDK
  installed. Enforced by `tests/test_package_contract.py`; the test
  suite runs in a venv containing only `pytest`.
- **Versioned behavioral defaults:** registry contents, per-model
  `max_tokens` / `temperature` defaults, the default `role_description`
  strings, retry counts, and timeout defaults are part of the versioned
  contract. Any change to them is at least a minor version bump with a
  changelog entry. (Downstream benchmark repos pin exact tags and verify
  request payloads on upgrade.)

## Provenance

Extraction plan: StoryDaemon's `docs/LLM_BACKENDS_INVENTORY.md`
(sections 6-7; decided assumptions A1-A8).

- **Stage 1 (0.0.1):** the StoryDaemon base, ported verbatim from
  `StoryDaemon/novel_agent/tools/` at commit
  `9032e63f75083db23bc3d7d74dc47e31baf54baa` (2026-07-15),
  byte-equivalent in behavior.
- **Stage 2 (0.1.0):** merged the analyzer's hardening (CLI
  key-stripping default ON, the codex user-namespace workaround, the
  hardened OpenRouter client, the `openrouter:` prefix passthrough,
  sampling-param omission for Fable 5 / Opus 4.8), NovelWriter's
  availability probes, the unified superset registry with the legacy
  alias table, and the `ANTHROPIC_API_KEY` canon. See `CHANGELOG.md`
  for the full behavior inventory, including the key-strip default
  flip.

## Tests

```bash
python3 -m venv venv && venv/bin/pip install pytest
venv/bin/python -m pytest -q
```
