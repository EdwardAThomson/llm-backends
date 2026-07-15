# llm-backends

Shared multi-provider LLM backend layer for the `send_prompt` family of
projects: API providers (OpenAI, Anthropic Claude, Google Gemini, hosted-llm,
OpenRouter, Venice) plus CLI-agent backends (`codex`, `claude`, `gemini`)
behind one model registry and a `MultiProviderInterface` /
`initialize_llm` dispatch layer.

## Provenance

Stage 1 of the extraction plan in StoryDaemon's
`docs/LLM_BACKENDS_INVENTORY.md` (sections 6-7): this package is the
StoryDaemon base, ported verbatim from
`StoryDaemon/novel_agent/tools/{multi_provider_llm, llm_interface,
codex_interface, claude_cli_interface, gemini_cli_interface, agent_cwd}.py`
at commit `9032e63f75083db23bc3d7d74dc47e31baf54baa` (2026-07-15). Behavior is
intentionally byte-equivalent to that commit; the analyzer's CLI hardening,
NovelWriter's `is_available()` probes, and the unified alias registry land in
stage 2. See the inventory doc in the StoryDaemon repo for the full plan and
the decided assumptions (A1-A8).

## Install

```bash
# Core (stdlib-only; the CLI backends and the whole import surface work with
# zero third-party packages installed):
pip install git+https://github.com/EdwardAThomson/llm-backends@v0.0.1

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
llm_interface.initialize_llm(backend="api", model="claude-sonnet-4.5", timeout=300)
text = llm_interface.send_prompt("...", max_tokens=200)

# CLI-agent backends (no API key; authenticate via each CLI's own login):
from llm_backends import CodexInterface
text = CodexInterface().generate("...", timeout=120)
```

Environment variables (read lazily, never loaded from `.env` by this package;
apps own their own dotenv loading): `OPENAI_API_KEY`, `CLAUDE_API_KEY`,
`GEMINI_API_KEY`, `HOSTED_LLM_URL` / `HOSTED_LLM_PORT` / `HOSTED_LLM_API_KEY`
/ `HOSTED_LLM_MODEL`, `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`,
`VENICE_API_KEY` / `VENICE_MODEL`.

## Contract notes

- **Lazy-import contract:** `import llm_backends` and the entire CLI-backend
  path must work in an environment with no provider SDK installed. Enforced
  by `tests/test_package_contract.py`; the test suite is run in a venv
  containing only `pytest`.
- **Versioned behavioral defaults:** registry contents, per-model
  `max_tokens` / `temperature` defaults, the default `role_description`
  strings, retry counts, and timeout defaults are part of the versioned
  contract (inventory doc section 7.3). Any change to them is at least a
  minor version bump with a changelog entry.

## Tests

```bash
python3 -m venv venv && venv/bin/pip install pytest
venv/bin/python -m pytest -q
```
