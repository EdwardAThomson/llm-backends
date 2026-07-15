# Changelog

## 0.0.1 (2026-07-15)

- Extracted from StoryDaemon `novel_agent/tools/` at commit
  `9032e63f75083db23bc3d7d74dc47e31baf54baa` (stage 1 of the extraction plan
  in StoryDaemon `docs/LLM_BACKENDS_INVENTORY.md`). Modules:
  `multi_provider_llm`, `llm_interface`, `codex_interface`,
  `claude_cli_interface`, `gemini_cli_interface`, `agent_cwd`. Behavior is
  byte-equivalent to the source commit; only intra-package imports and
  provenance docstring lines changed.
- Per the inventory doc section 7.3, the following are part of the versioned
  contract from this release on, and changing any of them is at least a minor
  bump with a changelog line:
  - model registry contents (14 keys, incl. `hosted-llm`, `openrouter`,
    `venice`) and the `-latest` suffix fallback;
  - sampling defaults: `temperature=0.7` (OpenAI-shaped and Claude paths),
    `temperature=0.9` (Gemini path);
  - token defaults: `max_tokens=2000` (OpenAI-shaped), `4096` (Claude),
    `max_output_tokens=2048` (Gemini);
  - default system prompt strings ("You are a helpful fiction writing
    assistant. You will create original text only." on the OpenAI-shaped
    paths; "You are a skilled creative writer focused on producing original
    fiction." on the Claude path) - apps should pass their own explicitly
    (assumption A5);
  - retry/timeout behavior: `SDK_MAX_RETRIES = 1`, `max_retries=3` in the
    `generate_with_retry` layers, `initialize_llm`'s 300 s fallback timeout,
    and per-request timeout plumbing semantics.
