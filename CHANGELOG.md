# Changelog

## 0.1.0 (2026-07-15)

Stage 2 of the extraction plan (StoryDaemon `docs/LLM_BACKENDS_INVENTORY.md`
sections 5, 7.1, 7.7): merges the analyzer's hardening
(`llm_creative_writing-analyser` @ `e263af6` / `2fed1b0`), NovelWriter's
probes (`core/generation/llm_interface/` @ `95bb5bf`), and the unified alias
registry into the StoryDaemon base.

### Headline behavior change for adopters: CLI key-stripping defaults ON

- Every CLI backend now strips the matching provider API keys from a COPY of
  the subprocess environment by default (assumption A4): `CodexInterface`
  strips `OPENAI_API_KEY`; `ClaudeCliInterface` strips `ANTHROPIC_API_KEY`
  and `CLAUDE_API_KEY`; `GeminiCliInterface` strips `GEMINI_API_KEY` and
  `GOOGLE_API_KEY`. Rationale (the analyzer's June 2026 billing incident): an
  env-var key outranks the CLI's configured subscription login, so a key
  pulled from `.env` by `load_dotenv()` in the consuming app was silently
  billed instead of the subscription. The failure mode of stripping is a
  visible auth error; the failure mode of not stripping is silent money.
  **Adopters coming from StoryDaemon/NovelWriter/PIT:** your CLI subprocesses
  previously inherited these keys; if a workflow deliberately relied on
  API-key auth inside a CLI backend, pass the new per-interface opt-out
  `strip_provider_keys=False`.

### Other behavior additions

- **Codex bubblewrap/user-namespace workaround for hardened Linux** (from the
  analyzer's `cli_backends/codex_interface.py`): when
  `kernel.apparmor_restrict_unprivileged_userns` blocks codex's bundled
  bubblewrap (Ubuntu 23.10+), detected via a cached `unshare --map-root-user`
  probe, codex runs inside an identity-mapped user namespace built with the
  setuid `newuidmap`/`newgidmap` helpers, with codex's own sandbox disabled
  (the outer namespace is the sandbox). Requires the `uidmap` package on such
  hosts. Unrestricted hosts and macOS keep codex's own read-only sandbox.
- **Hardened OpenRouter client** (from the analyzer's `ai_helper.py`): the
  OpenRouter client is constructed with `max_retries=6`, `timeout=120.0`
  (`OPENROUTER_MAX_RETRIES` / `OPENROUTER_TIMEOUT`), overriding
  `SDK_MAX_RETRIES=1` for that client only; measured evidence in the module
  comment (20-26% paragraph loss at 4-way fan-out under SDK-default retries).
- **`openrouter:<upstream-model-id>` prefix passthrough** (from the
  analyzer): any model string starting `openrouter:` routes to OpenRouter
  with the remainder as the upstream id, checked before the exact-match
  registry lookup, in `send_prompt`, `send_prompt_meta`,
  `MultiProviderInterface`, and `resolve_model`.
- **Unified superset registry with alias table** (assumption A6): registry
  keys now use the analyzer-style hyphenated naming as primary. Added:
  `gpt-5.4-mini`, `gemini-3.1-flash-preview`, `claude-fable-5`,
  `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
  `claude-sonnet-4-5`, `openrouter-deepseek`, `openrouter-haiku`. Removed as
  registry keys but preserved as aliases (`MODEL_ALIASES`, resolved by
  `resolve_model()` and all dispatch paths): `claude-sonnet-4.5`,
  `claude-haiku-4.5`, `claude-4.5` (StoryDaemon spellings) and
  `claude-4-5-sonnet` (NovelWriter spelling). `get_supported_models()`
  returns primaries only. The primary `claude-haiku-4-5` sends the undated
  API id `claude-haiku-4-5` (analyzer convention) instead of StoryDaemon's
  pinned `claude-haiku-4-5-20251001`; `claude-sonnet-4-5` keeps StoryDaemon's
  pinned `claude-sonnet-4-5-20250929`.
- **Sampling-param omission for Fable 5 / Opus 4.8** (from the analyzer):
  those models reject `temperature`/`top_p`/`top_k` with a 400, so
  `send_prompt_claude_meta` now takes `temperature: Optional[float]` and
  omits the param entirely when None; the `claude-fable-5` /
  `claude-opus-4-8` registry entries pass None. All other Claude entries keep
  `temperature=0.7`.
- **Optional `reasoning_effort` on OpenAI calls** (from the analyzer):
  `send_prompt_openai_meta(..., reasoning_effort=...)`; None (the default)
  omits it, keeping 0.0.1-identical payloads.
- **`is_available()` static probes** on `CodexInterface`,
  `ClaudeCliInterface`, `GeminiCliInterface` and the aggregating
  `check_cli_availability()` in `llm_interface` (from NovelWriter). Probes
  are `shutil.which` only; no subprocess runs.
- **Env-var canon** (assumption A6): `ANTHROPIC_API_KEY` is canonical for the
  Claude API path; `CLAUDE_API_KEY` still works as a fallback with a one-time
  `DeprecationWarning`. When both are set, the canonical variable wins
  (behavior change for StoryDaemon/NovelWriter machines that set both).
- **Claude CLI model heuristic** now also forwards `fable` model ids (from
  the analyzer).
- **De-branding of move artifacts**: the shared scratch dir prefix is now
  `llm-backends-agent-` (was `storydaemon-agent-`); docstrings and error
  strings refer to llm-backends / consuming apps, with provenance lines kept.

### Versioned defaults (restating per inventory doc 7.3)

All 0.0.1 sampling/token/prompt/retry defaults are unchanged and remain part
of the versioned contract: `temperature=0.7` (OpenAI-shaped and Claude paths,
now "or omitted" for the two sampling-free Claude models), `temperature=0.9`
(Gemini), `max_tokens=2000` (OpenAI-shaped), `4096` (Claude),
`max_output_tokens=2048` (Gemini), the neutral default system prompts
(apps pass their own, assumption A5), `SDK_MAX_RETRIES=1` (all clients except
OpenRouter, which now uses `max_retries=6` / `timeout=120.0`), `max_retries=3`
in the `generate_with_retry` layers, `initialize_llm`'s 300 s fallback
timeout, and the per-request timeout plumbing semantics. New in the contract:
the key-strip defaults (ON), the alias table contents, the OpenRouter
hardening constants, and the registry key set listed above.

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
