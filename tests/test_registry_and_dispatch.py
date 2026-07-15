"""Tests for the stage-2 unified registry and dispatch merges.

Covers: the superset registry (assumption A6), alias resolution, the
"openrouter:<upstream-id>" prefix passthrough, the sampling-param omission for
Fable 5 / Opus 4.8, optional reasoning_effort on OpenAI calls, the hardened
OpenRouter client construction, and the ANTHROPIC_API_KEY/CLAUDE_API_KEY canon.
"""

import warnings

import pytest

import llm_backends
from llm_backends import _env, multi_provider_llm
from llm_backends.multi_provider_llm import (
    MODEL_ALIASES,
    MultiProviderInterface,
    get_supported_models,
    resolve_model,
    send_prompt,
    send_prompt_meta,
)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """No cached client or env key leaks between tests."""
    monkeypatch.setattr(multi_provider_llm, "_openrouter_client", None)
    monkeypatch.setattr(multi_provider_llm, "_venice_client", None)
    monkeypatch.setattr(multi_provider_llm, "_hosted_llm_client", None)
    monkeypatch.setattr(multi_provider_llm, "_openai_client", None)
    monkeypatch.setattr(multi_provider_llm, "_anthropic_client", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    yield


# ---- registry contents (superset, assumption A6) ----------------------------------

EXPECTED_PRIMARIES = {
    # StoryDaemon providers
    "hosted-llm", "openrouter", "venice",
    # OpenAI GPT-5 family (superset incl. the analyzer's 5.4-mini)
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.2",
    # Anthropic (analyzer-style hyphenated primaries)
    "claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6",
    "claude-sonnet-4-5", "claude-haiku-4-5",
    # Gemini (superset incl. the analyzer's 3.1-flash-preview)
    "gemini-3.1-pro-preview", "gemini-3.1-flash-preview",
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "gemini-2.5-pro", "gemini-2.5-flash",
    # OpenRouter convenience keys (analyzer)
    "openrouter-deepseek", "openrouter-haiku",
}


def test_supported_models_is_the_unified_superset():
    assert set(get_supported_models()) == EXPECTED_PRIMARIES


def test_supported_models_lists_primaries_only():
    supported = set(get_supported_models())
    for alias in MODEL_ALIASES:
        assert alias not in supported
    # And every alias points at a real primary.
    for target in MODEL_ALIASES.values():
        assert target in supported


# ---- alias resolution --------------------------------------------------------------

@pytest.mark.parametrize(
    ("legacy", "primary"),
    [
        ("claude-sonnet-4.5", "claude-sonnet-4-5"),   # StoryDaemon spelling
        ("claude-haiku-4.5", "claude-haiku-4-5"),     # StoryDaemon spelling
        ("claude-4.5", "claude-sonnet-4-5"),          # StoryDaemon family alias
        ("claude-4-5-sonnet", "claude-sonnet-4-5"),   # NovelWriter spelling
    ],
)
def test_resolve_model_accepts_legacy_aliases(legacy, primary):
    assert resolve_model(legacy) == primary


def test_resolve_model_passes_primaries_through():
    for key in get_supported_models():
        assert resolve_model(key) == key


def test_resolve_model_keeps_latest_suffix_fallback(monkeypatch):
    monkeypatch.setitem(
        multi_provider_llm._model_config_meta, "somemodel-latest",
        lambda prompt, max_tokens, timeout=None: ("x", None),
    )
    assert resolve_model("somemodel") == "somemodel-latest"


def test_resolve_model_openrouter_prefix_verbatim():
    assert resolve_model("openrouter:deepseek/deepseek-chat") == "openrouter:deepseek/deepseek-chat"
    with pytest.raises(ValueError, match="missing upstream model id"):
        resolve_model("openrouter:")


def test_resolve_model_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model("not-a-model")


def test_send_prompt_routes_alias_to_primary_entry(monkeypatch):
    calls = {}
    monkeypatch.setitem(
        multi_provider_llm._model_config, "claude-sonnet-4-5",
        lambda prompt, max_tokens: calls.update(prompt=prompt) or "aliased",
    )
    assert send_prompt("hi", model="claude-sonnet-4.5") == "aliased"
    assert calls == {"prompt": "hi"}


def test_interface_accepts_alias(monkeypatch):
    monkeypatch.setitem(
        multi_provider_llm._model_config, "claude-sonnet-4-5",
        lambda prompt, max_tokens: "via interface",
    )
    client = MultiProviderInterface(model="claude-4.5")
    assert client.generate("p", max_tokens=5) == "via interface"


# ---- "openrouter:" prefix passthrough (analyzer ai_helper.py:83-90) ----------------

class _CapturingChatClient:
    def __init__(self, text="routed"):
        captured = self.captured = {}

        class _Message:
            content = text

        class _Choice:
            message = _Message()
            finish_reason = "stop"

        class _Response:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return _Response()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_send_prompt_prefix_routes_upstream_id_verbatim(monkeypatch):
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    out = send_prompt("hi", model="openrouter:deepseek/deepseek-chat", max_tokens=64)

    assert out == "routed"
    assert client.captured["model"] == "deepseek/deepseek-chat"
    assert client.captured["max_tokens"] == 64


def test_send_prompt_meta_prefix_returns_tuple_and_threads_timeout(monkeypatch):
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    result = send_prompt_meta(
        "hi", model="openrouter:anthropic/claude-haiku-4.5", timeout=30
    )

    assert result == ("routed", "stop")
    assert client.captured["model"] == "anthropic/claude-haiku-4.5"
    assert client.captured["timeout"] == 30


def test_prefix_works_without_a_registry_key(monkeypatch):
    # The whole point: any OpenRouter model routes with no code change,
    # checked BEFORE the exact-match registry lookup.
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)
    assert "openrouter:brand-new/model" not in multi_provider_llm._model_config

    send_prompt("hi", model="openrouter:brand-new/model")

    assert client.captured["model"] == "brand-new/model"


def test_prefix_missing_upstream_id_raises_value_error():
    with pytest.raises(ValueError, match="missing upstream model id"):
        send_prompt("hi", model="openrouter:")
    with pytest.raises(ValueError, match="missing upstream model id"):
        send_prompt_meta("hi", model="openrouter:")


def test_prefix_via_interface(monkeypatch):
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    iface = MultiProviderInterface(model="openrouter:deepseek/deepseek-chat")
    assert iface.generate("p", max_tokens=8) == "routed"
    assert client.captured["model"] == "deepseek/deepseek-chat"


# ---- Fable 5 / Opus 4.8 sampling-param omission (analyzer :119-128, :302-303) ------

class _CapturingAnthropicClient:
    def __init__(self):
        captured = self.captured = {}

        class _Block:
            text = "claude prose"

        class _Response:
            content = [_Block()]
            stop_reason = "end_turn"

        class _Messages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return _Response()

        self.messages = _Messages()


@pytest.mark.parametrize("key", ["claude-fable-5", "claude-opus-4-8"])
def test_sampling_free_models_omit_temperature(monkeypatch, key):
    client = _CapturingAnthropicClient()
    monkeypatch.setattr(multi_provider_llm, "_get_anthropic_client", lambda: client)

    assert send_prompt("p", model=key) == "claude prose"

    assert client.captured["model"] == key
    # Omitted, not sent as None: these models 400 on any sampling param.
    assert "temperature" not in client.captured


@pytest.mark.parametrize(
    ("key", "api_id"),
    [
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-haiku-4-5", "claude-haiku-4-5"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
    ],
)
def test_sampling_models_keep_temperature(monkeypatch, key, api_id):
    client = _CapturingAnthropicClient()
    monkeypatch.setattr(multi_provider_llm, "_get_anthropic_client", lambda: client)

    send_prompt("p", model=key)

    assert client.captured["model"] == api_id
    assert client.captured["temperature"] == 0.7


def test_send_prompt_claude_meta_temperature_none_omits(monkeypatch):
    client = _CapturingAnthropicClient()
    monkeypatch.setattr(multi_provider_llm, "_get_anthropic_client", lambda: client)

    multi_provider_llm.send_prompt_claude_meta("p", temperature=None)
    assert "temperature" not in client.captured

    multi_provider_llm.send_prompt_claude_meta("p", temperature=0.3)
    assert client.captured["temperature"] == 0.3


# ---- optional reasoning_effort on OpenAI calls (analyzer :225) ---------------------

def test_reasoning_effort_omitted_by_default(monkeypatch):
    client = _CapturingChatClient(text="prose")
    monkeypatch.setattr(multi_provider_llm, "_get_openai_client", lambda: client)

    multi_provider_llm.send_prompt_openai_meta("p")
    assert "reasoning_effort" not in client.captured


def test_reasoning_effort_sent_when_set(monkeypatch):
    client = _CapturingChatClient(text="prose")
    monkeypatch.setattr(multi_provider_llm, "_get_openai_client", lambda: client)

    multi_provider_llm.send_prompt_openai_meta("p", reasoning_effort="high")
    assert client.captured["reasoning_effort"] == "high"


# ---- hardened OpenRouter client (analyzer ai_helper.py:52-57) ----------------------

def test_openrouter_client_constructed_with_hardening_kwargs(monkeypatch):
    class KwargCapturingOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(multi_provider_llm, "OpenAI", KwargCapturingOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    client = multi_provider_llm._get_openrouter_client()

    assert client.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert client.kwargs["max_retries"] == 6
    assert client.kwargs["timeout"] == 120.0
    # The constants are part of the versioned contract (inventory doc 7.3).
    assert multi_provider_llm.OPENROUTER_MAX_RETRIES == 6
    assert multi_provider_llm.OPENROUTER_TIMEOUT == 120.0


# ---- ANTHROPIC_API_KEY canon with deprecated CLAUDE_API_KEY fallback (A6) ----------

class _FakeAnthropicSDK:
    class Anthropic:
        def __init__(self, api_key=None, max_retries=None):
            self.api_key = api_key


def test_anthropic_canonical_key_wins(monkeypatch):
    monkeypatch.setattr(multi_provider_llm, "anthropic", _FakeAnthropicSDK)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-canonical")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-legacy")
    monkeypatch.setattr(_env, "_claude_api_key_warned", False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # canonical path must not warn
        client = multi_provider_llm._get_anthropic_client()

    assert client.api_key == "sk-canonical"


def test_anthropic_legacy_key_still_works_with_one_time_warning(monkeypatch):
    monkeypatch.setattr(multi_provider_llm, "anthropic", _FakeAnthropicSDK)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-legacy")
    monkeypatch.setattr(_env, "_claude_api_key_warned", False)

    with pytest.warns(DeprecationWarning, match="CLAUDE_API_KEY is deprecated"):
        client = multi_provider_llm._get_anthropic_client()
    assert client.api_key == "sk-legacy"

    # One-time: a second read keeps working but stays silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _env.anthropic_api_key() == "sk-legacy"


def test_anthropic_no_key_raises_with_canonical_name(monkeypatch):
    monkeypatch.setattr(multi_provider_llm, "anthropic", _FakeAnthropicSDK)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        multi_provider_llm._get_anthropic_client()


# ---- package surface ----------------------------------------------------------------

def test_stage2_public_surface_and_version():
    assert llm_backends.__version__ == "0.1.1"
    assert llm_backends.resolve_model is resolve_model
    assert llm_backends.MODEL_ALIASES is MODEL_ALIASES
    assert callable(llm_backends.check_cli_availability)
