"""Tests for the system-prompt contract.

Before this change, every backend defaulted `role_description` to a
fiction-writing persona *and* the registry never forwarded the parameter, so
every call through `send_prompt()` silently shipped

    "You are a helpful fiction writing assistant. You will create original text only."

as its system prompt, with no way for a caller to override it. For a non-fiction
consumer that is wrong; for a coding agent "create original text only" is
actively harmful, since it discourages reproducing existing files.

The contract now: **no system prompt unless one is asked for**, and the
parameter reaches every backend through the registry.
"""

import pytest

import llm_backends
from llm_backends import multi_provider_llm
from llm_backends.multi_provider_llm import (
    FICTION_ROLE,
    _call_model_fn,
    send_prompt,
    send_prompt_meta,
)

from test_registry_and_dispatch import (  # reuse the established fakes
    _CapturingAnthropicClient,
    _CapturingChatClient,
)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(multi_provider_llm, "_openrouter_client", None)
    monkeypatch.setattr(multi_provider_llm, "_anthropic_client", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield


def _roles(captured):
    return [m["role"] for m in captured["messages"]]


# ---- the regression this file exists for -----------------------------------------


def test_no_system_prompt_by_default(monkeypatch):
    """The fiction persona must not appear unless asked for."""
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    send_prompt("hi", model="openrouter:deepseek/deepseek-chat")

    assert _roles(client.captured) == ["user"]
    assert "fiction" not in str(client.captured["messages"]).lower()


def test_fiction_role_is_opt_in_and_still_available(monkeypatch):
    """Existing callers restore the old behaviour in one line."""
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    send_prompt(
        "hi", model="openrouter:deepseek/deepseek-chat", role_description=FICTION_ROLE
    )

    msgs = client.captured["messages"]
    assert _roles(client.captured) == ["system", "user"]
    assert msgs[0]["content"] == FICTION_ROLE
    assert llm_backends.FICTION_ROLE == FICTION_ROLE  # exported at package level


def test_role_reaches_backend_through_the_registry(monkeypatch):
    """The parameter used to exist on the backends but was never forwarded."""
    client = _CapturingChatClient()
    monkeypatch.setattr(multi_provider_llm, "_get_openrouter_client", lambda: client)

    send_prompt_meta(
        "hi",
        model="openrouter:x/y",
        role_description="You are a terse coding assistant.",
    )

    assert client.captured["messages"][0] == {
        "role": "system",
        "content": "You are a terse coding assistant.",
    }


# ---- Anthropic takes a system kwarg, not a message --------------------------------


def test_anthropic_omits_system_kwarg_when_no_role(monkeypatch):
    """Anthropic rejects system=None: the param must be absent, not null."""
    client = _CapturingAnthropicClient()
    monkeypatch.setattr(multi_provider_llm, "_get_anthropic_client", lambda: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    send_prompt("hi", model="claude-haiku-4-5")

    assert "system" not in client.captured


def test_anthropic_sends_system_kwarg_when_role_given(monkeypatch):
    client = _CapturingAnthropicClient()
    monkeypatch.setattr(multi_provider_llm, "_get_anthropic_client", lambda: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    send_prompt("hi", model="claude-haiku-4-5", role_description="be terse")

    assert client.captured["system"] == "be terse"


# ---- graceful degradation in _call_model_fn ---------------------------------------
#
# Tests and older code monkeypatch two-arg entries into the registries; adding a
# kwarg must not break them. Degradation is per-kwarg, so a timeout-only entry
# keeps its timeout even when a role is also supplied.


def test_two_arg_entry_still_works():
    calls = []

    def two_arg(prompt, max_tokens):
        calls.append((prompt, max_tokens))
        return "ok"

    assert _call_model_fn(two_arg, "p", 10, None, "a role") == "ok"
    assert calls == [("p", 10)]


def test_timeout_only_entry_keeps_its_timeout_when_role_supplied():
    """The naive fix (one try, then bare call) would silently drop the timeout."""
    seen = {}

    def timeout_only(prompt, max_tokens, timeout=None):
        seen["timeout"] = timeout
        return "ok"

    assert _call_model_fn(timeout_only, "p", 10, 30, "a role") == "ok"
    assert seen["timeout"] == 30


def test_full_entry_receives_both():
    seen = {}

    def full(prompt, max_tokens, timeout=None, role_description=None):
        seen.update(timeout=timeout, role_description=role_description)
        return "ok"

    assert _call_model_fn(full, "p", 10, 30, "a role") == "ok"
    assert seen == {"timeout": 30, "role_description": "a role"}
