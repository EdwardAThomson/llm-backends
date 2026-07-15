"""Tests for the stage-2 CLI hardening merged from the analyzer and NovelWriter.

All fakes/monkeypatch: no real CLIs, no subprocesses, no network. Covers:
- subprocess key-stripping default ON (assumption A4) and the
  strip_provider_keys=False opt-out, per interface;
- the codex bubblewrap/user-namespace workaround probe logic (fake subprocess,
  no real `unshare` required);
- is_available() static probes and check_cli_availability();
- the claude model heuristic including "fable".
"""

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

from llm_backends import codex_interface as codex_mod
from llm_backends import check_cli_availability
from llm_backends.claude_cli_interface import ClaudeCliInterface
from llm_backends.codex_interface import CodexInterface
from llm_backends.gemini_cli_interface import GeminiCliInterface


class _Cap:
    cmd = None
    env = "UNCAPTURED"  # distinguishes "never captured" from env=None


def _patch_run(monkeypatch, *, stdout="", last_message=None):
    """Patch subprocess.run + shutil.which; capture command and env kwarg."""
    cap = _Cap()

    def fake_run(cmd, **kw):
        cap.cmd = cmd
        cap.env = kw.get("env")
        if last_message is not None and "--output-last-message" in cmd:
            path = cmd[cmd.index("--output-last-message") + 1]
            with open(path, "w", encoding="utf-8") as f:
                f.write(last_message)
        return SimpleNamespace(stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
    return cap


def _no_userns(monkeypatch):
    """Pin the codex userns probe to 'not needed' so it never runs a probe."""
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", [])


# ---- key-stripping (assumption A4: default ON, explicit opt-out) ------------------

def test_codex_strips_openai_key_by_default(monkeypatch):
    _no_userns(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-metered")
    cap = _patch_run(monkeypatch, last_message="prose")

    out = CodexInterface().generate("hi")

    assert out == "prose"
    assert isinstance(cap.env, dict)
    assert "OPENAI_API_KEY" not in cap.env
    # It's a filtered COPY, not an empty env: the rest of the environment rides along.
    assert "PATH" in cap.env
    # And the parent process environment is untouched.
    assert os.environ["OPENAI_API_KEY"] == "sk-metered"


def test_codex_strip_opt_out_inherits_parent_env(monkeypatch):
    _no_userns(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deliberate")
    cap = _patch_run(monkeypatch, last_message="prose")

    CodexInterface(strip_provider_keys=False).generate("hi")

    # env=None means subprocess inherits the full parent environment (key included).
    assert cap.env is None


def test_claude_cli_strips_both_anthropic_key_spellings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-metered")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-legacy")
    cap = _patch_run(monkeypatch, stdout='{"result": "ok"}')

    out = ClaudeCliInterface().generate("hi")

    assert out == "ok"
    assert isinstance(cap.env, dict)
    assert "ANTHROPIC_API_KEY" not in cap.env
    assert "CLAUDE_API_KEY" not in cap.env
    assert "PATH" in cap.env
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-metered"


def test_claude_cli_strip_opt_out(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-deliberate")
    cap = _patch_run(monkeypatch, stdout='{"result": "ok"}')

    ClaudeCliInterface(strip_provider_keys=False).generate("hi")

    assert cap.env is None


def test_gemini_cli_strips_gemini_and_google_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-metered")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-metered-too")
    cap = _patch_run(monkeypatch, stdout="prose")

    out = GeminiCliInterface().generate("hi")

    assert out == "prose"
    assert isinstance(cap.env, dict)
    assert "GEMINI_API_KEY" not in cap.env
    assert "GOOGLE_API_KEY" not in cap.env
    assert "PATH" in cap.env


def test_gemini_cli_strip_opt_out(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-deliberate")
    cap = _patch_run(monkeypatch, stdout="prose")

    GeminiCliInterface(strip_provider_keys=False).generate("hi")

    assert cap.env is None


# ---- codex bubblewrap/user-namespace workaround (probe logic only) ----------------

def test_userns_prefix_empty_when_mapping_allowed(monkeypatch):
    """Probe succeeds -> codex's own bwrap sandbox works -> no wrapping."""
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")

    probes = []

    def fake_run(cmd, **kw):
        probes.append(cmd)
        return SimpleNamespace(stdout="", stderr="")  # success

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert codex_mod._userns_launch_prefix() == []
    assert probes == [["unshare", "--user", "--map-root-user", "true"]]


def test_userns_prefix_wraps_when_mapping_restricted(monkeypatch):
    """Probe fails (apparmor_restrict_unprivileged_userns) -> identity-mapped wrap."""
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")

    def failing_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", failing_run)

    prefix = codex_mod._userns_launch_prefix()
    uid, gid = os.getuid(), os.getgid()
    assert prefix == [
        "unshare", "--user",
        f"--map-users={uid}:{uid}:1",
        f"--map-groups={gid}:{gid}:1",
        "--",
    ]


def test_userns_wrapped_codex_disables_inner_sandbox(monkeypatch):
    """When wrapped, codex's own (broken) sandbox is disabled: the outer
    namespace is the sandbox. When not wrapped, the read-only flags stay
    (covered by test_agent_cwd.py's codex flags test)."""
    fake_prefix = ["unshare", "--user", "--map-users=1:1:1", "--map-groups=1:1:1", "--"]
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", fake_prefix)
    cap = _patch_run(monkeypatch, last_message="prose")

    out = CodexInterface().generate("hi")

    assert out == "prose"
    assert cap.cmd[:5] == fake_prefix
    assert "--dangerously-bypass-approvals-and-sandbox" in cap.cmd
    assert "--sandbox" not in cap.cmd
    assert "--ask-for-approval" not in cap.cmd


def test_userns_prefix_empty_on_non_linux(monkeypatch):
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", None)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")

    def must_not_run(cmd, **kw):
        raise AssertionError("probe must not run off Linux")

    monkeypatch.setattr(subprocess, "run", must_not_run)

    assert codex_mod._userns_launch_prefix() == []


def test_userns_prefix_empty_when_helpers_missing(monkeypatch):
    """No setuid newuidmap/newgidmap -> we cannot build the mapped namespace."""
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        shutil, "which",
        lambda b: None if b in ("newuidmap", "newgidmap") else f"/usr/bin/{b}",
    )

    def must_not_run(cmd, **kw):
        raise AssertionError("probe must not run without the helpers")

    monkeypatch.setattr(subprocess, "run", must_not_run)

    assert codex_mod._userns_launch_prefix() == []


def test_userns_probe_result_is_cached(monkeypatch):
    monkeypatch.setattr(codex_mod, "_userns_prefix_cache", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")

    calls = {"n": 0}

    def counting_run(cmd, **kw):
        calls["n"] += 1
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", counting_run)

    assert codex_mod._userns_launch_prefix() == []
    assert codex_mod._userns_launch_prefix() == []
    assert calls["n"] == 1  # probed once per process, then cached


# ---- is_available() static probes (from NovelWriter) ------------------------------

def test_is_available_true_when_binary_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
    assert CodexInterface.is_available() is True
    assert ClaudeCliInterface.is_available() is True
    assert GeminiCliInterface.is_available() is True


def test_is_available_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda b: None)
    assert CodexInterface.is_available() is False
    assert ClaudeCliInterface.is_available() is False
    assert GeminiCliInterface.is_available() is False
    # A custom binary path is probed, not the default.
    monkeypatch.setattr(shutil, "which", lambda b: "/opt/x" if b == "my-codex" else None)
    assert CodexInterface.is_available("my-codex") is True


def test_check_cli_availability_aggregates_probes(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda b: f"/usr/bin/{b}" if b == "codex" else None
    )
    assert check_cli_availability() == {
        "codex": True,
        "gemini-cli": False,
        "claude-cli": False,
    }


# ---- claude model heuristic (analyzer: includes "fable") --------------------------

def test_claude_model_heuristic_includes_fable(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda b: f"/usr/bin/{b}")
    assert ClaudeCliInterface(model="fable").model == "fable"
    assert ClaudeCliInterface(model="sonnet").model == "sonnet"
    # Cross-backend defaults are still ignored, not forwarded to --model.
    assert ClaudeCliInterface(model="gpt-5.5").model is None


def test_claude_cli_forwards_fable_model_flag(monkeypatch):
    cap = _patch_run(monkeypatch, stdout='{"result": "ok"}')
    ClaudeCliInterface(model="fable").generate("hi")
    assert cap.cmd[cap.cmd.index("--model") + 1] == "fable"
