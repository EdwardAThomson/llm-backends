"""Package-level contract tests (new in the extraction, not ported).

Two guarantees from the inventory doc:
- Lazy-import contract (section 7.2): `import llm_backends` and constructing
  MultiProviderInterface must succeed with NO provider SDK installed. The
  whole suite already runs in a pytest-only venv, but the subprocess test
  below enforces it even when the developer's environment happens to have the
  SDKs, by blocking their imports outright.
- Registry integrity: every model key maps to a callable in both registries.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_BLOCKED_IMPORT_SCRIPT = """
import importlib.abc
import sys

BLOCKED = ("openai", "anthropic", "google")


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"blocked for lazy-import contract test: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())
for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]

import llm_backends
from llm_backends import (
    ClaudeCliInterface,   # noqa: F401 - import surface must resolve
    CodexInterface,       # noqa: F401
    GeminiCliInterface,   # noqa: F401
    MultiProviderInterface,
    llm_interface,        # noqa: F401
    multi_provider_llm,
)

# The SDK degradation pattern: module-level names become None, nothing raises.
assert multi_provider_llm.OpenAI is None
assert multi_provider_llm.genai is None
assert multi_provider_llm.anthropic is None

# Constructing the explicit-instance interface needs no SDK at all.
client = MultiProviderInterface(model="gpt-5.5", timeout=42)
assert client.model == "gpt-5.5" and client.timeout == 42

print("LAZY_IMPORT_OK")
"""


def test_import_and_construct_without_provider_sdks():
    """import llm_backends + MultiProviderInterface() with all SDKs blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "LAZY_IMPORT_OK" in result.stdout


def test_every_registry_key_maps_to_a_callable():
    from llm_backends import multi_provider_llm

    assert multi_provider_llm._model_config_meta, "meta registry must not be empty"
    for key, fn in multi_provider_llm._model_config_meta.items():
        assert callable(fn), f"meta registry entry {key!r} is not callable"
    for key, fn in multi_provider_llm._model_config.items():
        assert callable(fn), f"text registry entry {key!r} is not callable"
    # And the public model list is exactly the registry's key set.
    assert set(multi_provider_llm.get_supported_models()) == set(
        multi_provider_llm._model_config_meta
    )
