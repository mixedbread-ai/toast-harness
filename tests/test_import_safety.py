"""Import safety is a contract: the package imports clean and offline.

Each case runs in a fresh interpreter, because the guarantee is about what an
import does the *first* time -- an already-imported module proves nothing. The
optional heavy dependencies are poisoned rather than merely absent so that a
regression fails here instead of on a machine that happens to have them.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import site
import subprocess
import sys
import textwrap

import pytest

# searcher_spec is listed explicitly: it is opt-in, so `agent_harness/__init__.py`
# never imports it and it would otherwise sit outside the contract entirely.
MODULES = (
    "agent_harness",
    "agent_harness.searcher_spec",
    "agent_harness.aio",
    "agent_harness.testing",
)

# weave/wandb/torch/oauth_codex must never be touched by this package -- generation
# is injected, so no provider client is bundled; gigatoken ships as a core dependency
# but is imported lazily at first token-counter use, never at package import.
_POISONED = ("weave", "wandb", "gigatoken", "torch", "oauth_codex")

_GUARD = """
import socket, sys, types
from importlib.machinery import ModuleSpec


class _Forbidden(BaseException):
    pass


class _Poison(types.ModuleType):
    # Availability probes (importlib.util.find_spec, `hasattr(mod, "__all__")`)
    # are legitimate and go through the dunders, so only real use is fatal.
    def __getattribute__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return types.ModuleType.__getattribute__(self, name)
        raise _Forbidden(f"optional dependency touched at import time: {name}")


for _name in %(poisoned)r:
    _module = _Poison(_name)
    _module.__spec__ = ModuleSpec(_name, loader=None)
    sys.modules[_name] = _module


def _no_network(*args, **kwargs):
    raise _Forbidden("network access at import time")


socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.create_connection = _no_network
socket.getaddrinfo = _no_network
"""


def _import_in_fresh_interpreter(module: str) -> subprocess.CompletedProcess[str]:
    script = _GUARD % {"poisoned": _POISONED} + f"\nimport {module}\n"
    # The ambient environment is passed through deliberately: no environment
    # variable may make this package touch a poisoned module or the network at
    # import, so scrubbing any name here would only hide a regression that sets
    # one up as an opt-in.
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module", MODULES)
def test_import_is_side_effect_free(module: str) -> None:
    result = _import_in_fresh_interpreter(module)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module", MODULES)
def test_module_resolves_from_the_installed_distribution(module: str) -> None:
    """Guards against a src/ shadow silently replacing the wheel under test."""
    _installed_files()  # editable install: resolution location is not meaningful
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__file__)"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    resolved = pathlib.Path(result.stdout.strip())
    assert resolved.name
    # The subprocess is sys.executable, so this interpreter's own site dirs are
    # the ones it resolved against. User site counts: `pip install --user` is a
    # real wheel install, and it is not in getsitepackages().
    site_dirs = [pathlib.Path(d) for d in (*site.getsitepackages(), site.getusersitepackages())]

    assert any(resolved.is_relative_to(d) for d in site_dirs), (resolved, site_dirs)


def _installed_files() -> list[str]:
    """The distribution's file list, or a skip if this is an editable install.

    An editable install records a `.pth` shim instead of the package tree, so it
    can say nothing about wheel contents. Its callers are packaging assertions
    and only mean something in the CI gate, which installs the built wheel.
    """
    files = [str(f) for f in importlib.metadata.files("toast-harness") or []]
    if not any(f.endswith("agent_harness/__init__.py") for f in files):
        pytest.skip("editable install: wheel contents are not observable")
    return files


def test_the_wheel_ships_the_package_and_nothing_else() -> None:
    """The distribution ships exactly one top-level package: the harness."""
    tops = {f.split("/")[0] for f in _installed_files()}
    modules = {t for t in tops if not t.endswith((".dist-info", ".pth")) and "." not in t}

    assert modules == {"agent_harness"}, modules


def test_py_typed_is_shipped() -> None:
    """Typed package data is easy to drop in a packaging change and silent when lost."""
    assert any(f.endswith("agent_harness/py.typed") for f in _installed_files())


def test_searcher_spec_resources_are_shipped() -> None:
    """The contract schema and goldens are read at runtime via importlib.resources.

    They are the package's only non-``py.typed`` data files. A packaging change
    that drops them surfaces as a FileNotFoundError from
    ``load_searcher_contract_schema`` rather than as a packaging error.
    """
    installed = _installed_files()
    for resource in (
        "agent_harness/searcher_spec/schemas/mixedbread_searcher.v1.schema.json",
        "agent_harness/searcher_spec/fixtures/submit_ranking.top5.strict.v1.json",
    ):
        assert any(f.endswith(resource) for f in installed), resource
