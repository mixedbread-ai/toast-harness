"""Importing the tool surface must not read a .env or mutate os.environ."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ENV_LINE = "MXBAI_API_KEY=from-dotenv\n"


def _python(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    # Strip every accepted key alias, not just the canonical prefix: an exported
    # MBREAD_API_KEY or MIXEDBREAD_API_KEY would satisfy the resolver before the
    # .env fallback and fail the lazy-read assertion below.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MXBAI_", "MBREAD_", "MIXEDBREAD_"))
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


@pytest.fixture
def dotenv_dir(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text(_ENV_LINE)
    return tmp_path


def test_import_does_not_load_dotenv(dotenv_dir: Path) -> None:
    """Import behavior must not depend on the working directory."""
    result = _python(
        """
        import os
        from agent_harness.tools import functions  # noqa: F401

        assert "MXBAI_API_KEY" not in os.environ, "import mutated os.environ from .env"
        """,
        cwd=dotenv_dir,
    )

    assert result.returncode == 0, result.stderr


def test_resolvers_still_read_dotenv_lazily(dotenv_dir: Path) -> None:
    result = _python(
        """
        from agent_harness.tools.functions import resolve_mixedbread_api_key

        assert resolve_mixedbread_api_key() == "from-dotenv"
        """,
        cwd=dotenv_dir,
    )

    assert result.returncode == 0, result.stderr
