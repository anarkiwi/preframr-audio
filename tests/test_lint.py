"""Run the project's linters as part of the test suite. Catches lint errors locally before CI does. The CI ``Lint`` job runs the same commands, so a green test_lint here means a green CI lint."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LINT_TARGETS = ["preframr_audio", "tests"]


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


@pytest.mark.skipif(not _have("ruff"), reason="ruff not installed")
def test_ruff_check_clean():
    """``ruff check`` must report zero findings on src + tests."""
    result = subprocess.run(
        ["ruff", "check", *LINT_TARGETS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "ruff check found issues:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.skipif(not _have("black"), reason="black not installed")
def test_black_check_clean():
    """``black --check`` must report no would-reformat files on src + tests."""
    result = subprocess.run(
        ["black", "--check", *LINT_TARGETS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "black --check found files that need reformatting:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.skipif(not _have("pylint"), reason="pylint not installed")
def test_pylint_clean():
    """``pylint`` must report no findings on src + tests. The pyproject.toml ``[tool.pylint.main]`` enable list defines which checks are active -- this test covers the gap where ruff doesn't flag pylint-only findings (e.g. W0612 unused-variable on tuple-unpacked assignments)."""
    result = subprocess.run(
        ["pylint", *LINT_TARGETS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "pylint found issues:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
