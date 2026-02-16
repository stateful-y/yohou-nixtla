"""Tests for marimo notebook examples.

This module validates that all notebook examples in the examples/ directory
can be executed successfully as Python scripts.

Marimo notebooks can be run directly with Python:
https://docs.marimo.io/getting_started/quickstart/#run-as-scripts
"""

import pathlib
import subprocess
import sys

import pytest

# Discover all example notebooks recursively
EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"

_notebook_files = sorted(EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.example
@pytest.mark.parametrize(
    "notebook_file",
    _notebook_files,
    ids=[str(f.relative_to(EXAMPLES_DIR)) for f in _notebook_files],
)
def test_notebook_runs_without_error(notebook_file: pathlib.Path) -> None:
    """Test that a notebook example runs successfully as a script.

    Parameters
    ----------
    notebook_file : pathlib.Path
        Path to the notebook file to test.

    """
    result = subprocess.run(
        [sys.executable, str(notebook_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Notebook {notebook_file.name} failed with:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
