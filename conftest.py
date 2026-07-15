"""Pytest configuration shared by every pytest invocation in this repository.

Both the test suite (``pytest tests``) and the docstring session
(``pytest --doctest-modules src/yohou_nixtla``) load this root conftest, so
warning filters registered here apply to both.
"""

from __future__ import annotations

import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register warning filters that depend on the installed pandas version.

    ``pandas.errors.Pandas4Warning`` only exists on pandas >= 3.0. Naming it in
    the static ``filterwarnings`` list in ``pyproject.toml`` makes pytest abort
    while parsing its own config on pandas 2.x, which this project supports
    (``pandas>=2.2``) and which dependency resolution now selects by default
    because ``statsforecast>=2.1`` requires ``pandas<3``. Registering the filter
    at runtime keeps it active on pandas 3.x without breaking pandas 2.x.

    Parameters
    ----------
    config : pytest.Config
        Configuration object for the current pytest run.

    """
    if hasattr(pd.errors, "Pandas4Warning"):
        config.addinivalue_line(
            "filterwarnings",
            "ignore::pandas.errors.Pandas4Warning",
        )
