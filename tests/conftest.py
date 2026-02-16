"""Test configuration and fixtures for yohou-nixtla."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest


def run_checks(
    estimator: Any,
    checks: Generator[tuple[str, Any, dict], None, None],
    *,
    expected_failures: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Run all checks from a generator, collecting and reporting all failures.

    Unlike a simple for-loop, this function does **not** stop at the first
    failure. All checks are executed and a single ``pytest.fail`` is raised
    at the end summarising every unexpected failure (and every expected
    failure that unexpectedly passed).

    Parameters
    ----------
    estimator : object
        Fitted estimator instance passed as the first positional argument
        to each check function.
    checks : generator of (str, callable, dict)
        Output of a ``_yield_yohou_*_checks`` generator.
    expected_failures : set of str, optional
        Check names that are expected to fail.  Unexpected passes are
        reported alongside unexpected failures.

    """
    failures: list[str] = []
    xfail_passed: list[str] = []

    for check_name, check_func, check_kwargs in checks:
        passes_estimator = (
            "splitter" in check_kwargs
            or "splitter_class" in check_kwargs
            or "scorer" in check_kwargs
            or "scorer_class" in check_kwargs
        )

        try:
            if passes_estimator:
                check_func(**check_kwargs)
            else:
                check_func(estimator, **check_kwargs)
        except Exception as exc:
            if check_name in expected_failures:
                continue
            failures.append(f"  {check_name}: {type(exc).__name__}: {exc}")
        else:
            if check_name in expected_failures:
                xfail_passed.append(check_name)

    messages: list[str] = []
    if failures:
        messages.append(f"{len(failures)} check(s) failed:\n" + "\n".join(failures))
    if xfail_passed:
        xfail_lines = "\n".join(f"  {name}" for name in xfail_passed)
        messages.append(f"{len(xfail_passed)} expected failure(s) unexpectedly passed:\n" + xfail_lines)

    if messages:
        pytest.fail("\n\n".join(messages))


@pytest.fixture
def daily_y_X_factory():
    """Factory for generating daily-frequency (y, X) tuples.

    Returns a callable that produces daily time series, suitable for
    statsforecast models that require regular date-based frequencies.
    """

    def _factory(length=100, n_targets=1, n_features=0, seed=42, panel=False, n_groups=2):
        """Generate daily time series test data.

        Parameters
        ----------
        length : int
            Number of days.
        n_targets : int
            Number of target features.
        n_features : int
            Number of exogenous features (0 for None).
        seed : int
            Random seed.
        panel : bool
            Whether to create panel data with ``__`` separator.
        n_groups : int
            Number of panel groups when ``panel=True``.

        Returns
        -------
        y : pl.DataFrame
            Target data with ``time`` column (daily frequency).
        X : pl.DataFrame or None
            Features with ``time`` column, or None.

        """
        rng = np.random.default_rng(seed)

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=length - 1),
            interval="1d",
            eager=True,
        )

        if panel:
            # Convention: <entity>__<variable> (e.g., group_0__y_0)
            y = pl.DataFrame({"time": time})
            for group_idx in range(n_groups):
                for i in range(n_targets):
                    base_values = rng.random(length) * 100
                    variation = group_idx * 10
                    col_name = f"group_{group_idx}__y_{i}"
                    y = y.with_columns(pl.Series(col_name, base_values + variation))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time})
                for group_idx in range(n_groups):
                    for i in range(n_features):
                        base_values = rng.random(length) * 10
                        variation = group_idx * 1.0
                        col_name = f"group_{group_idx}__X_{i}"
                        X = X.with_columns(pl.Series(col_name, base_values + variation))
        else:
            y = pl.DataFrame({"time": time})
            for i in range(n_targets):
                y = y.with_columns(pl.Series(f"y_{i}", rng.random(length) * 100))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time})
                for i in range(n_features):
                    X = X.with_columns(pl.Series(f"X_{i}", rng.random(length) * 10))

        return y, X

    return _factory
