"""Tests for exogenous feature (X_future) integration with Nixtla backends.

Covers:
- fit/predict with X_future on stats and neural models
- Panel data with X_future
- Per-model gating (supports_exogenous tag)
- Column mismatch validation at predict time
- Combined X_actual + X_future
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou_nixtla.stats import (
    AutoARIMAForecaster,
    NaiveForecaster,
)


def _make_daily_data(length=100, n_targets=1, n_future_features=1, seed=42, panel=False, n_groups=2):
    """Generate y and X_future test data with daily frequency.

    Returns (y, X_future) where X_future covers `length + horizon` days
    to provide future values at predict time.
    """
    rng = np.random.default_rng(seed)
    horizon = 5
    total_length = length + horizon

    time_train = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=length - 1),
        interval="1d",
        eager=True,
    )
    time_future = pl.datetime_range(
        start=datetime(2020, 1, 1) + timedelta(days=length),
        end=datetime(2020, 1, 1) + timedelta(days=total_length - 1),
        interval="1d",
        eager=True,
    )

    if panel:
        y = pl.DataFrame({"time": time_train})
        X_future_fit = pl.DataFrame({"time": time_train})
        X_future_predict = pl.DataFrame({"time": time_future})
        for group_idx in range(n_groups):
            for i in range(n_targets):
                col_name = f"group_{group_idx}__y_{i}"
                y = y.with_columns(pl.Series(col_name, rng.random(length) * 100))
            for i in range(n_future_features):
                col_name = f"group_{group_idx}__f_{i}"
                X_future_fit = X_future_fit.with_columns(pl.Series(col_name, rng.random(length) * 10))
                X_future_predict = X_future_predict.with_columns(pl.Series(col_name, rng.random(horizon) * 10))
    else:
        y = pl.DataFrame({"time": time_train})
        for i in range(n_targets):
            y = y.with_columns(pl.Series(f"y_{i}", rng.random(length) * 100))

        X_future_fit = pl.DataFrame({"time": time_train})
        X_future_predict = pl.DataFrame({"time": time_future})
        for i in range(n_future_features):
            X_future_fit = X_future_fit.with_columns(pl.Series(f"f_{i}", rng.random(length) * 10))
            X_future_predict = X_future_predict.with_columns(pl.Series(f"f_{i}", rng.random(horizon) * 10))

    return y, X_future_fit, X_future_predict, horizon


class TestStatsExogenous:
    """Tests for X_future with statsforecast models."""

    def test_fit_predict_with_x_future(self):
        """AutoARIMA should fit and predict with X_future."""
        y, X_future_fit, X_future_predict, horizon = _make_daily_data()

        forecaster = AutoARIMAForecaster()
        forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=horizon)
        result = forecaster.predict(X_future=X_future_predict)

        assert "time" in result.columns
        assert result.shape[0] == horizon

    def test_fit_predict_panel_with_x_future(self):
        """AutoARIMA should fit and predict with X_future on panel data."""
        y, X_future_fit, X_future_predict, horizon = _make_daily_data(panel=True)

        forecaster = AutoARIMAForecaster()
        forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=horizon)
        result = forecaster.predict(X_future=X_future_predict)

        assert "time" in result.columns
        assert result.shape[0] == horizon
        # Should have columns for both groups
        value_cols = [c for c in result.columns if c not in ("time", "observed_time", "vintage_time")]
        assert len(value_cols) == 2

    def test_fit_predict_with_x_actual_and_x_future(self):
        """AutoARIMA should fit with both X_actual and X_future.

        When both X_actual and X_future are provided, both sets of columns
        end up in the Nixtla training DataFrame. Verify that fit succeeds
        and the model stores the correct attributes.
        """
        rng = np.random.default_rng(123)
        length = 100
        horizon = 5
        time_train = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=length - 1),
            interval="1d",
            eager=True,
        )

        # Create y that depends on the features
        x_actual_vals = rng.random(length) * 10
        x_future_vals_fit = rng.random(length) * 5
        y_vals = 50 + 2 * x_actual_vals + 1.5 * x_future_vals_fit + rng.normal(0, 1, length)

        y = pl.DataFrame({"time": time_train, "y_0": y_vals})
        X_actual = pl.DataFrame({"time": time_train, "actual_feat": x_actual_vals})
        X_future_fit = pl.DataFrame({"time": time_train, "future_feat": x_future_vals_fit})

        forecaster = AutoARIMAForecaster()
        forecaster.fit(
            y,
            X_actual=X_actual,
            X_future=X_future_fit,
            forecasting_horizon=horizon,
        )

        # Verify X_future columns were stored
        assert forecaster.futr_exog_columns_ == ["future_feat"]


class TestNeuralExogenous:
    """Tests for X_future with neuralforecast models."""

    def test_fit_predict_with_x_future(self):
        """NHITS should fit and predict with X_future."""
        from yohou_nixtla.neural import NHITSForecaster

        y, X_future_fit, X_future_predict, horizon = _make_daily_data()

        forecaster = NHITSForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=horizon)
        result = forecaster.predict(X_future=X_future_predict)

        assert "time" in result.columns
        assert result.shape[0] == horizon


class TestExogenousGating:
    """Tests for per-model exogenous gating."""

    def test_unsupported_model_raises_on_fit(self):
        """NaiveForecaster should raise ValueError when X_future is provided at fit."""
        y, X_future_fit, _, _ = _make_daily_data()

        forecaster = NaiveForecaster()
        with pytest.raises(ValueError, match="does not support exogenous features"):
            forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=5)

    def test_unsupported_model_raises_on_predict(self, daily_y_X_factory):
        """NaiveForecaster should raise ValueError when X_future is provided at predict."""
        y, _ = daily_y_X_factory(length=100)

        forecaster = NaiveForecaster()
        forecaster.fit(y, forecasting_horizon=5)

        _, _, X_future_predict, _ = _make_daily_data()
        with pytest.raises(ValueError, match="does not support exogenous features"):
            forecaster.predict(X_future=X_future_predict)

    def test_unsupported_neural_model_raises_on_fit(self):
        """NBEATSForecaster should raise ValueError when X_future is provided."""
        from yohou_nixtla.neural import NBEATSForecaster

        y, X_future_fit, _, _ = _make_daily_data()

        forecaster = NBEATSForecaster(input_size=12, max_steps=5)
        with pytest.raises(ValueError, match="does not support exogenous features"):
            forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=5)

    def test_x_forecast_raises(self):
        """Any model should raise ValueError when X_forecast is provided."""
        y, X_future_fit, _, _ = _make_daily_data()

        forecaster = AutoARIMAForecaster()
        with pytest.raises(ValueError, match="do not support X_forecast"):
            forecaster.fit(y, X_forecast=X_future_fit, forecasting_horizon=5)

    def test_x_forecast_predict_raises(self, daily_y_X_factory):
        """Predict with X_forecast should raise ValueError."""
        y, _ = daily_y_X_factory(length=100)

        forecaster = AutoARIMAForecaster()
        forecaster.fit(y, forecasting_horizon=5)

        _, X_future_fit, _, _ = _make_daily_data()
        with pytest.raises(ValueError, match="do not support X_forecast"):
            forecaster.predict(X_forecast=X_future_fit)


class TestExogenousValidation:
    """Tests for predict-time column validation."""

    def test_column_mismatch_raises(self):
        """Predict with different X_future columns should raise ValueError."""
        y, X_future_fit, _, horizon = _make_daily_data()

        forecaster = AutoARIMAForecaster()
        forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=horizon)

        # Create X_future with different column names
        time_future = pl.datetime_range(
            start=datetime(2020, 4, 11),
            end=datetime(2020, 4, 11) + timedelta(days=horizon - 1),
            interval="1d",
            eager=True,
        )
        X_future_wrong = pl.DataFrame({
            "time": time_future,
            "wrong_column": [1.0] * horizon,
        })

        with pytest.raises(ValueError, match="do not match"):
            forecaster.predict(X_future=X_future_wrong)


class TestGetTag:
    """Tests for the _get_tag helper method."""

    def test_get_tag_returns_none_for_unknown(self):
        """_get_tag should return None for a tag not defined anywhere."""
        forecaster = NaiveForecaster()
        assert forecaster._get_tag("nonexistent_tag") is None


class TestXFutureToNixtla:
    """Tests for x_future_to_nixtla conversion edge cases."""

    def test_empty_x_future_raises(self):
        """X_future with only a time column should raise ValueError."""
        from yohou_nixtla._conversion import x_future_to_nixtla

        X = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
        with pytest.raises(ValueError, match="at least one feature column"):
            x_future_to_nixtla(X, y_columns=["y_0"])

    def test_panel_suffix_matching(self):
        """Panel X_future should use suffix matching when prefix doesn't match."""
        from yohou_nixtla._conversion import x_future_to_nixtla

        time = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        # y columns use prefix__suffix format: target__store_1
        y_columns = ["target__store_1", "target__store_2"]
        # X_future uses variable__suffix format: feature__store_1
        X_future = pl.DataFrame({
            "time": time,
            "feature__store_1": [1.0, 2.0],
            "feature__store_2": [3.0, 4.0],
        })

        result = x_future_to_nixtla(X_future, y_columns)
        assert set(result["unique_id"].unique()) == {"target__store_1", "target__store_2"}
        assert "feature" in result.columns

    def test_panel_skips_uid_without_separator(self):
        """Panel X_future should skip y_columns entries without __ separator."""
        from yohou_nixtla._conversion import x_future_to_nixtla

        time = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        # y_columns has both panel and non-panel entries
        y_columns = ["global_col", "group_0__y_0"]
        X_future = pl.DataFrame({
            "time": time,
            "group_0__f_0": [1.0, 2.0],
        })

        result = x_future_to_nixtla(X_future, y_columns)
        # Only the panel entry should be in the result
        assert list(result["unique_id"].unique()) == ["group_0__y_0"]


class TestNeuralPanelExogenous:
    """Tests for neural model with panel exogenous features."""

    def test_fit_with_panel_x_future_deduplicates_futr_exog_list(self):
        """Neural model should deduplicate panel X_future columns into futr_exog_list."""
        from yohou_nixtla.neural import NHITSForecaster

        rng = np.random.default_rng(42)
        length = 100
        horizon = 5
        time_train = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=length - 1),
            interval="1d",
            eager=True,
        )
        time_future = pl.datetime_range(
            start=datetime(2020, 1, 1) + timedelta(days=length),
            end=datetime(2020, 1, 1) + timedelta(days=length + horizon - 1),
            interval="1d",
            eager=True,
        )

        y = pl.DataFrame({
            "time": time_train,
            "group_0__y_0": rng.random(length) * 100,
            "group_1__y_0": rng.random(length) * 100,
        })
        X_future_fit = pl.DataFrame({
            "time": time_train,
            "group_0__f_0": rng.random(length) * 10,
            "group_1__f_0": rng.random(length) * 10,
        })
        X_future_predict = pl.DataFrame({
            "time": time_future,
            "group_0__f_0": rng.random(horizon) * 10,
            "group_1__f_0": rng.random(horizon) * 10,
        })

        forecaster = NHITSForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, X_future=X_future_fit, forecasting_horizon=horizon)

        # futr_exog_list should have deduplicated names (just "f_0")
        assert forecaster.params.get("futr_exog_list") == ["f_0"]

        result = forecaster.predict(X_future=X_future_predict)
        assert result.shape[0] == horizon
