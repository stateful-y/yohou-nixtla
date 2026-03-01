"""Tests for the data conversion module."""

from datetime import datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from yohou_nixtla._conversion import infer_freq, nixtla_to_yohou, yohou_to_nixtla


class TestInferFreq:
    """Tests for ``infer_freq`` frequency inference."""

    def test_daily(self):
        """Daily frequency is inferred as 'D'."""
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True),
            "value": list(range(10)),
        })
        assert infer_freq(y) == "D"

    def test_hourly(self):
        """Hourly frequency is inferred as 'h'."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1), end=datetime(2020, 1, 1, 9), interval="1h", eager=True
            ),
            "value": list(range(10)),
        })
        assert infer_freq(y) == "h"

    def test_secondly(self):
        """Second-level frequency is inferred as 's'."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(seconds=9),
                interval="1s",
                eager=True,
            ),
            "value": list(range(10)),
        })
        assert infer_freq(y) == "s"

    def test_monthly(self):
        """Monthly frequency is inferred as 'MS'."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1), end=datetime(2020, 10, 1), interval="1mo", eager=True
            ),
            "value": list(range(10)),
        })
        assert infer_freq(y) == "MS"

    def test_single_row_raises(self):
        """A single-row DataFrame raises ValueError."""
        y = pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [1],
        })
        with pytest.raises(ValueError, match="at least 2"):
            infer_freq(y)

    def test_unknown_frequency_raises(self):
        """Unknown frequency mapping raises ValueError."""
        # Create a DataFrame with 2-second intervals (not in mapping)
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(seconds=18),
                interval="2s",
                eager=True,
            ),
            "value": list(range(10)),
        })
        with pytest.raises(ValueError, match="Cannot map polars interval"):
            infer_freq(y)


class TestYohouToNixtla:
    """Tests for ``yohou_to_nixtla`` wide-to-long conversion."""

    def test_univariate_global(self, daily_y_X_factory):
        """Single-column non-panel data converts correctly."""
        y, _ = daily_y_X_factory(length=10, n_targets=1)
        df = yohou_to_nixtla(y)

        assert list(df.columns) == ["unique_id", "ds", "y"]
        assert len(df) == 10
        assert df["unique_id"].unique().tolist() == ["y_0"]

    def test_multivariate_global(self, daily_y_X_factory):
        """Multi-column non-panel data creates one unique_id per column."""
        y, _ = daily_y_X_factory(length=10, n_targets=3)
        df = yohou_to_nixtla(y)

        assert len(df) == 30  # 10 rows * 3 series
        assert sorted(df["unique_id"].unique()) == ["y_0", "y_1", "y_2"]

    def test_panel_data(self, daily_y_X_factory):
        """Panel data preserves unique_ids from prefixed columns."""
        y, _ = daily_y_X_factory(length=10, n_targets=1, panel=True, n_groups=2)
        df = yohou_to_nixtla(y)

        assert sorted(df["unique_id"].unique()) == ["group_0__y_0", "group_1__y_0"]
        assert len(df) == 20  # 10 rows * 2 groups

    def test_with_exogenous(self, daily_y_X_factory):
        """Exogenous features are merged into the long-format output."""
        y, X = daily_y_X_factory(length=10, n_targets=1, n_features=2)
        df = yohou_to_nixtla(y, X)

        assert "X_0" in df.columns
        assert "X_1" in df.columns
        assert len(df) == 10

    def test_with_panel_exogenous_prefix_matching(self):
        """Panel exogenous matched by prefix (convention: <entity>__<variable>)."""
        # Convention 1: entity prefix shared between y and X columns
        # y: group_0__y_0, group_1__y_0  →  X: group_0__X_0, group_1__X_0
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "group_0__y_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group_1__y_0": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        X = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "group_0__X_0": [10.0, 20.0, 30.0, 40.0, 50.0],
            "group_1__X_0": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        df = yohou_to_nixtla(y, X)

        # Prefix matching: group_0__X_0 matches group_0__y_0 by prefix "group_0"
        assert "X_0" in df.columns
        assert len(df) == 10
        g0_rows = df[df["unique_id"] == "group_0__y_0"]
        assert g0_rows["X_0"].iloc[0] == 10.0

    def test_with_panel_exogenous_proper_naming(self):
        """Panel exogenous features with proper naming are merged by group suffix."""
        # Create panel y data with proper naming (no underscore before __)
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "target__store1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target__store2": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        # Create panel X with matching group structure
        X = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "feature__store1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature__store2": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        df = yohou_to_nixtla(y, X)

        # Panel exogenous should be matched by suffix and renamed to prefix
        assert "feature" in df.columns
        # 5 rows * 2 groups = 10 rows
        assert len(df) == 10
        # Each unique_id should only have their matching feature values
        store1_rows = df[df["unique_id"] == "target__store1"]
        assert store1_rows["feature"].iloc[0] == 10.0  # First feature__store1 value

    def test_with_panel_exogenous_no_matching_suffix(self):
        """Panel exogenous with no matching suffixes should not add columns."""
        # Create panel y data
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "target__store1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target__store2": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        # Create panel X with NON-matching group structure
        X = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "feature__region1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature__region2": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        df = yohou_to_nixtla(y, X)

        # No matching suffix means no feature column added
        assert "feature" not in df.columns
        assert len(df) == 10

    def test_with_panel_exogenous_global_y(self):
        """Panel exogenous with global y should skip feature matching."""
        # Create global y data (no __ separator)
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # Create panel X - won't match global y unique_ids
        X = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "feature__store1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature__store2": [11.0, 21.0, 31.0, 41.0, 51.0],
        })
        df = yohou_to_nixtla(y, X)

        # Global y unique_id "value" has no __ so no panel features matched
        assert "feature" not in df.columns
        assert len(df) == 5

    def test_with_exogenous_empty_columns(self, daily_y_X_factory):
        """Exogenous with only time column returns unchanged long_df."""
        y, _ = daily_y_X_factory(length=10, n_targets=1)
        X = pl.DataFrame({"time": y.select("time").to_series()})
        df = yohou_to_nixtla(y, X)

        # Should not have additional columns beyond the basic ones
        assert len([c for c in df.columns if c.startswith("X_")]) == 0

    def test_empty_value_columns_raises(self):
        """A DataFrame with only a ``time`` column raises ValueError."""
        y = pl.DataFrame({"time": [datetime(2020, 1, 1)]})
        with pytest.raises(ValueError, match="at least one value column"):
            yohou_to_nixtla(y)

    def test_ds_column_type(self, daily_y_X_factory):
        """The ``ds`` column should contain datetime objects."""
        y, _ = daily_y_X_factory(length=5, n_targets=1)
        df = yohou_to_nixtla(y)
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])


class TestNixtlaToYohou:
    """Tests for ``nixtla_to_yohou`` long-to-wide conversion."""

    def test_single_series(self):
        """Single unique_id reconstructs correctly."""
        forecast = pd.DataFrame({
            "unique_id": ["value"] * 3,
            "ds": pd.date_range("2020-01-06", periods=3, freq="D"),
            "model": [25.0, 30.0, 35.0],
        })
        result = nixtla_to_yohou(forecast, y_columns=["value"])

        assert "time" in result.columns
        assert "value" in result.columns
        assert len(result) == 3

    def test_panel_reconstruction(self):
        """Panel unique_ids are mapped back to prefixed columns."""
        forecast = pd.DataFrame({
            "unique_id": ["group_0__y_0"] * 2 + ["group_1__y_0"] * 2,
            "ds": list(pd.date_range("2020-01-06", periods=2, freq="D")) * 2,
            "model": [10.0, 20.0, 30.0, 40.0],
        })
        result = nixtla_to_yohou(forecast, y_columns=["group_0__y_0", "group_1__y_0"])

        assert "group_0__y_0" in result.columns
        assert "group_1__y_0" in result.columns
        assert len(result) == 2

    def test_observed_time_column(self):
        """When ``observed_time`` is provided, it is included in the output."""
        forecast = pd.DataFrame({
            "unique_id": ["value"] * 2,
            "ds": pd.date_range("2020-01-06", periods=2, freq="D"),
            "model": [25.0, 30.0],
        })
        obs_time = pl.Series("time", [datetime(2020, 1, 5)])
        result = nixtla_to_yohou(forecast, y_columns=["value"], observed_time=obs_time)

        assert "observed_time" in result.columns
        assert result.columns[0] == "observed_time"

    def test_empty_forecast_raises(self):
        """Empty forecast DataFrame raises ValueError."""
        forecast = pd.DataFrame(columns=["unique_id", "ds", "model"])
        with pytest.raises(ValueError, match="empty"):
            nixtla_to_yohou(forecast, y_columns=["value"])

    def test_no_prediction_columns_raises(self):
        """Forecast with only unique_id and ds raises ValueError."""
        forecast = pd.DataFrame({
            "unique_id": ["value"],
            "ds": [datetime(2020, 1, 1)],
        })
        with pytest.raises(ValueError, match="no prediction columns"):
            nixtla_to_yohou(forecast, y_columns=["value"])

    def test_column_order_preserved(self):
        """Output columns follow the y_columns order."""
        forecast = pd.DataFrame({
            "unique_id": ["b"] * 2 + ["a"] * 2,
            "ds": list(pd.date_range("2020-01-01", periods=2, freq="D")) * 2,
            "model": [1.0, 2.0, 3.0, 4.0],
        })
        result = nixtla_to_yohou(forecast, y_columns=["a", "b"])

        # Columns should be: time, a, b (order from y_columns)
        value_cols = [c for c in result.columns if c != "time"]
        assert value_cols == ["a", "b"]

    def test_y_column_not_in_pivot_is_skipped(self):
        """y_columns not matching any unique_id are silently skipped."""
        forecast = pd.DataFrame({
            "unique_id": ["a"] * 2,
            "ds": pd.date_range("2020-01-01", periods=2, freq="D"),
            "model": [1.0, 2.0],
        })
        # Request both "a" (exists) and "nonexistent" (doesn't exist)
        with pytest.warns(UserWarning, match="nonexistent"):
            result = nixtla_to_yohou(forecast, y_columns=["a", "nonexistent"])

        # Only "a" should be in result, "nonexistent" is skipped
        assert list(result.columns) == ["time", "a"]


class TestRoundTrip:
    """Tests for round-trip conversion fidelity."""

    def test_roundtrip_univariate(self, daily_y_X_factory):
        """Univariate round-trip preserves values."""
        y, _ = daily_y_X_factory(length=10, n_targets=1, seed=123)
        long_df = yohou_to_nixtla(y)

        # Simulate a prediction by renaming 'y' to 'model'
        pred_df = long_df.rename(columns={"y": "model"})
        result = nixtla_to_yohou(pred_df, y_columns=["y_0"])

        assert result.shape == (10, 2)  # time + y_0
        # Values should match
        original_values = y.select("y_0").to_series().to_list()
        roundtrip_values = result.select("y_0").to_series().to_list()
        for orig, rt in zip(original_values, roundtrip_values, strict=True):
            assert abs(orig - rt) < 1e-10

    def test_roundtrip_panel(self, daily_y_X_factory):
        """Panel data round-trip preserves group structure and values."""
        y, _ = daily_y_X_factory(length=10, n_targets=1, panel=True, n_groups=2, seed=456)
        long_df = yohou_to_nixtla(y)
        pred_df = long_df.rename(columns={"y": "model"})

        y_columns = [c for c in y.columns if c != "time"]
        result = nixtla_to_yohou(pred_df, y_columns=y_columns)

        assert set(result.columns) == {"time"} | set(y_columns)
        assert len(result) == 10


class TestEdgeCases:
    """Edge case tests for conversion functions."""

    def test_infer_freq_weekly_raises(self):
        """Weekly frequency (7d interval) is not directly mapped."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 6),
                end=datetime(2020, 1, 6) + timedelta(weeks=9),
                interval="1w",
                eager=True,
            ),
            "value": list(range(10)),
        })
        with pytest.raises(ValueError, match="Cannot map polars interval"):
            infer_freq(y)

    def test_infer_freq_minutely(self):
        """Minute-level frequency is inferred as 'min'."""
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1) + timedelta(minutes=9),
                interval="1m",
                eager=True,
            ),
            "value": list(range(10)),
        })
        assert infer_freq(y) == "min"

    def test_nixtla_to_yohou_warns_on_missing_y_column(self):
        """nixtla_to_yohou emits a warning when y_columns are not found."""
        forecast = pd.DataFrame({
            "unique_id": ["a"] * 2,
            "ds": pd.date_range("2020-01-01", periods=2, freq="D"),
            "model": [1.0, 2.0],
        })
        with pytest.warns(UserWarning, match="not found in the forecast output"):
            nixtla_to_yohou(forecast, y_columns=["a", "missing_col"])

    def test_yohou_to_nixtla_single_target_column(self):
        """Single target column should produce correct unique_id."""
        y = pl.DataFrame({
            "time": pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True),
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        df = yohou_to_nixtla(y)
        assert list(df.columns) == ["unique_id", "ds", "y"]
        assert df["unique_id"].unique().tolist() == ["sales"]
        assert len(df) == 5

    def test_roundtrip_multivariate_values_preserved(self, daily_y_X_factory):
        """Multivariate round-trip preserves per-column values."""
        y, _ = daily_y_X_factory(length=10, n_targets=3, seed=789)
        long_df = yohou_to_nixtla(y)
        pred_df = long_df.rename(columns={"y": "model"})

        y_columns = [c for c in y.columns if c != "time"]
        result = nixtla_to_yohou(pred_df, y_columns=y_columns)

        assert set(result.columns) == {"time"} | set(y_columns)
        for col in y_columns:
            original = y.select(col).to_series().to_list()
            roundtrip = result.select(col).to_series().to_list()
            for orig, rt in zip(original, roundtrip, strict=True):
                assert abs(orig - rt) < 1e-10
