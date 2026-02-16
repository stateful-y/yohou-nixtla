"""Tests for statsforecast model wrappers.

Covers ``BaseStatsForecaster`` and all concrete wrapper classes:
- fit / predict lifecycle
- get_params / set_params / clone
- panel data support
- horizon changes at predict time
"""

from __future__ import annotations

import polars as pl
import pytest
from sklearn.base import clone

from yohou_nixtla.stats import (
    ARIMAForecaster,
    AutoARIMAForecaster,
    AutoCESForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    BaseStatsForecaster,
    CrostonForecaster,
    HoltWintersForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)

FAST_FORECASTERS = [
    NaiveForecaster,
    SeasonalNaiveForecaster,
    CrostonForecaster,
]

ALL_FORECASTERS = [
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoCESForecaster,
    AutoThetaForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ARIMAForecaster,
    HoltWintersForecaster,
    ThetaForecaster,
    CrostonForecaster,
]


def _make_forecaster(cls, **extra_kwargs):
    """Create a forecaster with sensible defaults for required params."""
    defaults = {
        SeasonalNaiveForecaster: {"season_length": 7},
    }
    kwargs = {**defaults.get(cls, {}), **extra_kwargs}
    return cls(**kwargs)


@pytest.fixture(params=FAST_FORECASTERS, ids=lambda cls: cls.__name__)
def fast_forecaster_cls(request):
    """Parameterized fast forecaster class (for expensive tests)."""
    return request.param


@pytest.fixture(params=ALL_FORECASTERS, ids=lambda cls: cls.__name__)
def forecaster_cls(request):
    """Parameterized forecaster class (all concrete wrappers)."""
    return request.param


class TestBaseStatsForecaster:
    """Tests for the BaseStatsForecaster base class."""

    def test_is_abstract(self):
        """BaseStatsForecaster without _estimator_default_class should still instantiate."""
        # BaseStatsForecaster has _estimator_default_class = None
        # When no model is passed and no default, BaseClassWrapper raises
        with pytest.raises(TypeError):
            BaseStatsForecaster()

    def test_has_estimator_name(self):
        """Verify _estimator_name is 'model'."""
        assert BaseStatsForecaster._estimator_name == "model"

    def test_sklearn_tags(self, fast_forecaster_cls):
        """__sklearn_tags__ should return proper forecaster tags."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == "point"
        assert tags.forecaster_tags.uses_reduction is False

    def test_sklearn_tags_without_forecaster_tags(self, fast_forecaster_cls):
        """__sklearn_tags__ raises when forecaster_tags is None."""
        from unittest.mock import MagicMock, patch

        from yohou.point import BasePointForecaster

        forecaster = _make_forecaster(fast_forecaster_cls)

        # Create mock tags with forecaster_tags=None
        mock_tags = MagicMock()
        mock_tags.forecaster_tags = None

        with (
            patch.object(
                BasePointForecaster,
                "__sklearn_tags__",
                return_value=mock_tags,
            ),
            pytest.raises(AssertionError),
        ):
            forecaster.__sklearn_tags__()


class TestConstructor:
    """Tests for forecaster construction and parameter handling."""

    def test_default_construction(self, forecaster_cls):
        """Default construction should succeed for all concrete classes."""
        forecaster = _make_forecaster(forecaster_cls)
        assert forecaster is not None

    def test_freq_parameter(self, fast_forecaster_cls):
        """Freq parameter should be stored."""
        forecaster = _make_forecaster(fast_forecaster_cls, freq="D")
        assert forecaster.freq == "D"

    def test_default_freq_is_none(self, fast_forecaster_cls):
        """Default freq should be None (auto-inferred)."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        assert forecaster.freq is None


class TestGetSetParams:
    """Tests for sklearn get_params / set_params / clone compatibility."""

    def test_get_params(self, fast_forecaster_cls):
        """get_params should return freq and model class."""
        forecaster = _make_forecaster(fast_forecaster_cls, freq="D")
        params = forecaster.get_params()
        assert params["freq"] == "D"
        assert "model" in params

    def test_set_params(self, fast_forecaster_cls):
        """set_params should update parameters."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.set_params(freq="h")
        assert forecaster.freq == "h"

    def test_set_params_without_freq(self, fast_forecaster_cls):
        """set_params without freq should still work."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        original_freq = forecaster.freq
        # Call set_params with empty dict (no changes)
        forecaster.set_params()
        assert forecaster.freq == original_freq

    def test_clone(self, fast_forecaster_cls):
        """sklearn clone should produce an unfitted copy."""
        forecaster = _make_forecaster(fast_forecaster_cls, freq="D")
        cloned = clone(forecaster)
        assert cloned.freq == "D"
        assert cloned is not forecaster

    def test_repr(self, fast_forecaster_cls):
        """repr should be a valid string."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        r = repr(forecaster)
        assert fast_forecaster_cls.__name__ in r


class TestFitPredict:
    """Tests for the fit → predict lifecycle."""

    def test_fit_returns_self(self, fast_forecaster_cls, daily_y_X_factory):
        """fit should return self."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        result = forecaster.fit(y, forecasting_horizon=5)
        assert result is forecaster

    def test_fit_sets_fitted_attributes(self, fast_forecaster_cls, daily_y_X_factory):
        """Fitted forecaster should have expected attributes."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        assert hasattr(forecaster, "nixtla_forecaster_")
        assert hasattr(forecaster, "freq_")
        assert hasattr(forecaster, "y_columns_")
        assert forecaster.fit_forecasting_horizon_ == 5

    def test_predict_default_horizon(self, fast_forecaster_cls, daily_y_X_factory):
        """predict() with no args should use fit horizon."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        y_pred = forecaster.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert "observed_time" in y_pred.columns
        assert len(y_pred) == 5

    def test_predict_custom_horizon(self, fast_forecaster_cls, daily_y_X_factory):
        """predict() with different horizon should work."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_predict_columns_match_y(self, fast_forecaster_cls, daily_y_X_factory):
        """Predictions should have the same target columns as training y."""
        y, _ = daily_y_X_factory(length=60, n_targets=2)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        y_pred = forecaster.predict()
        expected_cols = {"time", "observed_time", "y_0", "y_1"}
        assert set(y_pred.columns) == expected_cols

    def test_predict_time_continuity(self, fast_forecaster_cls, daily_y_X_factory):
        """Prediction times should directly follow training times."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        last_train_time = y["time"][-1]
        first_pred_time = y_pred["time"][0]

        # First prediction time should be 1 day after last training time
        from datetime import timedelta

        assert first_pred_time == last_train_time + timedelta(days=1)

    def test_predict_before_fit_raises(self, fast_forecaster_cls):
        """predict before fit should raise."""
        from sklearn.exceptions import NotFittedError

        forecaster = _make_forecaster(fast_forecaster_cls)
        with pytest.raises(NotFittedError):
            forecaster.predict()

    def test_freq_auto_inferred(self, fast_forecaster_cls, daily_y_X_factory):
        """Freq should be auto-inferred as 'D' for daily data."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)
        assert forecaster.freq_ == "D"

    def test_freq_manual_override(self, fast_forecaster_cls, daily_y_X_factory):
        """Manual freq should override auto-inference."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls, freq="D")
        forecaster.fit(y, forecasting_horizon=5)
        assert forecaster.freq_ == "D"

    def test_predict_one_internal(self, fast_forecaster_cls, daily_y_X_factory):
        """Internal _predict_one method should return valid predictions."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        # Directly call the internal _predict_one method
        y_pred = forecaster._predict_one()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 5


class TestLifecycleMethods:
    """Tests for observe, rewind, and observe_predict lifecycle methods."""

    def test_observe_updates_state(self, fast_forecaster_cls, daily_y_X_factory):
        """observe() should accept new data without raising."""
        y, _ = daily_y_X_factory(length=80)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y[:60], forecasting_horizon=5)

        result = forecaster.observe(y[60:70])
        assert result is forecaster

    def test_observe_predict_returns_predictions(self, fast_forecaster_cls, daily_y_X_factory):
        """observe_predict() should return predictions after observing."""
        y, _ = daily_y_X_factory(length=80)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y[:60], forecasting_horizon=5)

        y_pred = forecaster.observe_predict(y[60:70])
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 15

    def test_rewind_does_not_raise(self, fast_forecaster_cls, daily_y_X_factory):
        """rewind() after observe should run without error."""
        y, _ = daily_y_X_factory(length=80)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y[:60], forecasting_horizon=5)

        forecaster.observe(y[60:70])
        result = forecaster.rewind(y[60:70])
        assert result is forecaster


class TestCloneRoundtrip:
    """Tests for clone → fit → predict roundtrip fidelity."""

    def test_clone_before_fit_preserves_params(self, fast_forecaster_cls):
        """clone() before fit should preserve all constructor params."""
        forecaster = _make_forecaster(fast_forecaster_cls, freq="D")
        cloned = clone(forecaster)
        assert cloned.get_params() == forecaster.get_params()
        assert cloned is not forecaster

    def test_clone_after_fit_produces_unfitted_copy(self, fast_forecaster_cls, daily_y_X_factory):
        """clone() after fit should produce an unfitted copy with same params."""
        from sklearn.exceptions import NotFittedError

        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        cloned = clone(forecaster)
        assert cloned.get_params() == forecaster.get_params()
        with pytest.raises(NotFittedError):
            cloned.predict()

    def test_get_params_set_params_roundtrip(self, fast_forecaster_cls):
        """get_params of a clone should match original's get_params."""
        forecaster = _make_forecaster(fast_forecaster_cls, freq="h")
        params = forecaster.get_params()
        cloned = clone(forecaster)
        assert cloned.get_params() == params


class TestMultivariate:
    """Tests for multivariate (multiple target columns) support."""

    def test_multivariate_fit_predict(self, fast_forecaster_cls, daily_y_X_factory):
        """Should handle multiple target columns."""
        y, _ = daily_y_X_factory(length=60, n_targets=3)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        y_pred = forecaster.predict()
        assert len(y_pred) == 5
        # 3 targets + time + observed_time
        assert len(y_pred.columns) == 5


class TestPanelData:
    """Tests for panel (grouped) time series data."""

    def test_panel_fit_predict(self, fast_forecaster_cls, daily_y_X_factory):
        """Should handle panel data with __ separator."""
        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 3

    def test_panel_columns_preserved(self, fast_forecaster_cls, daily_y_X_factory):
        """Panel column names should be preserved in predictions."""
        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        y_cols = [c for c in y.columns if c != "time"]
        pred_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        assert set(pred_cols) == set(y_cols)

    def test_panel_group_mismatch_raises(self):
        """Mismatched panel group names between y and X raises ValueError."""
        from datetime import datetime, timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=59),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "target__store1": list(range(60)),
            "target__store2": list(range(60, 120)),
        })
        X = pl.DataFrame({
            "time": time,
            "feature__region1": list(range(60)),
            "feature__region2": list(range(60, 120)),
        })
        forecaster = NaiveForecaster()
        with pytest.raises(ValueError, match="do not have the same local group names"):
            forecaster.fit(y, X, forecasting_horizon=3)

    def test_panel_fit_with_matching_exogenous(self):
        """Panel X with matching group names is accepted without error."""
        from datetime import datetime, timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=59),
            interval="1d",
            eager=True,
        )
        # Use same prefix (group name) for both y and X
        y = pl.DataFrame({
            "time": time,
            "group_0__target": list(range(60)),
            "group_1__target": list(range(60, 120)),
        })
        X = pl.DataFrame({
            "time": time,
            "group_0__feature": list(range(60)),
            "group_1__feature": list(range(60, 120)),
        })
        forecaster = NaiveForecaster()
        forecaster.fit(y, X, forecasting_horizon=3)
        y_pred = forecaster.predict()
        assert len(y_pred) == 3


class TestPredictTruncation:
    """Test prediction truncation when backend returns extra steps."""

    def test_truncate_extra_steps(self, daily_y_X_factory):
        """Predictions are truncated when backend returns more steps than h."""
        from unittest.mock import patch

        import pandas as pd

        y, _ = daily_y_X_factory(length=60)
        forecaster = NaiveForecaster()
        forecaster.fit(y, forecasting_horizon=3)

        # Build a mock forecast_df with MORE rows than h=3
        extra_rows = 5
        mock_df = pd.DataFrame({
            "ds": pd.date_range("2020-03-01", periods=extra_rows, freq="D"),
            "unique_id": ["y_0"] * extra_rows,
            "model": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        with patch.object(forecaster, "_predict_backend", return_value=mock_df):
            y_pred = forecaster.predict(forecasting_horizon=3)

        # Should be truncated to h=3, not extra_rows=5
        assert len(y_pred) == 3


class TestAutoModels:
    """Integration tests for auto-selection models (slower)."""

    @pytest.mark.slow
    def test_auto_arima(self, daily_y_X_factory):
        """AutoARIMA should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = AutoARIMAForecaster()
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    def test_auto_ets(self, daily_y_X_factory):
        """AutoETS should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = AutoETSForecaster()
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Upstream numba typing error in statsforecast AutoCES",
        raises=Exception,
        strict=False,
    )
    def test_auto_ces(self, daily_y_X_factory):
        """AutoCES should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = AutoCESForecaster()
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    def test_auto_theta(self, daily_y_X_factory):
        """AutoTheta should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = AutoThetaForecaster()
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    def test_arima_manual(self, daily_y_X_factory):
        """ARIMA with manual orders should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = ARIMAForecaster(order=(1, 0, 0))
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    def test_holt_winters(self, daily_y_X_factory):
        """HoltWinters should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = HoltWintersForecaster(season_length=7)
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5

    @pytest.mark.slow
    def test_theta_manual(self, daily_y_X_factory):
        """Theta method should fit and predict."""
        y, _ = daily_y_X_factory(length=100)
        forecaster = ThetaForecaster()
        forecaster.fit(y, forecasting_horizon=5)
        y_pred = forecaster.predict()
        assert len(y_pred) == 5


class TestTransformerIntegration:
    """Tests for feature_transformer / target_transformer / target_as_feature."""

    def test_get_params_includes_transformer_params(self, fast_forecaster_cls):
        """get_params should include transformer-related params."""
        forecaster = _make_forecaster(fast_forecaster_cls)
        params = forecaster.get_params()
        assert "feature_transformer" in params
        assert "target_transformer" in params
        assert "target_as_feature" in params
        assert params["feature_transformer"] is None
        assert params["target_transformer"] is None
        assert params["target_as_feature"] is None

    def test_set_params_transformer(self, fast_forecaster_cls):
        """set_params should update transformer params."""
        from yohou.preprocessing import FunctionTransformer

        ft = FunctionTransformer()
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.set_params(feature_transformer=ft)
        assert forecaster.feature_transformer is ft

    def test_clone_preserves_transformers(self, fast_forecaster_cls):
        """clone should preserve transformer params."""
        from yohou.preprocessing import FunctionTransformer

        ft = FunctionTransformer()
        forecaster = _make_forecaster(fast_forecaster_cls, feature_transformer=ft)
        cloned = clone(forecaster)
        assert cloned.feature_transformer is not None
        assert type(cloned.feature_transformer) is type(ft)

    def test_fit_predict_with_target_transformer(self, fast_forecaster_cls, daily_y_X_factory):
        """Predictions with target_transformer should be inverse-transformed."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        assert "time" in y_pred.columns
        assert "observed_time" in y_pred.columns

    def test_predict_transformed_flag(self, fast_forecaster_cls, daily_y_X_factory):
        """predict_transformed=True should skip inverse transform."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)

        y_inv = forecaster.predict(predict_transformed=False)
        y_raw = forecaster.predict(predict_transformed=True)

        assert len(y_inv) == 3
        assert len(y_raw) == 3

    def test_sklearn_tags_reflect_transformers(self, fast_forecaster_cls):
        """Tags should reflect whether transformers are set."""
        from yohou.preprocessing import FunctionTransformer

        forecaster = _make_forecaster(fast_forecaster_cls)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.uses_target_transformer is False
        assert tags.forecaster_tags.uses_feature_transformer is False

        ft = FunctionTransformer()
        forecaster_with = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        tags_with = forecaster_with.__sklearn_tags__()
        assert tags_with.forecaster_tags.uses_target_transformer is True

    def test_target_transformer_panel(self, fast_forecaster_cls, daily_y_X_factory):
        """target_transformer should work with panel data (per-group inverse)."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        y_cols = [c for c in y.columns if c != "time"]
        pred_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        assert set(pred_cols) == set(y_cols)

    def test_target_transformer_inverse_values_correct(self, fast_forecaster_cls, daily_y_X_factory):
        """Inverse transform should halve the transformed-space prediction values."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)

        y_inv = forecaster.predict(predict_transformed=False)
        y_raw = forecaster.predict(predict_transformed=True)

        value_cols = [c for c in y_inv.columns if c not in ("time", "observed_time")]
        for col in value_cols:
            inv_vals = y_inv[col].to_list()
            raw_vals = y_raw[col].to_list()
            expected = [v / 2 for v in raw_vals]
            assert inv_vals == pytest.approx(expected, rel=1e-6), (
                f"Column {col}: inverse-transformed values should be half of transformed-space values"
            )

    def test_target_transformer_inverse_values_correct_panel(self, fast_forecaster_cls, daily_y_X_factory):
        """Panel inverse transform should halve values per group."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(fast_forecaster_cls, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)

        y_inv = forecaster.predict(predict_transformed=False)
        y_raw = forecaster.predict(predict_transformed=True)

        value_cols = [c for c in y_inv.columns if c not in ("time", "observed_time")]
        for col in value_cols:
            inv_vals = y_inv[col].to_list()
            raw_vals = y_raw[col].to_list()
            expected = [v / 2 for v in raw_vals]
            assert inv_vals == pytest.approx(expected, rel=1e-6), (
                f"Column {col}: panel inverse-transformed values mismatch"
            )

    def test_target_as_feature_transformed_fit_predict(self, fast_forecaster_cls, daily_y_X_factory):
        """target_as_feature='transformed' should fit and predict without X."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls, target_as_feature="transformed")
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        assert "time" in y_pred.columns

    def test_target_as_feature_raw_fit_predict(self, fast_forecaster_cls, daily_y_X_factory):
        """target_as_feature='raw' should fit and predict without X."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls, target_as_feature="raw")
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        assert "time" in y_pred.columns

    def test_feature_transformer_no_X_raises(self, fast_forecaster_cls, daily_y_X_factory):
        """feature_transformer with target_as_feature=None and no X should raise."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)
        ft = FunctionTransformer()
        forecaster = _make_forecaster(fast_forecaster_cls, feature_transformer=ft)
        with pytest.raises(ValueError, match="target_as_feature=None requires X"):
            forecaster.fit(y, forecasting_horizon=3)

    def test_target_as_feature_with_feature_transformer(self, fast_forecaster_cls, daily_y_X_factory):
        """target_as_feature='transformed' with feature_transformer should work end-to-end."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        target_ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )
        feature_ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) + 1 for c in X.columns if c != "time"]),
        )

        forecaster = _make_forecaster(
            fast_forecaster_cls,
            target_transformer=target_ft,
            feature_transformer=feature_ft,
            target_as_feature="transformed",
        )
        forecaster.fit(y, forecasting_horizon=3)

        assert forecaster.target_transformer_ is not None
        assert forecaster.feature_transformer_ is not None

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        assert "time" in y_pred.columns


class TestIgnoresExogenous:
    """Tests for per-model ignores_exogenous tags."""

    def test_naive_ignores_exogenous(self):
        """Naive models should ignore exogenous features."""
        forecaster = NaiveForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

    def test_seasonal_naive_ignores_exogenous(self):
        """Seasonal naive should ignore exogenous features."""
        forecaster = SeasonalNaiveForecaster(season_length=7)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

    def test_croston_ignores_exogenous(self):
        """Croston should ignore exogenous features."""
        forecaster = CrostonForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

    def test_arima_supports_exogenous(self):
        """ARIMA models should support exogenous features."""
        forecaster = ARIMAForecaster(order=(1, 0, 0))
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is False

    def test_auto_arima_supports_exogenous(self):
        """AutoARIMA should support exogenous features."""
        forecaster = AutoARIMAForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is False

    def test_holt_winters_ignores_exogenous(self):
        """HoltWinters should ignore exogenous features (default tag)."""
        forecaster = HoltWintersForecaster(season_length=7)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

    def test_theta_ignores_exogenous(self):
        """Theta should ignore exogenous features (default tag)."""
        forecaster = ThetaForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True


class TestFitValidation:
    """Tests for fit-time input validation."""

    def test_invalid_forecasting_horizon(self, fast_forecaster_cls, daily_y_X_factory):
        """forecasting_horizon < 1 should raise ValueError."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        with pytest.raises(ValueError, match="forecasting_horizon must be a positive integer"):
            forecaster.fit(y, forecasting_horizon=0)

    def test_convert_nixtla_to_yohou_no_reset(self, fast_forecaster_cls, daily_y_X_factory):
        """_convert_nixtla_to_yohou with reset_index=False should work."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_forecaster(fast_forecaster_cls)
        forecaster.fit(y, forecasting_horizon=5)

        forecast_df = forecaster.nixtla_forecaster_.predict(h=5)
        y_pred = forecaster._convert_nixtla_to_yohou(forecast_df, reset_index=False)
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns


class TestSystematicChecks:
    """Run yohou's systematic forecaster checks on each wrapper."""

    EXPECTED_FAILURES = frozenset()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster_cls",
        FAST_FORECASTERS,
        ids=lambda cls: cls.__name__,
    )
    def test_yohou_forecaster_checks(self, forecaster_cls, daily_y_X_factory):
        """Systematic forecaster checks should pass."""
        from yohou.testing import _yield_yohou_forecaster_checks

        from conftest import run_checks

        y, _ = daily_y_X_factory(length=200)
        y_train, y_test = y[:160], y[160:]

        forecaster = _make_forecaster(forecaster_cls)
        forecaster.fit(y_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures=self.EXPECTED_FAILURES,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster_cls",
        FAST_FORECASTERS,
        ids=lambda cls: cls.__name__,
    )
    def test_yohou_forecaster_checks_panel(self, forecaster_cls, daily_y_X_factory):
        """Systematic forecaster checks should pass for panel data."""
        from yohou.testing import _yield_yohou_forecaster_checks

        from conftest import run_checks

        y, _ = daily_y_X_factory(length=200, panel=True, n_groups=2)
        y_train, y_test = y[:160], y[160:]

        forecaster = _make_forecaster(forecaster_cls)
        forecaster.fit(y_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures=self.EXPECTED_FAILURES,
        )
