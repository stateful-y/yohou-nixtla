"""Tests for neuralforecast model wrappers.

Covers ``BaseNeuralForecaster`` and concrete wrapper classes:
- construction and parameter handling
- get_params / set_params / clone
- fit / predict lifecycle (marked slow  --  trains PyTorch models)
"""

from __future__ import annotations

import tempfile

import polars as pl
import pytest
from sklearn.base import clone

from yohou_nixtla.neural import (
    BaseNeuralForecaster,
    MLPForecaster,
    NBEATSForecaster,
    NHITSForecaster,
    PatchTSTForecaster,
    TimesNetForecaster,
)

FAST_NEURAL_FORECASTERS = [
    MLPForecaster,
]

ALL_NEURAL_FORECASTERS = [
    NBEATSForecaster,
    NHITSForecaster,
    MLPForecaster,
    PatchTSTForecaster,
    TimesNetForecaster,
]


@pytest.fixture(params=FAST_NEURAL_FORECASTERS, ids=lambda cls: cls.__name__)
def fast_neural_cls(request):
    """Parameterized fast neural forecaster class."""
    return request.param


@pytest.fixture(params=ALL_NEURAL_FORECASTERS, ids=lambda cls: cls.__name__)
def neural_cls(request):
    """Parameterized neural forecaster class (all variants)."""
    return request.param


class TestBaseNeuralForecaster:
    """Tests for the BaseNeuralForecaster base class."""

    def test_is_abstract(self):
        """BaseNeuralForecaster without _estimator_default_class should raise."""
        with pytest.raises(TypeError):
            BaseNeuralForecaster()

    def test_has_estimator_name(self):
        """Verify _estimator_name is 'model'."""
        assert BaseNeuralForecaster._estimator_name == "model"


class TestConstructor:
    """Tests for neural forecaster construction and parameters."""

    def test_default_construction(self, neural_cls):
        """Default construction should succeed."""
        forecaster = neural_cls()
        assert forecaster is not None

    def test_input_size_parameter(self, fast_neural_cls):
        """input_size parameter should be stored."""
        forecaster = fast_neural_cls(input_size=48)
        assert forecaster.input_size == 48

    def test_max_steps_parameter(self, fast_neural_cls):
        """max_steps parameter should be stored."""
        forecaster = fast_neural_cls(max_steps=200)
        assert forecaster.max_steps == 200

    def test_freq_parameter(self, fast_neural_cls):
        """freq parameter should be stored."""
        forecaster = fast_neural_cls(freq="D")
        assert forecaster.freq == "D"

    def test_defaults(self, fast_neural_cls):
        """Default values should be correct."""
        forecaster = fast_neural_cls()
        assert forecaster.input_size == 24
        assert forecaster.max_steps == 100
        assert forecaster.freq is None


class TestGetSetParams:
    """Tests for sklearn get_params / set_params / clone compatibility."""

    def test_get_params_includes_neural_params(self, fast_neural_cls):
        """get_params should include input_size, max_steps, freq."""
        forecaster = fast_neural_cls(input_size=48, max_steps=200, freq="D")
        params = forecaster.get_params()
        assert params["input_size"] == 48
        assert params["max_steps"] == 200
        assert params["freq"] == "D"
        assert "model" in params

    def test_set_params(self, fast_neural_cls):
        """set_params should update neural-specific parameters."""
        forecaster = fast_neural_cls()
        forecaster.set_params(input_size=12, max_steps=50)
        assert forecaster.input_size == 12
        assert forecaster.max_steps == 50

    def test_clone(self, fast_neural_cls):
        """sklearn clone should produce an unfitted copy."""
        forecaster = fast_neural_cls(input_size=48, freq="D")
        cloned = clone(forecaster)
        assert cloned.input_size == 48
        assert cloned.freq == "D"
        assert cloned is not forecaster

    def test_repr(self, fast_neural_cls):
        """repr should be a valid string."""
        forecaster = fast_neural_cls()
        r = repr(forecaster)
        assert fast_neural_cls.__name__ in r

    def test_get_params_set_params_roundtrip(self, fast_neural_cls):
        """get_params should preserve key neural parameters after clone."""
        forecaster = fast_neural_cls(input_size=48, max_steps=200, freq="D")
        cloned = clone(forecaster)
        assert cloned.input_size == 48
        assert cloned.max_steps == 200
        assert cloned.freq == "D"
        assert type(cloned) is type(forecaster)

    def test_clone_after_fit_produces_unfitted_copy(self, fast_neural_cls, daily_y_X_factory):
        """clone() after fit should produce unfitted copy with same params."""
        from sklearn.exceptions import NotFittedError

        forecaster = fast_neural_cls(input_size=12, max_steps=5)
        y, _ = daily_y_X_factory(length=60)
        forecaster.fit(y, forecasting_horizon=3)

        cloned = clone(forecaster)

        # Neuralforecast loss objects (e.g. MAE()) are PyTorch modules whose
        # __eq__ falls back to identity, so two separately-created MAE() instances
        # compare as not-equal even when they are semantically the same.  Normalise
        # non-primitive param values to their type for comparison purposes.
        def _normalise(v):
            return v if v is None or isinstance(v, int | float | str | bool) else type(v)

        cloned_norm = {k: _normalise(v) for k, v in cloned.get_params().items()}
        orig_norm = {k: _normalise(v) for k, v in forecaster.get_params().items()}
        assert cloned_norm == orig_norm
        with pytest.raises(NotFittedError):
            cloned.predict()

    test_clone_after_fit_produces_unfitted_copy = pytest.mark.slow(test_clone_after_fit_produces_unfitted_copy)


class TestFitPredict:
    """Tests for the fit -> predict lifecycle (slow  --  trains PyTorch models)."""

    @pytest.mark.slow
    def test_fit_returns_self(self, daily_y_X_factory):
        """fit should return self."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        result = forecaster.fit(y, forecasting_horizon=3)
        assert result is forecaster

    @pytest.mark.slow
    def test_fit_sets_fitted_attributes(self, daily_y_X_factory):
        """Fitted forecaster should have expected attributes."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)

        assert hasattr(forecaster, "nixtla_forecaster_")
        assert hasattr(forecaster, "freq_")
        assert hasattr(forecaster, "y_columns_")
        assert forecaster.fit_forecasting_horizon_ == 3

    @pytest.mark.slow
    def test_predict_returns_correct_shape(self, daily_y_X_factory):
        """predict() should return h rows."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert "observed_time" in y_pred.columns
        assert len(y_pred) == 3

    @pytest.mark.slow
    def test_predict_columns_match_y(self, daily_y_X_factory):
        """Predictions should have the same target columns as training y."""
        y, _ = daily_y_X_factory(length=60, n_targets=2)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        expected_cols = {"time", "observed_time", "y_0", "y_1"}
        assert set(y_pred.columns) == expected_cols

    @pytest.mark.slow
    def test_predict_time_continuity(self, daily_y_X_factory):
        """Prediction times should follow training times."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        last_train_time = y["time"][-1]
        first_pred_time = y_pred["time"][0]

        from datetime import timedelta

        assert first_pred_time == last_train_time + timedelta(days=1)

    @pytest.mark.slow
    def test_predict_truncation(self, daily_y_X_factory):
        """predict(forecasting_horizon=k) with k < h should truncate."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    @pytest.mark.slow
    def test_freq_auto_inferred(self, daily_y_X_factory):
        """Freq should be auto-inferred as 'D' for daily data."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)
        assert forecaster.freq_ == "D"

    @pytest.mark.slow
    def test_predict_one_internal(self, daily_y_X_factory):
        """Internal _predict_one method should return valid predictions."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y, forecasting_horizon=3)

        # Directly call the internal _predict_one method
        y_pred = forecaster._predict_one()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 3

    @pytest.mark.slow
    def test_predict_before_fit_raises(self):
        """predict before fit should raise."""
        from sklearn.exceptions import NotFittedError

        forecaster = MLPForecaster()
        with pytest.raises(NotFittedError):
            forecaster.predict()

    @pytest.mark.slow
    def test_observe_updates_state(self, daily_y_X_factory):
        """observe() should accept new data without raising."""
        y, _ = daily_y_X_factory(length=80)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y[:60], forecasting_horizon=3)

        result = forecaster.observe(y[60:70])
        assert result is forecaster

    @pytest.mark.slow
    def test_observe_predict_returns_predictions(self, daily_y_X_factory):
        """observe_predict() should return predictions after observing."""
        y, _ = daily_y_X_factory(length=80)
        forecaster = MLPForecaster(input_size=12, max_steps=5)
        forecaster.fit(y[:60], forecasting_horizon=3)

        y_pred = forecaster.observe_predict(y[60:70])
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        # observe_predict does an initial predict then rolls with stride=h over
        # the provided window.  With y[60:70] (10 rows) and h=3:
        #   initial(3) + range(0, 10, 3) → [0,3,6,9] = 4 iterations × 3 = 15.
        h = forecaster.fit_forecasting_horizon_
        stride = h
        n_obs = len(y[60:70])
        expected_len = h * (1 + len(range(0, n_obs, stride)))
        assert len(y_pred) == expected_len

    @pytest.mark.slow
    def test_nbeats(self, daily_y_X_factory):
        """NBEATS should fit and predict."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = NBEATSForecaster(
            input_size=12,
            max_steps=5,
            mlp_units=[[8, 8]],
            n_blocks=[1],
            stack_types=["identity"],
        )
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict()
        assert len(y_pred) == 3

    @pytest.mark.slow
    def test_nhits(self, daily_y_X_factory):
        """NHITS should fit and predict."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = NHITSForecaster(
            input_size=12,
            max_steps=5,
            mlp_units=[[8, 8]],
            n_blocks=[1],
            stack_types=["identity"],
            n_pool_kernel_size=[1],
            n_freq_downsample=[1],
        )
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict()
        assert len(y_pred) == 3


def _make_neural_forecaster(cls, **extra_kwargs):
    """Create a neural forecaster with sensible defaults for fast testing."""
    defaults = {
        "input_size": 12,
        "max_steps": 5,
        # Use a unique temp directory so parallel xdist workers don't race on
        # the shared ``lightning_logs/version_N`` directory.
        # neuralforecast forwards unknown kwargs straight to pl.Trainer, so
        # ``default_root_dir`` is passed at the top level, not nested.
        "default_root_dir": tempfile.mkdtemp(),
    }
    # Per-model architecture reduction for faster tests.
    _model_defaults = {
        "MLPForecaster": {"hidden_size": 8, "num_layers": 1},
        "NBEATSForecaster": {
            "mlp_units": [[8, 8]],
            "n_blocks": [1],
            "stack_types": ["identity"],
        },
        "NHITSForecaster": {
            "mlp_units": [[8, 8]],
            "n_blocks": [1],
            "stack_types": ["identity"],
            "n_pool_kernel_size": [1],
            "n_freq_downsample": [1],
        },
        "PatchTSTForecaster": {"hidden_size": 8, "encoder_layers": 1, "n_heads": 4},
        "TimesNetForecaster": {"hidden_size": 8, "conv_hidden_size": 8, "encoder_layers": 1},
    }
    defaults.update(_model_defaults.get(cls.__name__, {}))
    kwargs = {**defaults, **extra_kwargs}
    return cls(**kwargs)


class TestMultivariate:
    """Tests for multivariate (multiple target columns) support."""

    @pytest.mark.slow
    def test_multivariate_fit_predict(self, daily_y_X_factory):
        """Should handle multiple target columns."""
        y, _ = daily_y_X_factory(length=60, n_targets=3)
        forecaster = _make_neural_forecaster(MLPForecaster)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        # 3 targets + time + observed_time
        assert len(y_pred.columns) == 5


class TestPanelData:
    """Tests for panel (grouped) time series data."""

    @pytest.mark.slow
    def test_panel_fit_predict(self, daily_y_X_factory):
        """Should handle panel data with __ separator."""
        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)
        forecaster = _make_neural_forecaster(MLPForecaster)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 3

    @pytest.mark.slow
    def test_panel_columns_preserved(self, daily_y_X_factory):
        """Panel column names should be preserved in predictions."""
        y, _ = daily_y_X_factory(length=60, panel=True, n_groups=2)
        forecaster = _make_neural_forecaster(MLPForecaster)
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        y_cols = [c for c in y.columns if c != "time"]
        pred_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        assert set(pred_cols) == set(y_cols)


class TestTransformerIntegration:
    """Tests for feature_transformer / target_transformer / target_as_feature."""

    def test_get_params_includes_transformer_params(self, fast_neural_cls):
        """get_params should include transformer-related params."""
        forecaster = fast_neural_cls()
        params = forecaster.get_params()
        assert "feature_transformer" in params
        assert "target_transformer" in params
        assert "target_as_feature" in params
        assert params["feature_transformer"] is None
        assert params["target_transformer"] is None
        assert params["target_as_feature"] is None

    def test_set_params_transformer(self, fast_neural_cls):
        """set_params should update transformer params."""
        from yohou.preprocessing import FunctionTransformer

        ft = FunctionTransformer()
        forecaster = fast_neural_cls()
        forecaster.set_params(feature_transformer=ft)
        assert forecaster.feature_transformer is ft

    def test_clone_preserves_transformers(self, fast_neural_cls):
        """clone should preserve transformer params."""
        from yohou.preprocessing import FunctionTransformer

        ft = FunctionTransformer()
        forecaster = fast_neural_cls(feature_transformer=ft)
        cloned = clone(forecaster)
        assert cloned.feature_transformer is not None
        assert type(cloned.feature_transformer) is type(ft)

    @pytest.mark.slow
    def test_fit_predict_with_target_transformer(self, daily_y_X_factory):
        """Predictions with target_transformer should be inverse-transformed."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_neural_forecaster(MLPForecaster, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict()

        assert len(y_pred) == 3
        assert "time" in y_pred.columns
        assert "observed_time" in y_pred.columns

    @pytest.mark.slow
    def test_predict_transformed_flag(self, daily_y_X_factory):
        """predict_transformed=True should skip inverse transform."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_neural_forecaster(MLPForecaster, target_transformer=ft)
        forecaster.fit(y, forecasting_horizon=3)

        y_inv = forecaster.predict(predict_transformed=False)
        y_raw = forecaster.predict(predict_transformed=True)

        assert len(y_inv) == 3
        assert len(y_raw) == 3

    def test_sklearn_tags_reflect_transformers(self, fast_neural_cls):
        """Tags should reflect whether transformers are set."""
        from yohou.preprocessing import FunctionTransformer

        forecaster = fast_neural_cls()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.uses_target_transformer is False
        assert tags.forecaster_tags.uses_feature_transformer is False

        ft = FunctionTransformer()
        forecaster_with = fast_neural_cls(target_transformer=ft)
        tags_with = forecaster_with.__sklearn_tags__()
        assert tags_with.forecaster_tags.uses_target_transformer is True

    @pytest.mark.slow
    def test_target_transformer_inverse_values_correct(self, daily_y_X_factory):
        """Inverse transform should halve the transformed-space prediction values."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)

        ft = FunctionTransformer(
            func=lambda X: X.with_columns([pl.col(c) * 2 for c in X.columns if c != "time"]),
            inverse_func=lambda X: X.with_columns([pl.col(c) / 2 for c in X.columns if c != "time"]),
        )

        forecaster = _make_neural_forecaster(MLPForecaster, target_transformer=ft)
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

    @pytest.mark.slow
    def test_target_as_feature_transformed_fit_predict(self, daily_y_X_factory):
        """target_as_feature='transformed' should fit and predict without X."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_neural_forecaster(MLPForecaster, target_as_feature="transformed")
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        assert "time" in y_pred.columns

    @pytest.mark.slow
    def test_target_as_feature_raw_fit_predict(self, daily_y_X_factory):
        """target_as_feature='raw' should fit and predict without X."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_neural_forecaster(MLPForecaster, target_as_feature="raw")
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict()
        assert len(y_pred) == 3
        assert "time" in y_pred.columns

    def test_feature_transformer_no_X_raises(self, fast_neural_cls, daily_y_X_factory):
        """feature_transformer with target_as_feature=None and no X should raise."""
        from yohou.preprocessing import FunctionTransformer

        y, _ = daily_y_X_factory(length=60)
        ft = FunctionTransformer()
        forecaster = fast_neural_cls(feature_transformer=ft)
        with pytest.raises(ValueError, match="target_as_feature=None requires X"):
            forecaster.fit(y, forecasting_horizon=3)

    @pytest.mark.slow
    def test_target_as_feature_with_feature_transformer(self, daily_y_X_factory):
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

        forecaster = _make_neural_forecaster(
            MLPForecaster,
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
    """Tests for neural forecaster exogenous tags."""

    EXOGENOUS_CAPABLE = {PatchTSTForecaster, TimesNetForecaster}

    def test_neural_ignores_exogenous(self, neural_cls):
        """Non-exogenous neural forecasters should ignore exogenous."""
        if neural_cls in self.EXOGENOUS_CAPABLE:
            pytest.skip(f"{neural_cls.__name__} supports exogenous")
        forecaster = neural_cls()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is True

    def test_patchtst_supports_exogenous(self):
        """PatchTST should support exogenous features."""
        forecaster = PatchTSTForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is False

    def test_timesnet_supports_exogenous(self):
        """TimesNet should support exogenous features."""
        forecaster = TimesNetForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.ignores_exogenous is False


class TestFitValidation:
    """Tests for fit-time input validation."""

    def test_invalid_forecasting_horizon(self, fast_neural_cls, daily_y_X_factory):
        """forecasting_horizon < 1 should raise ValueError."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = fast_neural_cls()
        with pytest.raises(ValueError, match="forecasting_horizon must be a positive integer"):
            forecaster.fit(y, forecasting_horizon=0)

    def test_convert_nixtla_to_yohou_legacy(self, daily_y_X_factory):
        """Legacy _convert_nixtla_to_yohou should produce valid output."""
        y, _ = daily_y_X_factory(length=60)
        forecaster = _make_neural_forecaster(MLPForecaster)
        forecaster.fit(y, forecasting_horizon=3)

        forecast_df = forecaster.nixtla_forecaster_.predict()
        y_pred = forecaster._convert_nixtla_to_yohou(forecast_df)
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    test_convert_nixtla_to_yohou_legacy = pytest.mark.slow(test_convert_nixtla_to_yohou_legacy)


class TestSystematicChecks:
    """Run yohou's systematic forecaster checks on neural wrappers."""

    EXPECTED_FAILURES = frozenset()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster_cls",
        FAST_NEURAL_FORECASTERS,
        ids=lambda cls: cls.__name__,
    )
    def test_yohou_forecaster_checks(self, forecaster_cls, daily_y_X_factory):
        """Systematic forecaster checks should pass for neural forecasters."""
        from yohou.testing import _yield_yohou_forecaster_checks

        from conftest import run_checks

        y, _ = daily_y_X_factory(length=200)
        y_train, y_test = y[:160], y[160:]

        forecaster = _make_neural_forecaster(forecaster_cls)
        forecaster.fit(y_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures=self.EXPECTED_FAILURES,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster_cls",
        FAST_NEURAL_FORECASTERS,
        ids=lambda cls: cls.__name__,
    )
    def test_yohou_forecaster_checks_panel(self, forecaster_cls, daily_y_X_factory):
        """Systematic forecaster checks should pass for panel data."""
        from yohou.testing import _yield_yohou_forecaster_checks

        from conftest import run_checks

        y, _ = daily_y_X_factory(length=200, panel=True, n_groups=2)
        y_train, y_test = y[:160], y[160:]

        forecaster = _make_neural_forecaster(forecaster_cls)
        forecaster.fit(y_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures=self.EXPECTED_FAILURES,
        )
