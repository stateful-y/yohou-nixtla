"""Base neuralforecast forecaster class for Yohou.

This module provides ``BaseNeuralForecaster``, a base class that integrates
neuralforecast (PyTorch-based) models into the yohou framework via
``BaseNixtlaForecaster``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou_nixtla._base import BaseNixtlaForecaster


class BaseNeuralForecaster(BaseNixtlaForecaster):
    """Base class for neuralforecast model wrappers in yohou.

    Wraps a neuralforecast model class using the ``BaseNixtlaForecaster``
    pattern, making it fully compatible with yohou's forecaster API.

    Neuralforecast models require the forecast horizon (``h``) at model
    construction time.  This class therefore does **not** use
    ``_fit_context`` -- instead, it calls ``instantiate()`` manually
    inside ``fit()`` after syncing ``h`` with ``forecasting_horizon``.

    Subclasses only need to set ``_estimator_default_class`` to a specific
    neuralforecast model class (e.g., ``NBEATS``).

    Parameters
    ----------
    model : type or None, default=None
        The neuralforecast model class to wrap. Must be a subclass of
        ``neuralforecast.common._base_model.BaseModel``. If not provided,
        ``_estimator_default_class`` is used.
    input_size : int, default=24
        Number of past observations used as input (lookback window).
    max_steps : int, default=100
        Maximum number of training steps.
    freq : str or None, default=None
        Frequency string (pandas offset alias). If None, auto-inferred
        from the data at fit time.
    **params : dict
        Parameters forwarded to the neuralforecast model constructor.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator (internal).
    freq_ : str
        The inferred or provided frequency string.
    instance_ : BaseModel
        The constructed neuralforecast model instance (from ``BaseClassWrapper``).
    y_columns_ : list of str
        Original target column names from the training data.

    See Also
    --------
    yohou_nixtla.neural.NBEATSForecaster : NBEATS wrapper.
    yohou_nixtla.neural.NHITSForecaster : NHITS wrapper.

    """

    _estimator_name = "model"

    @property
    def _estimator_base_class(self):
        """Return the base class for validation (lazy import to avoid import cost)."""
        from neuralforecast.common._base_model import BaseModel

        return BaseModel

    _estimator_default_class: type | None = None

    def __init__(
        self,
        *,
        input_size: int = 24,
        max_steps: int = 100,
        freq: str | None = None,
        **params,
    ):
        self.input_size = input_size
        self.max_steps = max_steps
        super().__init__(freq=freq, **params)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters of this forecaster.

        Extends ``BaseNixtlaForecaster.get_params`` to include neural-specific
        parameters that are not forwarded to the wrapped model.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        params = super().get_params(deep=deep)
        params["input_size"] = self.input_size
        params["max_steps"] = self.max_steps
        return params

    def set_params(self, **params) -> BaseNeuralForecaster:
        """Set the parameters of this forecaster.

        Extends ``BaseNixtlaForecaster.set_params`` to handle neural-specific
        parameters that are not forwarded to the wrapped model.

        Parameters
        ----------
        **params : dict
            Forecaster parameters.

        Returns
        -------
        self

        """
        neural_params = ("input_size", "max_steps")
        for name in neural_params:
            if name in params:
                setattr(self, name, params.pop(name))
        return super().set_params(**params)

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        **params,
    ) -> BaseNeuralForecaster:
        """Fit the neural forecaster to the training data.

        Does NOT use ``_fit_context`` because neuralforecast models
        require ``h`` (the forecasting horizon) at construction time,
        which is only known at fit time.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``time`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features with ``time`` column.
        forecasting_horizon : int, default=1
            Number of steps to forecast.
        **params : dict
            Additional metadata routing parameters.

        Returns
        -------
        self
            Fitted forecaster.

        Raises
        ------
        ValueError
            If ``forecasting_horizon < 1``.

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be a positive integer, got {forecasting_horizon}.")

        # Inject h, input_size, and max_steps into model params before
        # instantiation. neuralforecast models require h at construction.
        self.params["h"] = forecasting_horizon
        self.params["input_size"] = self.input_size
        self.params["max_steps"] = self.max_steps
        self.instantiate()

        return super().fit(y=y, X=X, forecasting_horizon=forecasting_horizon, **params)

    def _fit_backend(self, nixtla_df: Any, forecasting_horizon: int) -> None:
        """Create and fit the NeuralForecast orchestrator.

        Parameters
        ----------
        nixtla_df : pd.DataFrame
            Training data in Nixtla long format.
        forecasting_horizon : int
            Number of steps to forecast.

        """
        from neuralforecast import NeuralForecast

        nf = NeuralForecast(
            models=[self.instance_],
            freq=self.freq_,
        )
        nf.fit(df=nixtla_df)
        self.nixtla_forecaster_ = nf

    def _predict_backend(self, forecasting_horizon: int, X: pl.DataFrame | None = None) -> Any:
        """Generate raw predictions from the NeuralForecast orchestrator.

        Neural models always predict exactly ``h`` steps (the horizon set
        at model construction). If a shorter horizon is requested, the
        base class ``predict`` will truncate.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to forecast.
        X : pl.DataFrame or None, default=None
            Future exogenous features (unused currently).

        Returns
        -------
        pd.DataFrame
            Raw neuralforecast predictions.

        """
        check_is_fitted(self, ["nixtla_forecaster_"])
        return self.nixtla_forecaster_.predict()

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Neural models predict exactly ``h`` steps (set at construction).
        If ``forecasting_horizon`` is smaller than the training horizon,
        predictions are truncated.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Future exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps to forecast. If None, uses the horizon from fit.
        panel_group_names : list of str or None, default=None
            Panel groups to predict.
        predict_transformed : bool, default=False
            Whether to return transformed predictions.
        **params : dict
            Additional metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Point forecasts with ``observed_time`` and ``time`` columns.

        """
        check_is_fitted(self, ["nixtla_forecaster_"])

        y_pred = super().predict(
            X=X,
            forecasting_horizon=self.fit_forecasting_horizon_,
            panel_group_names=panel_group_names,
            predict_transformed=predict_transformed,
            **params,
        )

        # Neural models always predict h steps; truncate if shorter requested
        h = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_
        if h < len(y_pred):
            y_pred = y_pred.head(h)

        return y_pred
