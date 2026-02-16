"""Base statsforecast forecaster class for Yohou.

This module provides ``BaseStatsForecaster``, a base class that integrates
statsforecast models into the yohou framework via ``BaseNixtlaForecaster``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.utils.validation import check_is_fitted
from sklearn_wrap.base import _fit_context
from statsforecast import StatsForecast
from statsforecast.models import _TS

from yohou_nixtla._base import BaseNixtlaForecaster


class BaseStatsForecaster(BaseNixtlaForecaster):
    """Base class for statsforecast model wrappers in yohou.

    Wraps a statsforecast model class using the ``BaseNixtlaForecaster``
    pattern, making it fully compatible with yohou's forecaster API
    (``fit`` / ``predict`` / ``observe`` / ``clone`` / ``get_params``).

    Subclasses only need to set ``_estimator_default_class`` to a specific
    statsforecast model class (e.g., ``AutoARIMA``).

    Parameters
    ----------
    model : type or None, default=None
        The statsforecast model class to wrap. Must be a subclass of
        ``statsforecast.models._TS``. If not provided,
        ``_estimator_default_class`` is used.
    freq : str or None, default=None
        Frequency string (pandas offset alias). If None, auto-inferred
        from the data at fit time.
    **params : dict
        Parameters forwarded to the statsforecast model constructor.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla StatsForecast orchestrator (internal).
    freq_ : str
        The inferred or provided frequency string.
    instance_ : _TS
        The constructed statsforecast model instance (from ``BaseClassWrapper``).
    y_columns_ : list of str
        Original target column names from the training data.

    See Also
    --------
    yohou_nixtla.stats.AutoARIMAForecaster : AutoARIMA wrapper.
    yohou_nixtla.stats.NaiveForecaster : Naive baseline wrapper.

    """

    _estimator_name = "model"
    _estimator_base_class = _TS
    _estimator_default_class: type | None = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        **params,
    ) -> BaseStatsForecaster:
        """Fit the statsforecast model to the training data.

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

        """
        return super().fit(y=y, X=X, forecasting_horizon=forecasting_horizon, **params)

    def _fit_backend(self, nixtla_df: Any, forecasting_horizon: int) -> None:
        """Create and fit the StatsForecast orchestrator.

        Parameters
        ----------
        nixtla_df : pd.DataFrame
            Training data in Nixtla long format.
        forecasting_horizon : int
            Number of steps to forecast.

        """
        sf = StatsForecast(
            models=[self.instance_],
            freq=self.freq_,
        )
        sf.fit(df=nixtla_df)
        self.nixtla_forecaster_ = sf

    def _predict_backend(self, forecasting_horizon: int, X: pl.DataFrame | None = None) -> Any:
        """Generate raw predictions from the StatsForecast orchestrator.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to forecast.
        X : pl.DataFrame or None, default=None
            Future exogenous features (unused by stats models).

        Returns
        -------
        pd.DataFrame
            Raw statsforecast predictions.

        """
        check_is_fitted(self, ["nixtla_forecaster_"])
        return self.nixtla_forecaster_.predict(h=forecasting_horizon)
