"""Concrete statsforecast model wrappers for Yohou.

Each class wraps a single statsforecast model, making all of its constructor
parameters available as first-class ``__init__`` kwargs.  Adding a new model
is a 3-line class definition  --  just set ``_estimator_default_class``.
"""

from __future__ import annotations

from statsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
    CrostonClassic,
    HoltWinters,
    Naive,
    SeasonalNaive,
    Theta,
)

from yohou_nixtla.stats._base import BaseStatsForecaster


class AutoARIMAForecaster(BaseStatsForecaster):
    """AutoARIMA forecaster via statsforecast.

    Automatically selects the best ARIMA model using the Hyndman-Khandakar
    algorithm.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    d : int or None, default=None
        Order of first differencing. Auto-detected if None.
    D : int or None, default=None
        Order of seasonal differencing. Auto-detected if None.
    max_p : int, default=5
        Maximum non-seasonal AR order.
    max_q : int, default=5
        Maximum non-seasonal MA order.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.AutoARIMA``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : AutoARIMA
        The constructed AutoARIMA model instance.

    See Also
    --------
    ARIMAForecaster : ARIMA with manually specified orders.
    AutoETSForecaster : Automatic ETS model selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import AutoARIMAForecaster
    >>> forecaster = AutoARIMAForecaster(season_length=12)
    >>> forecaster  # doctest: +ELLIPSIS
    AutoARIMAForecaster(...)

    """

    _estimator_default_class = AutoARIMA

    def __sklearn_tags__(self):
        """Get estimator tags (supports exogenous features)."""
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = False
        return tags


class AutoETSForecaster(BaseStatsForecaster):
    """AutoETS forecaster via statsforecast.

    Automatically selects the best ETS (Error, Trend, Seasonality) model.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    model : str, default="ZZZ"
        ETS model specification. ``"ZZZ"`` selects automatically.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.AutoETS``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : AutoETS
        The constructed AutoETS model instance.

    See Also
    --------
    ETSForecaster : ETS with manually specified parameters.
    AutoARIMAForecaster : Automatic ARIMA model selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import AutoETSForecaster
    >>> forecaster = AutoETSForecaster(season_length=12)
    >>> forecaster  # doctest: +ELLIPSIS
    AutoETSForecaster(...)

    """

    _estimator_default_class = AutoETS


class AutoCESForecaster(BaseStatsForecaster):
    """AutoCES (Complex Exponential Smoothing) forecaster via statsforecast.

    Automatically selects the best CES model.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    model : str, default="Z"
        CES model specification.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.AutoCES``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : AutoCES
        The constructed AutoCES model instance.

    See Also
    --------
    AutoETSForecaster : Automatic ETS model selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import AutoCESForecaster
    >>> forecaster = AutoCESForecaster(season_length=12)
    >>> forecaster  # doctest: +ELLIPSIS
    AutoCESForecaster(...)

    """

    _estimator_default_class = AutoCES


class AutoThetaForecaster(BaseStatsForecaster):
    """AutoTheta forecaster via statsforecast.

    Automatic selection of the best Theta model.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    decomposition_type : str, default="multiplicative"
        Type of seasonal decomposition.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.AutoTheta``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : AutoTheta
        The constructed AutoTheta model instance.

    See Also
    --------
    ThetaForecaster : Theta with manually specified parameters.

    Examples
    --------
    >>> from yohou_nixtla.stats import AutoThetaForecaster
    >>> forecaster = AutoThetaForecaster(season_length=12)
    >>> forecaster  # doctest: +ELLIPSIS
    AutoThetaForecaster(...)

    """

    _estimator_default_class = AutoTheta


class NaiveForecaster(BaseStatsForecaster):
    """Naive forecaster via statsforecast.

    Repeats the last observed value for all forecast steps.

    Parameters
    ----------
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.Naive``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : Naive
        The constructed Naive model instance.

    See Also
    --------
    SeasonalNaiveForecaster : Seasonal naive baseline.

    Examples
    --------
    >>> from yohou_nixtla.stats import NaiveForecaster
    >>> forecaster = NaiveForecaster()
    >>> forecaster  # doctest: +ELLIPSIS
    NaiveForecaster(...)

    """

    _estimator_default_class = Naive


class SeasonalNaiveForecaster(BaseStatsForecaster):
    """Seasonal Naive forecaster via statsforecast.

    Repeats the values from the last seasonal cycle.

    Parameters
    ----------
    season_length : int, default=7
        Length of the seasonal period.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.SeasonalNaive``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : SeasonalNaive
        The constructed SeasonalNaive model instance.

    See Also
    --------
    NaiveForecaster : Non-seasonal naive baseline.

    Examples
    --------
    >>> from yohou_nixtla.stats import SeasonalNaiveForecaster
    >>> forecaster = SeasonalNaiveForecaster(season_length=7)
    >>> forecaster  # doctest: +ELLIPSIS
    SeasonalNaiveForecaster(...)

    """

    _estimator_default_class = SeasonalNaive


class ARIMAForecaster(BaseStatsForecaster):
    """ARIMA forecaster via statsforecast.

    ARIMA model with manually specified orders.

    Parameters
    ----------
    order : tuple, default=(0, 0, 0)
        The ``(p, d, q)`` order of the model.
    season_length : int, default=0
        Length of the seasonal period. 0 means non-seasonal.
    seasonal_order : tuple, default=(0, 0, 0)
        The ``(P, D, Q)`` seasonal order.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.ARIMA``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : ARIMA
        The constructed ARIMA model instance.

    See Also
    --------
    AutoARIMAForecaster : Automatic ARIMA order selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import ARIMAForecaster
    >>> forecaster = ARIMAForecaster(order=(1, 1, 1))
    >>> forecaster  # doctest: +ELLIPSIS
    ARIMAForecaster(...)

    """

    _estimator_default_class = ARIMA

    def __sklearn_tags__(self):
        """Get estimator tags (supports exogenous features)."""
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = False
        return tags


class HoltWintersForecaster(BaseStatsForecaster):
    """Holt-Winters (triple exponential smoothing) forecaster via statsforecast.

    ETS-style model with error, trend, and seasonality components.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    error_type : str, default="A"
        Error type: ``"A"`` (additive) or ``"M"`` (multiplicative).
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to
        ``statsforecast.models.HoltWinters``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : HoltWinters
        The constructed HoltWinters model instance.

    See Also
    --------
    AutoETSForecaster : Automatic ETS model selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import HoltWintersForecaster
    >>> forecaster = HoltWintersForecaster(season_length=12)
    >>> forecaster  # doctest: +ELLIPSIS
    HoltWintersForecaster(...)

    """

    _estimator_default_class = HoltWinters


class ThetaForecaster(BaseStatsForecaster):
    """Theta forecaster via statsforecast.

    Standard Theta method for time series forecasting.

    Parameters
    ----------
    season_length : int, default=1
        Length of the seasonal period.
    decomposition_type : str, default="multiplicative"
        Type of seasonal decomposition.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.Theta``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : Theta
        The constructed Theta model instance.

    See Also
    --------
    AutoThetaForecaster : Automatic Theta model selection.

    Examples
    --------
    >>> from yohou_nixtla.stats import ThetaForecaster
    >>> forecaster = ThetaForecaster()
    >>> forecaster  # doctest: +ELLIPSIS
    ThetaForecaster(...)

    """

    _estimator_default_class = Theta


class CrostonForecaster(BaseStatsForecaster):
    """Croston's method forecaster via statsforecast.

    Designed for intermittent demand forecasting.

    Parameters
    ----------
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    **params : dict
        Additional parameters forwarded to ``statsforecast.models.CrostonClassic``.

    Attributes
    ----------
    nixtla_forecaster_ : StatsForecast
        The fitted Nixtla orchestrator.
    instance_ : CrostonClassic
        The constructed Croston model instance.

    Examples
    --------
    >>> from yohou_nixtla.stats import CrostonForecaster
    >>> forecaster = CrostonForecaster()
    >>> forecaster  # doctest: +ELLIPSIS
    CrostonForecaster(...)

    """

    _estimator_default_class = CrostonClassic
