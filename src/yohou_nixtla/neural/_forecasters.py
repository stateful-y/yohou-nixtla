"""Concrete neuralforecast model wrappers for Yohou.

Each class wraps a single neuralforecast model, making its constructor
parameters available as first-class ``__init__`` kwargs.  Adding a new model
is a 3-line class definition  --  just set ``_estimator_default_class``.
"""

from __future__ import annotations

from neuralforecast.models import MLP, NBEATS, NHITS, PatchTST, TimesNet

from yohou_nixtla.neural._base import BaseNeuralForecaster


class NBEATSForecaster(BaseNeuralForecaster):
    """NBEATS (Neural Basis Expansion Analysis) forecaster via neuralforecast.

    Deep neural architecture that uses backward and forward residual links
    with basis expansion.

    Parameters
    ----------
    input_size : int, default=24
        Lookback window size.
    max_steps : int, default=100
        Maximum training steps.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Additional parameters forwarded to
        ``neuralforecast.models.NBEATS``.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator.
    instance_ : NBEATS
        The constructed NBEATS model instance.

    See Also
    --------
    NHITSForecaster : Hierarchical interpolation variant.

    References
    ----------
    .. [1] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio,
       "N-BEATS: Neural basis expansion analysis for interpretable time
       series forecasting," ICLR, 2020.

    Examples
    --------
    >>> from yohou_nixtla.neural import NBEATSForecaster
    >>> forecaster = NBEATSForecaster(input_size=12, max_steps=50)
    >>> forecaster  # doctest: +ELLIPSIS
    NBEATSForecaster(...)

    """

    _estimator_default_class = NBEATS


class NHITSForecaster(BaseNeuralForecaster):
    """NHITS (Neural Hierarchical Interpolation) forecaster via neuralforecast.

    Extension of NBEATS with multi-rate signal sampling and hierarchical
    interpolation for improved long-horizon forecasting.

    Parameters
    ----------
    input_size : int, default=24
        Lookback window size.
    max_steps : int, default=100
        Maximum training steps.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Additional parameters forwarded to
        ``neuralforecast.models.NHITS``.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator.
    instance_ : NHITS
        The constructed NHITS model instance.

    See Also
    --------
    NBEATSForecaster : Neural basis expansion variant.

    References
    ----------
    .. [1] C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza,
       M. Mergenthaler-Canseco, and A. Dubrawski, "N-HiTS: Neural
       hierarchical interpolation for time series forecasting,"
       AAAI, 2023.

    Examples
    --------
    >>> from yohou_nixtla.neural import NHITSForecaster
    >>> forecaster = NHITSForecaster(input_size=12, max_steps=50)
    >>> forecaster  # doctest: +ELLIPSIS
    NHITSForecaster(...)

    """

    _estimator_default_class = NHITS


class MLPForecaster(BaseNeuralForecaster):
    """MLP (Multi-Layer Perceptron) forecaster via neuralforecast.

    Simple feedforward neural network for time series forecasting.

    Parameters
    ----------
    input_size : int, default=24
        Lookback window size.
    max_steps : int, default=100
        Maximum training steps.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Additional parameters forwarded to ``neuralforecast.models.MLP``.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator.
    instance_ : MLP
        The constructed MLP model instance.

    See Also
    --------
    NBEATSForecaster : Neural basis expansion architecture.
    NHITSForecaster : Hierarchical interpolation variant.

    Examples
    --------
    >>> from yohou_nixtla.neural import MLPForecaster
    >>> forecaster = MLPForecaster(input_size=12, max_steps=50)
    >>> forecaster  # doctest: +ELLIPSIS
    MLPForecaster(...)

    """

    _estimator_default_class = MLP


class PatchTSTForecaster(BaseNeuralForecaster):
    """PatchTST (Patch Time Series Transformer) forecaster via neuralforecast.

    Transformer-based architecture that operates on patches of the time series.

    Parameters
    ----------
    input_size : int, default=24
        Lookback window size.
    max_steps : int, default=100
        Maximum training steps.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Additional parameters forwarded to
        ``neuralforecast.models.PatchTST``.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator.
    instance_ : PatchTST
        The constructed PatchTST model instance.

    See Also
    --------
    TimesNetForecaster : Temporal 2D-variation modeling.
    NBEATSForecaster : Neural basis expansion architecture.

    References
    ----------
    .. [1] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam,
       "A time series is worth 64 words: Long-term forecasting with
       transformers," ICLR, 2023.

    Examples
    --------
    >>> from yohou_nixtla.neural import PatchTSTForecaster
    >>> forecaster = PatchTSTForecaster(input_size=24, max_steps=50)
    >>> forecaster  # doctest: +ELLIPSIS
    PatchTSTForecaster(...)

    """

    _estimator_default_class = PatchTST

    def __sklearn_tags__(self):
        """Get estimator tags (supports exogenous features)."""
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = False
        return tags


class TimesNetForecaster(BaseNeuralForecaster):
    """TimesNet (Temporal 2D-Variation Modeling) forecaster via neuralforecast.

    Transforms 1D time series into 2D tensors to capture multi-periodicity.

    Parameters
    ----------
    input_size : int, default=24
        Lookback window size.
    max_steps : int, default=100
        Maximum training steps.
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include target values as additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Additional parameters forwarded to
        ``neuralforecast.models.TimesNet``.

    Attributes
    ----------
    nixtla_forecaster_ : NeuralForecast
        The fitted NeuralForecast orchestrator.
    instance_ : TimesNet
        The constructed TimesNet model instance.

    See Also
    --------
    PatchTSTForecaster : Patch-based transformer architecture.
    NBEATSForecaster : Neural basis expansion architecture.

    References
    ----------
    .. [1] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long,
       "TimesNet: Temporal 2D-variation modeling for general time
       series analysis," ICLR, 2023.

    Examples
    --------
    >>> from yohou_nixtla.neural import TimesNetForecaster
    >>> forecaster = TimesNetForecaster(input_size=24, max_steps=50)
    >>> forecaster  # doctest: +ELLIPSIS
    TimesNetForecaster(...)

    """

    _estimator_default_class = TimesNet

    def __sklearn_tags__(self):
        """Get estimator tags (supports exogenous features)."""
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = False
        return tags
