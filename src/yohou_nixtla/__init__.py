"""Yohou-Nixtla: Nixtla forecasting library integration for Yohou."""

from importlib.metadata import version

from yohou_nixtla._base import BaseNixtlaForecaster
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

__version__ = version(__name__)

__all__ = [
    "__version__",
    # Base
    "BaseNixtlaForecaster",
    # Stats forecasters
    "BaseStatsForecaster",
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "AutoCESForecaster",
    "AutoThetaForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ARIMAForecaster",
    "HoltWintersForecaster",
    "ThetaForecaster",
    "CrostonForecaster",
    # Neural forecasters
    "BaseNeuralForecaster",
    "NBEATSForecaster",
    "NHITSForecaster",
    "MLPForecaster",
    "PatchTSTForecaster",
    "TimesNetForecaster",
]


def __getattr__(name: str):
    """Lazily import neural forecasters to avoid requiring neuralforecast."""
    _neural_names = {
        "BaseNeuralForecaster",
        "MLPForecaster",
        "NBEATSForecaster",
        "NHITSForecaster",
        "PatchTSTForecaster",
        "TimesNetForecaster",
    }
    if name in _neural_names:
        from yohou_nixtla import neural

        return getattr(neural, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
