"""Yohou-Nixtla: Nixtla forecasting library integration for Yohou."""

from importlib.metadata import version

from yohou_nixtla._base import BaseNixtlaForecaster
from yohou_nixtla.neural import (
    BaseNeuralForecaster,
    MLPForecaster,
    NBEATSForecaster,
    NHITSForecaster,
    PatchTSTForecaster,
    TimesNetForecaster,
)
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
