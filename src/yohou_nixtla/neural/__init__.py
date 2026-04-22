"""Neuralforecast model wrappers for Yohou.

Requires the ``neural`` extra::

    pip install yohou-nixtla[neural]
"""

try:
    from yohou_nixtla.neural._base import BaseNeuralForecaster
    from yohou_nixtla.neural._forecasters import (
        MLPForecaster,
        NBEATSForecaster,
        NHITSForecaster,
        PatchTSTForecaster,
        TimesNetForecaster,
    )
except ImportError as _err:
    raise ImportError(
        "Neural forecasters require the 'neural' extra. Install it with: pip install yohou-nixtla[neural]"
    ) from _err

__all__ = [
    "BaseNeuralForecaster",
    "MLPForecaster",
    "NBEATSForecaster",
    "NHITSForecaster",
    "PatchTSTForecaster",
    "TimesNetForecaster",
]
