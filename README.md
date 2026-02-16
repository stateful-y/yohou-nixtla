<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-nixtla/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-nixtla/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/yohou-nixtla/main/docs/assets/logo_light.png" alt="Yohou-Nixtla">
  </picture>
</p>


[![Python Version](https://img.shields.io/pypi/pyversions/yohou_nixtla)](https://pypi.org/project/yohou_nixtla/)
[![License](https://img.shields.io/github/license/stateful-y/yohou-nixtla)](https://github.com/stateful-y/yohou-nixtla/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/yohou_nixtla)](https://pypi.org/project/yohou_nixtla/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/yohou_nixtla)](https://anaconda.org/conda-forge/yohou_nixtla)
[![codecov](https://codecov.io/gh/stateful-y/yohou-nixtla/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/yohou-nixtla)

## What is Yohou-Nixtla?

**Yohou-Nixtla** brings the power of [Nixtla's forecasting ecosystem](https://nixtla.io/) to [Yohou](https://github.com/stateful-y/yohou), providing Yohou-compatible wrappers for statistical, machine learning, and deep learning time series models.

This integration enables you to use Nixtla's high-performance forecasters (StatsForecast, NeuralForecast) within Yohou's unified API for time series forecasting. All models work seamlessly with Yohou's features: polars DataFrames, panel data support, cross-validation, and hyperparameter search via GridSearchCV/RandomizedSearchCV.

## What are the features of Yohou-Nixtla?

- **Statistical Models**: AutoARIMA, AutoETS, AutoTheta, ARIMA, Holt-Winters, Naive, and more from StatsForecast, providing fast, production-ready statistical forecasters.
- **Neural Models**: NBEATS, NHITS, MLP, PatchTST, TimesNet from NeuralForecast, offering state-of-the-art deep learning architectures.
- **Panel Data**: Native support for multiple time series with `__` column naming convention (e.g., `sales__store_1`, `sales__store_2`).
- **Yohou Compatible**: Full `fit/predict`, `get_params/set_params`, `clone` compatibility. Works with GridSearchCV, pipelines, and the Yohou ecosystem.
- **Polars Native**: All data handling uses polars DataFrames for high-performance time series operations.

> **Note**: Nixtla's MLForecast is not wrapped as Yohou already provides `PointReductionForecaster`, which turns any scikit-learn regressor (Ridge, LightGBM, XGBoost, …) into a recursive multi-step forecaster with full support for feature transformers, target transformers, and panel data.

## How to install Yohou-Nixtla?

Install the Yohou-Nixtla package using `pip`:

```bash
pip install yohou_nixtla
```

or using `uv`:

```bash
uv pip install yohou_nixtla
```

or using `conda`:

```bash
conda install -c conda-forge yohou_nixtla
```

or using `mamba`:

```bash
mamba install -c conda-forge yohou_nixtla
```

or alternatively, add `yohou_nixtla` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with Yohou-Nixtla?

### 1. Fit a Statistical Forecaster

Use AutoARIMA for automatic ARIMA model selection:

```python
import polars as pl
from yohou_nixtla import AutoARIMAForecaster

# Load your time series data (must have a "time" column)
y = pl.DataFrame({
    "time": pl.datetime_range(start="2020-01-01", end="2020-12-31", interval="1d", eager=True),
    "sales": [100 + i * 0.5 + (i % 7) * 10 for i in range(366)],
})

# Fit and predict
forecaster = AutoARIMAForecaster(season_length=7)
forecaster.fit(y, forecasting_horizon=14)
y_pred = forecaster.predict()
```

### 2. Train Deep Learning Models

Neural models for complex patterns:

```python
from yohou_nixtla import NHITSForecaster

forecaster = NHITSForecaster(input_size=30, max_steps=100)
forecaster.fit(y, forecasting_horizon=14)
y_pred = forecaster.predict()
```

### 3. Panel Data Forecasting

Forecast multiple time series simultaneously:

```python
# Panel data with __ separator
y_panel = pl.DataFrame({
    "time": pl.datetime_range(start="2020-01-01", end="2020-12-31", interval="1d", eager=True),
    "sales__store_1": [...],
    "sales__store_2": [...],
})

forecaster = AutoARIMAForecaster(season_length=7)
forecaster.fit(y_panel, forecasting_horizon=14)
y_pred = forecaster.predict()  # Predictions for all stores
```

## How do I use Yohou-Nixtla?

Full documentation is available at [https://yohou-nixtla.readthedocs.io/](https://yohou-nixtla.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://yohou-nixtla.readthedocs.io/en/latest/pages/examples/](https://yohou-nixtla.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `marimo edit examples/hello.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/yohou-nixtla/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/yohou-nixtla/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/yohou-nixtla/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [GitHub Discussions](https://github.com/stateful-y/yohou-nixtla/discussions).

## Where can I learn more?

Here are the main Yohou-Nixtla resources:

- Full documentation: [https://yohou-nixtla.readthedocs.io/](https://yohou-nixtla.readthedocs.io/)
- GitHub Discussions: [https://github.com/stateful-y/yohou-nixtla/discussions](https://github.com/stateful-y/yohou-nixtla/discussions)
- Interactive Examples: [https://yohou-nixtla.readthedocs.io/en/latest/pages/examples/](https://yohou-nixtla.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/yohou-nixtla/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou-nixtla/blob/main/LICENSE).

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
