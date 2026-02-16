# User Guide

This guide provides comprehensive documentation for Yohou-Nixtla.

## Overview

Yohou-Nixtla is a bridge between the [Yohou](https://github.com/stateful-y/yohou) time series forecasting framework and the [Nixtla](https://nixtla.io/) ecosystem. It wraps Nixtla's forecasting libraries (**StatsForecast** and **NeuralForecast**) as Yohou-compatible forecasters, so you get the best of both worlds: Nixtla's state-of-the-art models with Yohou's scikit-learn-compatible API.

Every forecaster in this package uses dual inheritance: it inherits from both `BaseClassWrapper` (for seamless sklearn `clone`/`get_params`/`set_params` support) and `BasePointForecaster` (for Yohou's `fit`/`predict`/`observe`/`rewind` lifecycle). Data is automatically converted between Yohou's polars wide-format and Nixtla's pandas long-format via the built-in conversion utilities.

## Prerequisites

Before diving into Yohou-Nixtla, it's helpful to understand:

### Yohou

Yohou is a scikit-learn-compatible time series forecasting framework built on polars. It provides the `BasePointForecaster` API that all Yohou-Nixtla forecasters inherit from, including `fit`, `predict`, `observe`, and `rewind` methods.

Learn more: [Yohou Documentation](https://yohou.readthedocs.io/)

### Nixtla

Nixtla provides open-source forecasting libraries: StatsForecast (classical statistics) and NeuralForecast (deep learning). Yohou-Nixtla wraps models from both backends.

Learn more: [Nixtla Documentation](https://nixtla.io/)

## Why Yohou-Nixtla?

Nixtla's libraries are powerful but operate on pandas DataFrames in long format with their own API conventions. Yohou-Nixtla solves three problems:

1. **Unified API**: Use the same `fit`/`predict`/`observe`/`rewind` interface regardless of whether you're running an ARIMA model or a neural network.
2. **Polars-native**: Work with polars DataFrames end-to-end. The conversion to/from Nixtla's pandas format happens automatically under the hood.
3. **Scikit-learn compatibility**: All forecasters support `clone()`, `get_params()`, `set_params()`, and integrate with Yohou's cross-validation, hyperparameter search, and scoring utilities.

### For Yohou Users

If you're already using Yohou and want access to Nixtla's model zoo, Yohou-Nixtla offers:

- **Drop-in forecasters**: Use `AutoARIMAForecaster` or `NBEATSForecaster` exactly like any other Yohou forecaster: same API, same data format, same workflow.
- **Composability**: Combine Nixtla forecasters with Yohou's `DecompositionPipeline`, `ColumnForecaster`, `GridSearchCV`, and other meta-estimators.
- **Panel data support**: Nixtla's native multi-series handling is exposed through Yohou's `__` column naming convention.

### For Nixtla Users

If you're familiar with Nixtla libraries and want sklearn integration, you'll appreciate:

- **Familiar models**: The same AutoARIMA, NBEATS, NHITS, and other models you know, just wrapped in a consistent interface.
- **Incremental updates**: Use `observe()` to add new observations without refitting, and `rewind()` to manage the observation window.
- **Ecosystem integration**: Plug Nixtla models into sklearn pipelines, cross-validation, and hyperparameter optimization.

## Core Concepts

### Two Forecasting Backends

Yohou-Nixtla provides forecasters from two Nixtla backends:

| Backend | Library | Forecasters | Best For |
|---------|---------|-------------|----------|
| **Stats** | `statsforecast` | AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive, SeasonalNaive, ARIMA, HoltWinters, Theta, Croston | Classical time series with known seasonal patterns |
| **Neural** | `neuralforecast` | NBEATS, NHITS, MLP, PatchTST, TimesNet | Complex patterns, large datasets, deep learning |

**Stats forecasters** are fast and interpretable. They require only target data (`y`) and handle seasonality through `season_length`.

**Neural forecasters** use PyTorch models and are best suited for large datasets or complex non-linear patterns. They require `input_size` and `max_steps` configuration.

!!! example "Interactive Example"
    See [**Model Comparison**](/examples/model_comparison/) ([View](/examples/model_comparison/) | [Editable](/examples/model_comparison/edit/)) for a side-by-side comparison of Stats and Neural forecasters on the Air Passengers dataset.

### Data Conversion

All data conversion between Yohou and Nixtla formats is handled automatically by the `_conversion` module:

- **`yohou_to_nixtla(y, X)`**: Converts a polars wide-format DataFrame to Nixtla's pandas long-format (`unique_id` / `ds` / `y` columns).
- **`nixtla_to_yohou(forecast_df, y_columns)`**: Converts Nixtla's long-format predictions back to polars wide-format.
- **`infer_freq(y)`**: Maps polars interval strings (e.g., `"1d"`, `"1mo"`) to pandas offset aliases (e.g., `"D"`, `"MS"`).

You never need to call these functions directly; they're used internally by the forecasters.

### Panel Data

Yohou-Nixtla supports panel (grouped) time series through Yohou's `__` column naming convention:

```python
import polars as pl

# Panel data: sales for multiple stores
y = pl.DataFrame({
    "time": dates,
    "sales__store_1": [...],
    "sales__store_2": [...],
    "sales__store_3": [...],
})

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
# y_pred has columns: "time", "sales__store_1", "sales__store_2", "sales__store_3"
```

Each group (e.g., `store_1`, `store_2`) is modeled independently by Nixtla, and the results are recombined into Yohou's wide format automatically.

!!! example "Interactive Example"
    See [**Panel Data**](/examples/panel_data/) ([View](/examples/panel_data/) | [Editable](/examples/panel_data/edit/)) for a hands-on walkthrough of multi-series forecasting with the `__` convention.

## Key Features

### 1. Unified Fit/Predict Interface

All forecasters share the same API:

```python
forecaster.fit(y, forecasting_horizon=12)      # Train
y_pred = forecaster.predict(forecasting_horizon=12)  # Forecast
forecaster.observe(y_new)                       # Add observations
forecaster.rewind(y)                            # Rewind state
```

### 2. Automatic Frequency Detection

The `infer_freq` utility automatically maps polars time intervals to Nixtla-compatible frequency strings. Supported intervals include seconds, minutes, hours, days, weeks, months, quarters, and years.

### 3. Exogenous Feature Support

Neural forecasters accept exogenous features (`X`) alongside the target (`y`):

```python
forecaster = NBEATSForecaster(input_size=24, max_steps=100)
forecaster.fit(y, X=X_train, forecasting_horizon=12)
y_pred = forecaster.predict(X=X_test, forecasting_horizon=12)
```

### 4. Scikit-learn Metadata Routing

All forecasters support sklearn's metadata routing, enabling integration with Yohou's `GridSearchCV`, `RandomizedSearchCV`, and other meta-estimators.

### 5. Clone and Parameter Management

Every forecaster supports `clone()`, `get_params(deep=True)`, and `set_params()` for hyperparameter search compatibility:

```python
from sklearn.base import clone

original = AutoARIMAForecaster(season_length=12, max_p=5)
cloned = clone(original)  # Independent copy with same parameters
```

### 6. Parallel Execution

Stats forecasters support parallel execution via the `n_jobs` parameter for multi-series data.

## Configuration

### Stats Forecasters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `season_length` | int | 1 | Length of the seasonal period |
| `freq` | str \| None | None | Frequency string (auto-detected if None) |
| `n_jobs` | int | 1 | Number of parallel jobs |

### Neural Forecasters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | 24 | Number of input time steps |
| `max_steps` | int | 100 | Maximum training steps |
| `freq` | str \| None | None | Frequency string (auto-detected if None) |

## Best Practices

### 1. Choose the Right Backend

- Start with **Stats** forecasters (e.g., `AutoARIMAForecaster`) for quick baselines. They're fast and require minimal configuration.
- Use **Neural** forecasters for large-scale datasets or when classical methods underperform.

### 2. Set Season Length Correctly

For Stats forecasters, `season_length` is critical. Common values:

- Daily data with weekly seasonality: `season_length=7`
- Monthly data with yearly seasonality: `season_length=12`
- Hourly data with daily seasonality: `season_length=24`

### 3. Use Observe for Streaming Scenarios

Call `observe()` instead of `refit()` when new observations arrive. This appends to the internal observation buffer without retraining the model, which is much faster for online/streaming use cases.

### 4. Leverage Panel Data

When forecasting many related time series, use Yohou's `__` column convention to fit all series at once. This is more efficient than fitting separate forecasters and enables Nixtla's native multi-series optimizations.

## Limitations and Considerations

1. **Point forecasts only**: Yohou-Nixtla currently wraps forecasters as `BasePointForecaster`. Interval/probabilistic forecasts from Nixtla are not yet exposed.

2. **Polars ↔ pandas overhead**: Each `fit`/`predict` call converts data between polars and pandas. For very high-frequency prediction loops, this may add latency.

3. **Neural forecaster dependencies**: Neural forecasters require PyTorch and `neuralforecast`, which are large dependencies. Install them separately if not needed.

4. **No custom model wrapping**: The current release provides a fixed set of 20 wrapped models. Custom Nixtla models require subclassing a base forecaster.

## FAQ

### Can I use Nixtla forecasters in a Yohou pipeline?

Yes. All Yohou-Nixtla forecasters are fully compatible with `DecompositionPipeline`, `ColumnForecaster`, `ForecastedFeatureForecaster`, and other Yohou meta-estimators.

### Do I need to install both Nixtla backends?

No. Install only the backend you need: `statsforecast` for Stats or `neuralforecast` for Neural forecasters.

### How does frequency detection work?

When `freq=None` (the default), the forecaster calls `infer_freq(y)` at fit time to detect the time interval from the polars DataFrame's `"time"` column. You can override this by setting `freq` explicitly (e.g., `freq="MS"` for monthly data).

## Next Steps

- **Try it out**: Follow the [Getting Started](getting-started.md) guide
- **See examples**: Explore the [Examples](examples.md) page
- **Browse the API**: Check the [API Reference](api-reference.md) for all classes
- **Contribute**: Read the [Contributing Guide](contributing.md)
