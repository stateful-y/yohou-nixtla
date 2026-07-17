---
template: api-index.html
---

# API Reference

Complete API reference for all Yohou-Nixtla classes and functions. Use the
search box to filter, or click any name to see full documentation.

## Quick Reference

### Stats Forecasters

Wrap Nixtla's `statsforecast` library. Fast and interpretable, best for
structured seasonal data.

| Class | Model | Best For |
|-------|-------|---------|
| `AutoARIMAForecaster` | AutoARIMA | General-purpose, auto model selection |
| `AutoETSForecaster` | AutoETS | Trend and seasonality, auto selection |
| `AutoCESForecaster` | AutoCES | Complex exponential smoothing |
| `AutoThetaForecaster` | AutoTheta | Automatic Theta model selection |
| `ARIMAForecaster` | ARIMA | Manual ARIMA order specification |
| `HoltWintersForecaster` | HoltWinters | Manually specified ETS components |
| `ThetaForecaster` | Theta | Manually specified Theta model |
| `NaiveForecaster` | Naive | Baseline: repeat last value |
| `SeasonalNaiveForecaster` | SeasonalNaive | Baseline: repeat last season |
| `CrostonForecaster` | Croston | Intermittent (sparse) demand |

### Neural Forecasters

Wrap Nixtla's `neuralforecast` library. Suited for complex patterns and large
datasets. Require `pip install yohou-nixtla[neural]`.

| Class | Model | Best For |
|-------|-------|---------|
| `NBEATSForecaster` | N-BEATS | General deep learning baseline |
| `NHITSForecaster` | N-HiTS | Long-horizon forecasting |
| `MLPForecaster` | MLP | Simple feedforward baseline |
| `PatchTSTForecaster` | PatchTST | Transformer, channel-independent |
| `TimesNetForecaster` | TimesNet | CNN-based temporal modeling |

### Utilities

| Function | Module | Description |
|----------|--------|-------------|
| `infer_freq(y)` | `yohou_nixtla._conversion` | Map polars DataFrame time interval to Nixtla frequency alias |
| `yohou_to_nixtla(y, X)` | `yohou_nixtla._conversion` | Convert Yohou wide-format to Nixtla long-format |
| `nixtla_to_yohou(forecast_df, y_columns)` | `yohou_nixtla._conversion` | Convert Nixtla long-format predictions to Yohou wide-format |

## Common Parameters

All forecasters inherit these parameters from `BaseNixtlaForecaster`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq` | `str \| None` | `None` | Frequency string. Auto-inferred from data if `None`. |
| `actual_transformer` | transformer or `None` | `None` | Applied to exogenous features before fitting and predicting. |
| `target_transformer` | transformer or `None` | `None` | Applied to target before fitting; inverse-applied after predicting. |
| `target_as_feature` | `"transformed" \| "raw" \| None` | `None` | Include lagged target values as additional features. |

Stats forecasters additionally accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `season_length` | `int` | `1` | Seasonal period length. |
| `n_jobs` | `int` | `1` | Parallel jobs for multi-series fitting. |

Neural forecasters additionally accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | `24` | Lookback window (number of past steps). |
| `max_steps` | `int` | `100` | Maximum training steps. |

## Lifecycle Methods

All forecasters implement the `BasePointForecaster` lifecycle:

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `fit(y, X=None, forecasting_horizon=1)` | Train on historical data. |
| `predict` | `predict(forecasting_horizon, X=None)` | Generate point forecasts. |
| `observe` | `observe(y_new)` | Append new observations without retraining. |
| `rewind` | `rewind(y)` | Reset the internal observation state. |

## Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `nixtla_forecaster_` | `StatsForecast \| NeuralForecast` | The fitted Nixtla orchestrator. |
| `freq_` | `str` | The inferred or provided frequency string. |
| `y_columns_` | `list[str]` | Target column names from training data. |
| `instance_` | model instance | The constructed backend model instance. |

---

<!-- API_TABLE -->
