# How to Configure Forecasters

This guide shows you how to configure Yohou-Nixtla forecasters for common
scenarios. Use this when you need to tune a model for your specific data
frequency, seasonal pattern, or performance requirements.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Basic familiarity with the `fit`/`predict` pattern

## Configure a Stats Forecaster

### 1. Set the seasonal period

The `season_length` parameter controls how many time steps make one seasonal
cycle. Choose based on your data's granularity:

| Data frequency | Pattern | `season_length` |
|----------------|---------|-----------------|
| Daily | Weekly seasonality | 7 |
| Monthly | Annual seasonality | 12 |
| Hourly | Daily seasonality | 24 |
| Quarterly | Annual seasonality | 4 |
| No seasonality | - | 1 (default) |

```python
from yohou_nixtla import AutoARIMAForecaster

# Monthly data with annual seasonality
forecaster = AutoARIMAForecaster(season_length=12)
```

### 2. Override the frequency string

By default, the forecaster infers the data frequency from your polars
DataFrame. To override:

```python
forecaster = AutoARIMAForecaster(season_length=12, freq="MS")
```

Common Nixtla frequency aliases:

| Polars interval | freq alias | Meaning |
|-----------------|------------|---------|
| `1d` | `"D"` | Daily |
| `1w` | `"W"` | Weekly |
| `1mo` | `"MS"` | Monthly (month start) |
| `3mo` | `"QS"` | Quarterly |
| `1y` | `"YS"` | Annual |
| `1h` | `"H"` | Hourly |

### 3. Enable parallel fitting for panel data

For multi-series (panel) data, pass `n_jobs=-1` to use all available CPU cores:

```python
forecaster = AutoARIMAForecaster(season_length=12, n_jobs=-1)
```

## Configure a Neural Forecaster

### 1. Set the lookback window and training budget

`input_size` controls how many historical steps the model sees. `max_steps`
controls training duration.

```python
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(input_size=24, max_steps=500)
```

Guidelines:

- Set `input_size` to at least twice your `forecasting_horizon`
- For short datasets (fewer than 200 observations), keep `max_steps` at 100-300
- For larger datasets or complex patterns, increase `max_steps` to 500-1000

### 2. Tune the learning rate

If training loss is unstable or the model underfits, adjust the learning rate:

```python
forecaster = NBEATSForecaster(input_size=24, max_steps=500, learning_rate=1e-4)
```

## Configure Target Transformation

Apply a transformer to the target before fitting and invert it after predicting.
This is useful for normalizing skewed or non-stationary series.

```python
from sklearn.preprocessing import StandardScaler
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(
    season_length=12,
    target_transformer=StandardScaler(),
)
```

The transformer must implement `fit_transform` and `inverse_transform`.

## Configure Feature Transformation

When using exogenous features, apply a transformer to the feature matrix:

```python
from sklearn.preprocessing import MinMaxScaler
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=100,
    feature_transformer=MinMaxScaler(),
)
```

## Use Lagged Target as a Feature

Set `target_as_feature` to include lagged target values as exogenous features:

```python
# Use transformed target lags as features
forecaster = AutoARIMAForecaster(
    season_length=12,
    target_as_feature="transformed",
)

# Use raw (un-transformed) target lags as features
forecaster = AutoARIMAForecaster(
    season_length=12,
    target_as_feature="raw",
)
```

## See Also

- [Concepts](../explanation/concepts.md): forecasting backends and configuration options
- [API Reference](../reference/api.md): full parameter list for each forecaster class
- [Troubleshooting](troubleshooting.md): fix common configuration errors
