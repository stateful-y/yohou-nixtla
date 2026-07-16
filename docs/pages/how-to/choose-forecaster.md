# How to Choose the Right Forecaster

This guide shows you how to select the right Yohou-Nixtla forecaster for your
data and use case.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))

<!-- COMPANION_NOTEBOOKS -->

## Establish a baseline first

Before comparing models, fit a `SeasonalNaiveForecaster` so you have a
reference any production model must beat:

```python
from yohou_nixtla import SeasonalNaiveForecaster

baseline = SeasonalNaiveForecaster(season_length=12)
baseline.fit(y, forecasting_horizon=12)
y_baseline = baseline.predict(forecasting_horizon=12)
```

If a more complex model cannot beat this baseline, revisit your data quality
or feature engineering before trying harder models.

## Pick a Stats forecaster

Start with Stats as they train in seconds and need minimal configuration.

### `AutoARIMAForecaster` (default choice)

Automatically selects the best ARIMA order. Robust across a wide range of
data patterns. Also accepts exogenous features via `X`.

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
```

### `AutoETSForecaster` for trend or multiplicative seasonality

Best when the data has a clear trend or the seasonal amplitude grows with
the level.

```python
from yohou_nixtla import AutoETSForecaster

forecaster = AutoETSForecaster(season_length=12)
```

### `CrostonForecaster` for intermittent demand

If your series contains many zeros (sparse demand), standard models overfit
the non-zero spikes. Use Croston instead:

```python
from yohou_nixtla import CrostonForecaster

forecaster = CrostonForecaster()
```

### Other Stats forecasters

| Forecaster | When to reach for it |
|------------|---------------------|
| `AutoCESForecaster` | Complex exponential smoothing with automatic selection |
| `AutoThetaForecaster` | Strong baseline for monthly M3/M4 competition data |
| `ARIMAForecaster` | You know the exact `(p, d, q)` order; supports exogenous features |
| `HoltWintersForecaster` | You want explicit control over error, trend, and seasonality types |
| `ThetaForecaster` | Simple, fast decomposition; good for short seasonal series |

## Switch to Neural when Stats fall short

Consider Neural forecasters when:

- Your dataset has more than 5,000 observations per series
- Stats models consistently underfit after tuning
- You need a transformer-based or deep learning architecture

Neural forecasters take longer to train and benefit from GPU acceleration.

### `NBEATSForecaster` (default neural choice)

General-purpose, no preprocessing requirements:

```python
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(input_size=24, max_steps=500)
```

### `NHITSForecaster` for long horizons

Outperforms N-BEATS when the forecast horizon is long relative to the input
window:

```python
from yohou_nixtla import NHITSForecaster

forecaster = NHITSForecaster(input_size=48, max_steps=500)
```

### `PatchTSTForecaster` for exogenous features with a transformer

Patch-based self-attention. Also accepts exogenous features via `X`:

```python
from yohou_nixtla import PatchTSTForecaster

forecaster = PatchTSTForecaster(input_size=48, max_steps=500)
```

### Other Neural forecasters

| Forecaster | When to reach for it |
|------------|---------------------|
| `MLPForecaster` | Lightweight neural baseline; fast to train |
| `TimesNetForecaster` | Multi-periodicity data; also supports exogenous features |

## If you need exogenous features

Only four forecasters accept external regressors through the `X` parameter:

| Backend | Forecaster |
|---------|-----------|
| Stats | `AutoARIMAForecaster`, `ARIMAForecaster` |
| Neural | `PatchTSTForecaster`, `TimesNetForecaster` |

If you have exogenous features and want a Stats model, use `AutoARIMAForecaster`.
For a Neural model, use `PatchTSTForecaster`.

See [How to Use Exogenous Features](exogenous-features.md) for details.

## Quick reference

| Scenario | Recommended Forecaster |
|----------|----------------------|
| Quick baseline | `SeasonalNaiveForecaster` |
| Monthly data, annual seasonality | `AutoARIMAForecaster(season_length=12)` |
| Clear trend or multiplicative seasonality | `AutoETSForecaster` |
| Sparse or intermittent demand | `CrostonForecaster` |
| Large dataset or complex patterns | `NBEATSForecaster` |
| Long forecast horizon | `NHITSForecaster` |
| Exogenous features (Stats) | `AutoARIMAForecaster` |
| Exogenous features (Neural) | `PatchTSTForecaster` |

!!! tip "Interactive version available"

    The [Comparing Forecasters](../examples/index.md) notebook lets you run
    side-by-side forecaster comparisons interactively.

## See Also

- [Concepts](../explanation/concepts.md): trade-offs between backends
- [How to Use Exogenous Features](exogenous-features.md): pass external regressors
- [API Reference](../reference/api.md): full parameter list for each class
