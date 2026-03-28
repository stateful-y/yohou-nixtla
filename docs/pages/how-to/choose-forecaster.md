# How to Choose the Right Forecaster

This guide shows you how to select the right Yohou-Nixtla forecaster for your
data and use case.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))

## Step 1: Decide between Stats and Neural

Start with a Stats forecaster unless you have a specific reason not to:

- Stats forecasters train in seconds and require minimal configuration
- Neural forecasters can take minutes to hours and require GPU for large runs

Switch to Neural when:

- Your dataset has more than 5,000 observations per series
- Classical models (AutoARIMA, AutoETS) consistently underfit after tuning
- You need a transformer-based or deep learning model specifically

## Step 2: Choose within Stats Forecasters

### Use `AutoARIMAForecaster` as the default

AutoARIMA selects the best ARIMA order automatically. It is robust and handles
a wide range of data patterns.

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
```

### Use `AutoETSForecaster` for clear trend or multiplicative seasonality

ETS models decompose data into Error, Trend, and Seasonality components.
AutoETS selects the best combination.

```python
from yohou_nixtla import AutoETSForecaster

forecaster = AutoETSForecaster(season_length=12)
```

### Use `NaiveForecaster` or `SeasonalNaiveForecaster` as baselines

Before investing in complex models, establish a baseline that any production
model must beat:

```python
from yohou_nixtla import SeasonalNaiveForecaster

baseline = SeasonalNaiveForecaster(season_length=12)
baseline.fit(y, forecasting_horizon=12)
y_baseline = baseline.predict(forecasting_horizon=12)
```

If a more complex model cannot beat the seasonal naive baseline, revisit your
data quality or feature engineering before trying harder models.

### Use `CrostonForecaster` for intermittent demand

If your series contains many zeros (sparse demand data), standard models
overfit the non-zero spikes. Use Croston instead:

```python
from yohou_nixtla import CrostonForecaster

forecaster = CrostonForecaster()
```

## Step 3: Choose within Neural Forecasters

### Use `NBEATSForecaster` as the default neural model

N-BEATS is a strong general-purpose neural forecaster with no preprocessing
requirements and good interpretability:

```python
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(input_size=24, max_steps=500)
```

### Use `NHITSForecaster` for long horizons

N-HiTS uses hierarchical interpolation and outperforms N-BEATS when the
forecast horizon is long relative to the input window:

```python
from yohou_nixtla import NHITSForecaster

forecaster = NHITSForecaster(input_size=48, max_steps=500)
```

### Use `PatchTSTForecaster` for a transformer-based approach

PatchTST uses patch-based self-attention. It works well on long sequences and
treats each series independently:

```python
from yohou_nixtla import PatchTSTForecaster

forecaster = PatchTSTForecaster(input_size=48, max_steps=500)
```

## Quick Reference

| Scenario | Recommended Forecaster |
|----------|----------------------|
| Quick baseline, then improve | `SeasonalNaiveForecaster` |
| Monthly data, annual seasonality | `AutoARIMAForecaster(season_length=12)` |
| Clear trend or multiplicative seasonality | `AutoETSForecaster` |
| Sparse or intermittent demand | `CrostonForecaster` |
| Large dataset or complex patterns | `NBEATSForecaster` |
| Long forecast horizon | `NHITSForecaster` |
| Transformer-based model | `PatchTSTForecaster` |

## See Also

- [Concepts](../explanation/concepts.md) - understand the trade-offs between backends
- [How to Configure Forecasters](configure.md) - tune your chosen forecaster
- [API Reference](../reference/api.md) - full parameter list for each class
