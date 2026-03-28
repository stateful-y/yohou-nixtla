# How to Use Exogenous Features

This guide shows you how to incorporate external variables (exogenous features)
into your forecasts.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Feature data aligned with your target time series

## Which Forecasters Support Exogenous Features

Only **neural forecasters** use exogenous features at prediction time. Stats
forecasters accept `X` in `fit` but do not use it during prediction.

| Forecaster type | `X` in `fit` | `X` in `predict` |
|-----------------|--------------|------------------|
| Stats (AutoARIMA, AutoETS, etc.) | Accepted (ignored) | Not used |
| Neural (NBEATS, NHITS, MLP, etc.) | Used | Required |

## Steps

### 1. Prepare your feature DataFrames

Create a polars DataFrame with a `time` column and one column per feature. The
`time` values must cover both the training period and the forecast horizon since
neural forecasters require future feature values at prediction time:

```python
import polars as pl

# Training features - same timesteps as y_train
X_train = pl.DataFrame({
    "time": y_train["time"],
    "price": [...],
    "promotion": [...],
})

# Forecast-horizon features - must be known in advance
X_test = pl.DataFrame({
    "time": forecast_dates,
    "price": [...],      # planned prices
    "promotion": [...],  # planned promotions
})
```

`X_train` must have the same number of rows as `y_train`. `X_test` must have
exactly `forecasting_horizon` rows.

### 2. Fit with exogenous features

Pass `X` to `fit`:

```python
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(input_size=24, max_steps=500)
forecaster.fit(y_train, X=X_train, forecasting_horizon=12)
```

### 3. Predict with future features

Pass the future-horizon feature DataFrame to `predict`:

```python
y_pred = forecaster.predict(forecasting_horizon=12, X=X_test)
```

### 4. Apply feature preprocessing

If your features have different scales, apply a transformer:

```python
from sklearn.preprocessing import StandardScaler
from yohou_nixtla import NBEATSForecaster

forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=500,
    feature_transformer=StandardScaler(),
)
forecaster.fit(y_train, X=X_train, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12, X=X_test)
```

The `feature_transformer` is fit on training features and applied consistently
at prediction time.

## See Also

- [Concepts](../explanation/concepts.md) - understand the exogenous feature design
- [How to Configure Forecasters](configure.md) - configure `feature_transformer` and `target_as_feature`
- [API Reference](../reference/api.md) - `fit` and `predict` method signatures
