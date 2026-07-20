---
description: Feed external variables into a forecast with X_actual, X_future, and X_forecast.
---

# How to Use Exogenous Features

This guide shows you how to incorporate external variables (exogenous features)
into your forecasts. Yohou-Nixtla supports two exogenous inputs:

- **`X_actual`**: observation features available at training time. These go
  through yohou's feature transformation pipeline (lags, rolling statistics)
  before reaching the Nixtla backend.
- **`X_future`**: known future features passed directly to the Nixtla backend
  at both fit and predict time, bypassing yohou's feature engineering.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Feature data aligned with your target time series

## Supported forecasters

Only certain forecasters accept exogenous features. Passing `X_future` to an
unsupported forecaster raises a `ValueError`.

| Backend | Forecaster | Exogenous support |
|---------|-----------|-------------------|
| Stats | `AutoARIMAForecaster` | Required |
| Stats | `ARIMAForecaster` | Required |
| Stats | `HoltWintersForecaster` | Optional |
| Neural | `NHITSForecaster` | Optional |
| Neural | `MLPForecaster` | Optional |
| Neural | `PatchTSTForecaster` | Required |
| Neural | `TimesNetForecaster` | Required |

Models marked "Required" always expect exogenous features at both fit and
predict time. Models marked "Optional" work with or without them.

## Use `X_actual` for observation features

`X_actual` provides historical observation features (values known only up to
the present). These flow through yohou's `actual_transformer` pipeline, so
lags, rolling statistics, and other engineered features are computed
automatically.

Create a polars DataFrame with a `time` column matching `y`:

```python
import polars as pl

X_actual = pl.DataFrame({
    "time": y_train["time"],
    "temperature": [...],
    "humidity": [...],
})
```

Pass it to `fit`:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y_train, X_actual=X_actual, forecasting_horizon=12)
y_pred = forecaster.predict()
```

`X_actual` does not require a `supports_exogenous` tag on the forecaster,
since it is processed by yohou's general feature pipeline.

## Use `X_future` for known future features

`X_future` provides features whose values are known in advance for the
forecast horizon (for example, planned promotions, holidays, or scheduled
prices). These bypass yohou's feature engineering and are passed directly
to the Nixtla backend.

### Prepare your feature DataFrames

Create a polars DataFrame with a `time` column and one column per feature.
The training DataFrame must cover the same time range as `y`:

```python
import polars as pl

X_future_train = pl.DataFrame({
    "time": y_train["time"],
    "price": [...],
    "promotion": [...],
})
```

For prediction, create a DataFrame covering the forecast horizon:

```python
X_future_predict = pl.DataFrame({
    "time": future_dates,
    "price": [...],
    "promotion": [...],
})
```

The predict DataFrame must contain the same feature columns as the training
DataFrame.

### Fit and predict with `X_future`

Pass `X_future` to both `fit` and `predict`:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y_train, X_future=X_future_train, forecasting_horizon=12)
y_pred = forecaster.predict(X_future=X_future_predict)
```

The features are merged into the Nixtla long-format DataFrame automatically.
At predict time, the conversion layer validates that `X_future` contains the
same columns that were present during training.

### Panel data with `X_future`

For panel (grouped) data, feature columns can be per-group or global. The
conversion module matches them automatically:

```python
# Per-group features share the group suffix
X_future_train = pl.DataFrame({
    "time": y_train["time"],
    "price__store_1": [...],
    "price__store_2": [...],
})

# Global features (no __) are broadcast to all groups
X_future_train = pl.DataFrame({
    "time": y_train["time"],
    "oil_price": [...],
})
```

### Neural forecasters

Neural forecasters (NHITS, MLP, PatchTST, TimesNet) receive the exogenous
columns through neuralforecast's `futr_exog_list` parameter. This parameter is
injected automatically during `fit` based on the columns in `X_future`:

```python
from yohou_nixtla.neural import NHITSForecaster

forecaster = NHITSForecaster(input_size=24, max_steps=500)
forecaster.fit(y_train, X_future=X_future_train, forecasting_horizon=12)
y_pred = forecaster.predict(X_future=X_future_predict)
```

## Combine `X_actual` and `X_future`

You can use both inputs together. `X_actual` feeds observation features into
yohou's feature pipeline, while `X_future` passes known future values directly
to the Nixtla backend:

```python
forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(
    y_train,
    X_actual=X_actual,
    X_future=X_future_train,
    forecasting_horizon=12,
)
y_pred = forecaster.predict(X_future=X_future_predict)
```

## Unsupported: `X_forecast`

Nixtla backends do not support yohou's `X_forecast` parameter (vintage-time
exogenous features). Passing `X_forecast` to any Nixtla forecaster raises a
`ValueError`.

## See Also

- [Concepts](../explanation/concepts.md): exogenous feature integration design
- [How to Choose a Forecaster](choose-forecaster.md): which forecasters support exogenous features
- [API Reference](../reference/api.md): `fit` and `predict` method signatures
