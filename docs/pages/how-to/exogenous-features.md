# How to Use Exogenous Features

This guide shows you how to incorporate external variables (exogenous features)
into your forecasts.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Feature data aligned with your target time series

## Supported forecasters

Only these forecasters accept exogenous features:

| Backend | Forecaster |
|---------|-----------|
| Stats | `AutoARIMAForecaster`, `ARIMAForecaster` |
| Neural | `PatchTSTForecaster`, `TimesNetForecaster` |

Other forecasters (e.g., `AutoETSForecaster`, `NBEATSForecaster`) ignore
exogenous data. If you pass `X` to a forecaster that does not support it,
it will be silently ignored during preprocessing.

## Prepare your feature DataFrames

Create a polars DataFrame with a `time` column and one column per feature.
`X_train` must have the same number of rows as `y_train`:

```python
import polars as pl

X_train = pl.DataFrame({
    "time": y_train["time"],
    "price": [...],
    "promotion": [...],
})
```

## Fit with exogenous features

Pass `X` to `fit`. The features are merged into the Nixtla long-format
training data automatically:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y_train, X=X_train, forecasting_horizon=12)
```

## Predict

Call `predict` with the desired horizon:

```python
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Apply feature preprocessing

If your features have different scales, set a `feature_transformer` to
normalize them before they reach the backend. The transformer is fit on
`X_train` during `fit` and applied consistently at prediction time:

```python
from sklearn.preprocessing import StandardScaler
from yohou_nixtla import PatchTSTForecaster

forecaster = PatchTSTForecaster(
    input_size=24,
    max_steps=500,
    feature_transformer=StandardScaler(),
)
forecaster.fit(y_train, X=X_train, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Panel data with exogenous features

For panel (grouped) data, feature columns can be global or per-group. The
conversion module matches them automatically:

```python
# Per-group features share the group suffix
X_train = pl.DataFrame({
    "time": y_train["time"],
    "price__store_1": [...],
    "price__store_2": [...],
})

# Global features (no __) are broadcast to all groups
X_train = pl.DataFrame({
    "time": y_train["time"],
    "oil_price": [...],
})
```

## See Also

- [Concepts](../explanation/concepts.md): exogenous feature design
- [How to Choose a Forecaster](choose-forecaster.md): which forecasters support exogenous features
- [API Reference](../reference/api.md): `fit` and `predict` method signatures
