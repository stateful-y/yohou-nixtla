# How to Forecast Panel Data

This guide shows you how to forecast multiple time series simultaneously using
Yohou-Nixtla's panel data convention.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- A polars DataFrame with a `time` column

Yohou uses the `__` separator in column names to signal panel (grouped) data
(e.g., `sales__store_1`). All Yohou-Nixtla forecasters detect this convention
automatically. See [Concepts](../explanation/concepts.md#panel-data) for details.

## Structure your DataFrame

Create a wide-format polars DataFrame with `time` and one column per group.
Column names follow the pattern `<feature>__<group>`:

```python
import polars as pl
from datetime import date

dates = pl.date_range(
    date(2022, 1, 1),
    date(2023, 12, 31),
    interval="1mo",
    eager=True,
)

y = pl.DataFrame({
    "time": dates,
    "sales__store_1": [120, 130, 125, 140, 135, 145, 150, 155, 160, 155,
                       145, 140, 125, 135, 130, 145, 140, 150, 155, 160,
                       165, 160, 150, 145],
    "sales__store_2": [90, 95, 88, 102, 98, 105, 108, 112, 115, 110,
                       100, 96, 92, 97, 93, 108, 104, 111, 115, 118,
                       122, 117, 108, 103],
})
```

The `time` column holds timestamps; all other columns are treated as separate
series. Each group is modeled independently by Nixtla and recombined into the
wide format automatically.

## Fit and predict

Fit exactly as you would for a single series. The panel structure is detected
from the column names:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y, forecasting_horizon=12)

y_pred = forecaster.predict(forecasting_horizon=12)
# y_pred columns: ["observed_time", "time", "sales__store_1", "sales__store_2"]
```

To fit all groups in parallel, use `n_jobs=-1`:

```python
forecaster = AutoARIMAForecaster(season_length=12, n_jobs=-1)
```

## Predict specific groups

If you only need forecasts for a subset of groups, pass `panel_group_names`:

```python
y_pred = forecaster.predict(
    forecasting_horizon=12,
    panel_group_names=["store_1"],
)
# y_pred columns: ["observed_time", "time", "sales__store_1"]
```

This filters the output without refitting.

## Handle unequal series lengths

If some groups start later, use `null` values for the missing periods. Nixtla
fits each series independently, so shorter series do not affect longer ones:

```python
y = pl.DataFrame({
    "time": dates,
    "sales__store_1": [120, 130, 125, ...],  # full history
    "sales__store_2": [None, None, 88, ...],  # starts later
})
```

Null rows at the beginning of a series are dropped before fitting.

## Add exogenous features to panel data

When combining panel data with exogenous features, feature columns can be
per-group or global:

```python
# Per-group features share the group suffix
X = pl.DataFrame({
    "time": y["time"],
    "price__store_1": [...],
    "price__store_2": [...],
})

# Global features (no __) are broadcast to all groups
X = pl.DataFrame({
    "time": y["time"],
    "oil_price": [...],
})
```

Only forecasters that accept exogenous features can use this (see
[How to Use Exogenous Features](exogenous-features.md)).

!!! tip "Interactive version available"

    The [Forecasting Panel Data](../tutorials/examples.md) notebook lets you
    experiment with panel data forecasting interactively.

## See Also

- [Concepts](../explanation/concepts.md): how panel data handling works internally
- [How to Use Exogenous Features](exogenous-features.md): external regressors with panel data
- [API Reference](../reference/api.md): forecaster parameter reference
