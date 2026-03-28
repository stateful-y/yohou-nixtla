# How to Forecast Panel Data

This guide shows you how to forecast multiple time series simultaneously using
Yohou-Nixtla's panel data convention.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- A polars DataFrame with a `time` column

## How Panel Data Works

Yohou uses the `__` separator in column names to signal panel (grouped) data. A
column named `sales__store_1` means: feature `sales`, group `store_1`.

All Yohou-Nixtla forecasters detect this convention automatically. Each group
is fitted independently by Nixtla and the results are recombined into wide
format.

## Steps

### 1. Structure your DataFrame

Create a wide-format polars DataFrame with `time` and one column per group:

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

Column names follow the pattern `<feature>__<group>`. The `time` column holds
timestamps; all other columns are treated as separate series.

### 2. Fit the forecaster

Fit exactly as you would for a single series - the panel structure is handled
automatically:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y, forecasting_horizon=12)
```

To fit all groups in parallel, use `n_jobs=-1`:

```python
forecaster = AutoARIMAForecaster(season_length=12, n_jobs=-1)
forecaster.fit(y, forecasting_horizon=12)
```

### 3. Generate predictions

Call `predict` as normal. The output is a wide-format DataFrame with the same
columns as the input:

```python
y_pred = forecaster.predict(forecasting_horizon=12)
# y_pred columns: ["time", "sales__store_1", "sales__store_2"]
```

### 4. Update with new observations

When new data arrives for all groups, call `observe`:

```python
import polars as pl

y_new = pl.DataFrame({
    "time": [date(2024, 1, 1)],
    "sales__store_1": [148],
    "sales__store_2": [107],
})

forecaster.observe(y_new)
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Handle Unequal Series Lengths

If some groups have missing data at the start, use `null` values. Nixtla fits
each series independently, so shorter series do not affect longer ones:

```python
y = pl.DataFrame({
    "time": dates,
    "sales__store_1": [120, 130, 125, ...],  # full history
    "sales__store_2": [None, None, 88, ...],  # starts later
})
```

Null rows at the beginning of a series are dropped before fitting.

## See Also

- [Concepts](../explanation/concepts.md) - understand how panel data handling works internally
- [How to Configure Forecasters](configure.md) - tune `season_length` and `n_jobs`
- [API Reference](../reference/api.md) - forecaster parameter reference
