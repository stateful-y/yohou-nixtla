# Troubleshooting

Solutions to common problems when using Yohou-Nixtla.

## Installation problems

### `ModuleNotFoundError: No module named 'yohou_nixtla'`

You installed into a different environment than the one you are running.
Verify both point to the same place:

```bash
which python
pip show yohou-nixtla
```

If the paths differ, install into the correct environment:

```bash
uv add yohou-nixtla
# or
pip install yohou-nixtla
```

### `ModuleNotFoundError: No module named 'neuralforecast'`

Neural forecasters require the `neuralforecast` extra:

```bash
uv add yohou-nixtla[neural]
# or
pip install yohou-nixtla[neural]
```

Stats forecasters only need the base install. You do not need both backends.

## Frequency detection

### `ValueError: Cannot map polars interval ...`

The `infer_freq` utility cannot map your DataFrame's time column interval to a
Nixtla frequency alias. Override it manually:

```python
forecaster = AutoARIMAForecaster(season_length=12, freq="MS")
```

See the [API Reference](../reference/api.md) for frequency alias details.

### Predictions are offset by one period

Your frequency string may not match your data's actual interval. Inspect the
interval:

```python
y.select(
    (pl.col("time").diff().drop_nulls().unique()).alias("intervals")
)
```

Then set `freq` explicitly to match.

## Panel data

### Predictions have fewer columns than the input

One or more series may contain all-null values. Nixtla drops series that
cannot be fitted. Check for nulls:

```python
y.null_count()
```

Fill missing values or remove the affected series before fitting.

### Panel structure not detected

Panel column names must contain the `__` separator (double underscore). For
example, `sales__store_1` not `sales_store_1`. Rename columns if needed:

```python
y = y.rename({"sales_store_1": "sales__store_1"})
```

### `ValueError: X and y do not have the same local group names`

Your exogenous feature DataFrame `X` has different group suffixes than `y`.
The group names after the `__` separator must match. Check:

```python
print(y.columns)
print(X.columns)
```

## Data format

### `KeyError: 'time'`

Yohou expects a column named exactly `time`. Rename your timestamp column:

```python
y = y.rename({"date": "time"})
```

### `ValueError: y must have at least one value column besides 'time'`

Your DataFrame has only a `time` column and no value columns. Add at least
one target series column.

### "This forecaster is not fitted yet"

Call `fit` before `predict` or `observe`:

```python
forecaster.fit(y, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Neural forecaster training

### Training loss is `NaN` from the first step

Your data likely contains `NaN`, `inf`, or very large values. Normalize the
target:

```python
from sklearn.preprocessing import StandardScaler

forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=100,
    target_transformer=StandardScaler(),
)
```

### Model underfits (predictions are flat or repeat the mean)

Increase `max_steps` and lower the learning rate:

```python
forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=1000,
    learning_rate=1e-4,
)
```

Also verify `input_size` is at least twice the `forecasting_horizon`.

### `CUDA out of memory`

Force CPU execution:

```python
forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=100,
    accelerator="cpu",
)
```

Or reduce `batch_size`:

```python
forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=100,
    batch_size=16,
)
```

## Getting more help

- [Open an issue](https://github.com/stateful-y/yohou-nixtla/issues/new): include
  your Python version, package version (`yohou_nixtla.__version__`), and a minimal
  reproducible example
- [Start a discussion](https://github.com/stateful-y/yohou-nixtla/discussions): for
  usage questions and design feedback
