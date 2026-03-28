# Troubleshooting

Solutions to common problems when using Yohou-Nixtla.

## Installation Problems

### `ModuleNotFoundError: No module named 'yohou_nixtla'`

You installed into a different environment than the one you are running. Verify both:

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

Neural forecasters require the `neuralforecast` extra. Install it:

```bash
uv add yohou-nixtla[neural]
# or
pip install yohou-nixtla[neural]
```

## Frequency Detection Failures

### `ValueError: unsupported polars interval ...`

The `infer_freq` utility cannot map your DataFrame's time column interval to a
Nixtla frequency alias. Override it manually:

```python
forecaster = AutoARIMAForecaster(season_length=12, freq="MS")
```

See the [Configuration guide](configure.md) for a
full table of frequency aliases.

### Predictions are offset by one period

Your frequency string may be mismatched with your data. Check the interval in
your polars DataFrame:

```python
y.select(
    (pl.col("time").diff().drop_nulls().unique()).alias("intervals")
)
```

Then set `freq` explicitly to match.

## Panel Data Errors

### Predictions have fewer columns than the input

One or more series in your panel data may contain missing values or all-null
periods. Nixtla drops series that cannot be fitted. Inspect your data:

```python
y.null_count()
```

Fill missing values or remove the affected series before fitting.

### Column naming convention not recognized

Panel column names must contain the `__` separator to signal grouping. For
example, `sales__store_1` not `sales_store_1`. Check your column names:

```python
print(y.columns)
```

Rename columns to include `__`:

```python
y = y.rename({"sales_store_1": "sales__store_1"})
```

## Neural Forecaster Training Problems

### Training loss is `NaN` from the first step

Your data contains `NaN`, `inf`, or very large values. Normalize the target
before fitting:

```python
from sklearn.preprocessing import StandardScaler

forecaster = NBEATSForecaster(
    input_size=24,
    max_steps=100,
    target_transformer=StandardScaler(),
)
```

### Model underfits - predictions are flat or repeat the mean

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

Force CPU execution with `accelerator="cpu"`:

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

## Data Format Errors

### `KeyError: 'time'`

Yohou expects a column named exactly `time`. Rename your timestamp column:

```python
y = y.rename({"date": "time"})
```

### `check_is_fitted`: "This forecaster is not fitted yet"

Call `fit` before `predict` or `observe`:

```python
forecaster.fit(y, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Getting More Help

- [Open an issue](https://github.com/stateful-y/yohou-nixtla/issues/new) - include
  your Python version, package version (`yohou_nixtla.__version__`), and a minimal
  reproducible example
- [Start a discussion](https://github.com/stateful-y/yohou-nixtla/discussions) - for
  usage questions and design feedback
