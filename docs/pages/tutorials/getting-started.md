# Getting Started

In this tutorial, we will install Yohou-Nixtla, fit a forecaster to the classic
Air Passengers dataset, and generate a 12-month forecast.

## Install Yohou-Nixtla

=== "pip"

    ```bash
    pip install yohou-nixtla
    ```

=== "uv"

    ```bash
    uv add yohou-nixtla
    ```

Verify the installation:

```python
import yohou_nixtla
print(yohou_nixtla.__version__)
```

The output should look something like:

```text
0.1.0
```

## Load the data

We will use the Air Passengers dataset, a monthly time series of international
airline passengers from 1949 to 1960. Yohou ships it as a built-in dataset:

```python
from yohou.datasets import load_air_passengers

y = load_air_passengers()
print(y.head(3))
```

```text
shape: (3, 2)
┌─────────────────────┬────────────┐
│ time                ┆ Passengers │
│ ---                 ┆ ---        │
│ datetime[μs]        ┆ i64        │
╞═════════════════════╪════════════╡
│ 1949-01-01 00:00:00 ┆ 112        │
│ 1949-02-01 00:00:00 ┆ 118        │
│ 1949-03-01 00:00:00 ┆ 132        │
└─────────────────────┴────────────┘
```

Notice the two columns: `time` holds timestamps, and `Passengers` holds the
values. This is Yohou's standard wide-format layout.

Now split the data, keeping the last 12 months as a holdout:

```python
y_train = y.head(len(y) - 12)
y_test = y.tail(12)
```

## Fit a forecaster

Create an `AutoARIMAForecaster` and fit it on the training data.
`season_length=12` tells the model to look for annual seasonality in monthly
data:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y_train, forecasting_horizon=12)
```

## Generate predictions

Predict the next 12 months:

```python
y_pred = forecaster.predict(forecasting_horizon=12)
print(y_pred.head(3))
```

```text
shape: (3, 3)
┌─────────────────────┬─────────────────────┬────────────┐
│ observed_time       ┆ time                ┆ Passengers │
│ ---                 ┆ ---                 ┆ ---        │
│ datetime[μs]        ┆ datetime[μs]        ┆ i64        │
╞═════════════════════╪═════════════════════╪════════════╡
│ 1959-12-01 00:00:00 ┆ 1960-01-01 00:00:00 ┆ 424        │
│ 1959-12-01 00:00:00 ┆ 1960-02-01 00:00:00 ┆ 407        │
│ 1959-12-01 00:00:00 ┆ 1960-03-01 00:00:00 ┆ 471        │
└─────────────────────┴─────────────────────┘────────────┘
```

Notice the output has three columns: `observed_time` (the last training
timestamp), `time` (the forecast timestamps), and `Passengers` (the predicted
values).

## What we built

You have installed Yohou-Nixtla, loaded a dataset, fit an AutoARIMA
forecaster, and generated a 12-month forecast. The same `fit`/`predict`
pattern works for all 15 forecasters in the package.

## Next steps

- [Examples](../examples/index.md): compare statistical and neural forecasters
  interactively
- [How to Choose a Forecaster](../how-to/choose-forecaster.md): pick the
  right model for your data
- [Concepts](../explanation/concepts.md): understand the forecasting backends
  and data conversion
