# Getting Started

This guide will help you install and start using Yohou-Nixtla in minutes.

## Installation

### Step 1: Install the package

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install yohou-nixtla
    ```

=== "uv"

    ```bash
    uv add yohou-nixtla
    ```

=== "conda"

    ```bash
    conda install -c conda-forge yohou-nixtla
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge yohou-nixtla
    ```

> **Note**: For conda/mamba, ensure the package is published to conda-forge first.

### Step 2: Verify installation

```python
import yohou_nixtla
print(yohou_nixtla.__version__)
```

## Basic Usage

### Step 1: Fit a statistical forecaster

```python
from yohou.datasets import load_air_passengers
from yohou_nixtla import AutoARIMAForecaster

# Load data: polars DataFrame with "time" and "passengers" columns
y = load_air_passengers()

# Create and fit an AutoARIMA forecaster
forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y, forecasting_horizon=12)
```

### Step 2: Generate predictions

```python
# Predict the next 12 months
y_pred = forecaster.predict(forecasting_horizon=12)
print(y_pred)
```

### Step 3: Update with new observations

```python
# When new data arrives, update without refitting
y_new = y.tail(3)
forecaster.observe(y_new)

# Predict again from the updated state
y_pred = forecaster.predict(forecasting_horizon=12)
```

## Complete Example

Here's a complete working example comparing statistical and neural forecasters:

```python
from yohou.datasets import load_air_passengers
from yohou_nixtla import AutoARIMAForecaster, NBEATSForecaster

# Load data
y = load_air_passengers()

# Split into train/test
y_train = y.head(len(y) - 12)

# Statistical Forecaster
arima = AutoARIMAForecaster(season_length=12)
arima.fit(y_train, forecasting_horizon=12)
y_pred_arima = arima.predict(forecasting_horizon=12)

# Neural Forecaster
nbeats = NBEATSForecaster(input_size=24, max_steps=100)
nbeats.fit(y_train, forecasting_horizon=12)
y_pred_nbeats = nbeats.predict(forecasting_horizon=12)

print("AutoARIMA predictions:", y_pred_arima)
print("NBEATS predictions:", y_pred_nbeats)
```

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser
- Experiment with different parameters
- See visual outputs in real-time
- Download standalone HTML versions

Or run locally:

=== "just"

    ```bash
    just example model_comparison.py
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/model_comparison.py
    ```

## Next Steps

Now that you have Yohou-Nixtla installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand the forecasting backends, data conversion, and panel data support
- **Explore examples**: Check out the [Examples](examples.md) for interactive demonstrations
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation on all forecaster classes
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/yohou-nixtla/discussions) or [open an issue](https://github.com/stateful-y/yohou-nixtla/issues)
