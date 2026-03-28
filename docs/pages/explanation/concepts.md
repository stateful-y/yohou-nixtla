# About Yohou-Nixtla

Yohou-Nixtla bridges the [Yohou](https://github.com/stateful-y/yohou) time series forecasting framework and the [Nixtla](https://nixtla.io/) ecosystem. It wraps Nixtla's forecasting libraries (**StatsForecast** and **NeuralForecast**) as Yohou-compatible forecasters, so you get Nixtla's state-of-the-art models with Yohou's scikit-learn-compatible API.

## Core Concepts

### Two Forecasting Backends

Yohou-Nixtla provides forecasters from two Nixtla backends:

| Backend | Library | Forecasters | Best For |
|---------|---------|-------------|----------|
| **Stats** | `statsforecast` | AutoARIMA, AutoETS, AutoCES, AutoTheta, Naive, SeasonalNaive, ARIMA, HoltWinters, Theta, Croston | Classical time series with known seasonal patterns |
| **Neural** | `neuralforecast` | NBEATS, NHITS, MLP, PatchTST, TimesNet | Complex patterns, large datasets, deep learning |

**Stats forecasters** are fast and interpretable. They require only target data (`y`) and handle seasonality through `season_length`.

**Neural forecasters** use PyTorch models and are best suited for large datasets or complex non-linear patterns. They require `input_size` and `max_steps` configuration.

!!! example "Interactive Example"
    See [**Model Comparison**](/examples/model_comparison/) ([View](/examples/model_comparison/) | [Editable](/examples/model_comparison/edit/)) for a side-by-side comparison of Stats and Neural forecasters on the Air Passengers dataset.

### Data Conversion

All data conversion between Yohou and Nixtla formats is handled automatically by the `_conversion` module:

- **`yohou_to_nixtla(y, X)`**: Converts a polars wide-format DataFrame to Nixtla's pandas long-format (`unique_id` / `ds` / `y` columns).
- **`nixtla_to_yohou(forecast_df, y_columns)`**: Converts Nixtla's long-format predictions back to polars wide-format.
- **`infer_freq(y)`**: Maps polars interval strings (e.g., `"1d"`, `"1mo"`) to pandas offset aliases (e.g., `"D"`, `"MS"`).

You never need to call these functions directly; they're used internally by the forecasters.

### Panel Data

Yohou-Nixtla supports panel (grouped) time series through Yohou's `__` column naming convention:

```python
import polars as pl

# Panel data: sales for multiple stores
y = pl.DataFrame({
    "time": dates,
    "sales__store_1": [...],
    "sales__store_2": [...],
    "sales__store_3": [...],
})

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y, forecasting_horizon=12)
y_pred = forecaster.predict(forecasting_horizon=12)
# y_pred has columns: "time", "sales__store_1", "sales__store_2", "sales__store_3"
```

Each group (e.g., `store_1`, `store_2`) is modeled independently by Nixtla, and the results are recombined into Yohou's wide format automatically.

!!! example "Interactive Example"
    See [**Panel Data**](/examples/panel_data/) ([View](/examples/panel_data/) | [Editable](/examples/panel_data/edit/)) for a hands-on walkthrough of multi-series forecasting with the `__` convention.

## Design Decisions

### Dual Inheritance

Every forecaster inherits from both `BaseClassWrapper` (for seamless sklearn `clone`/`get_params`/`set_params` support) and `BasePointForecaster` (for Yohou's `fit`/`predict`/`observe`/`rewind` lifecycle). This means forecasters work natively with both Yohou pipelines and sklearn meta-estimators like `GridSearchCV`.

### Automatic Data Conversion

Yohou uses polars wide-format DataFrames; Nixtla expects pandas long-format. Rather than forcing users to convert manually, every `fit` and `predict` call converts data transparently through the `_conversion` module. The trade-off is a small polars-to-pandas overhead per call, but it keeps the user-facing API clean.

### The `__` Convention for Panel Data

Yohou encodes panel structure in column names using the `__` separator (e.g., `sales__store_1`). This avoids needing a separate index or groupby column and keeps everything in a single flat DataFrame. Yohou-Nixtla maps these groups to Nixtla's `unique_id` column automatically.

## Limitations and Considerations

1. **Point forecasts only**: Yohou-Nixtla currently wraps forecasters as `BasePointForecaster`. Interval/probabilistic forecasts from Nixtla are not yet exposed.

2. **Polars ↔ pandas overhead**: Each `fit`/`predict` call converts data between polars and pandas. For very high-frequency prediction loops, this may add latency.

3. **Neural forecaster dependencies**: Neural forecasters require PyTorch and `neuralforecast`, which are large dependencies. Install them separately if not needed.

4. **No custom model wrapping**: The current release provides a fixed set of 20 wrapped models. Custom Nixtla models require subclassing a base forecaster.

## See Also

- [Getting Started](../tutorials/getting-started.md) - install and first example
- [Examples](../tutorials/examples.md) - interactive example notebooks
- [API Reference](../reference/api.md) - detailed API documentation
