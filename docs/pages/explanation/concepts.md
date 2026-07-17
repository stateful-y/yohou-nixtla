---
description: How Yohou-Nixtla bridges Yohou and Nixtla, covering the two backends, the wrapping model, and the panel data convention.
---

# About Yohou-Nixtla

Yohou-Nixtla bridges the [Yohou](https://github.com/stateful-y/yohou) time series forecasting framework and the [Nixtla](https://nixtla.io/) ecosystem. It wraps Nixtla's forecasting libraries (**StatsForecast** and **NeuralForecast**) as Yohou-compatible forecasters, so you get Nixtla's state-of-the-art models with Yohou's scikit-learn-compatible API.

## Two forecasting backends

Yohou-Nixtla wraps two Nixtla libraries, each serving a different part of the
forecasting spectrum.

**StatsForecast** provides 10 classical models (AutoARIMA, AutoETS, AutoCES,
AutoTheta, Naive, SeasonalNaive, ARIMA, HoltWinters, Theta, Croston). These
models are fast, interpretable, and require minimal configuration. A
`season_length` parameter is usually all you need. They are a natural starting
point for most time series work.

**NeuralForecast** provides 5 deep learning architectures (NBEATS, NHITS, MLP,
PatchTST, TimesNet). These models trade speed for capacity: they can capture
complex non-linear patterns that classical methods miss, but they need more
data, more configuration (`input_size`, `max_steps`), and benefit from GPU
acceleration.

The two backends are installed independently. If you only need classical
models, you never have to install PyTorch or `neuralforecast`. This keeps the
dependency footprint small for teams that do not need neural forecasters.

!!! example "Interactive Example"
    See [**Model Comparison**](/examples/model_comparison/) ([View](/examples/model_comparison/) | [Editable](/examples/model_comparison/edit/)) for a side-by-side comparison of Stats and Neural forecasters on the Air Passengers dataset.

## How data flows through the system

Yohou and Nixtla use different data formats. Yohou works with polars
wide-format DataFrames (one column per series, shared `time` column). Nixtla
expects pandas long-format DataFrames (one row per observation, with
`unique_id`, `ds`, and `y` columns).

Rather than making users convert between the two, every `fit` and `predict`
call handles the translation transparently:

1. **On fit**: the forecaster receives a polars wide-format DataFrame, converts
   it to Nixtla's long format, fits the backend model, and stores the result.
2. **On predict**: the backend generates long-format predictions, and the
   forecaster pivots them back to polars wide format with the original column
   names.

The trade-off is a small polars-to-pandas overhead per call. In practice this
is negligible compared to model fitting time, and it keeps the user-facing API
free from format concerns.

Frequency inference also happens automatically. The conversion layer reads the
polars `time` column, determines the interval (e.g., `"1mo"`), and maps it to
the pandas offset alias Nixtla expects (e.g., `"MS"`). If the automatic
mapping fails for an unusual frequency, you can set `freq` explicitly.

## Panel data and the `__` convention

Yohou encodes panel (grouped) data in column names using the `__` separator.
A column named `sales__store_1` means "the `sales` feature for the `store_1`
group." This encoding keeps everything in a single flat DataFrame with no
separate index or groupby column.

When Yohou-Nixtla sees `__` in column names, it maps each group to a Nixtla
`unique_id` and fits all groups in a single backend call. Nixtla's batch
fitting is efficient because it processes all series together rather than
looping one at a time. After prediction, the results are pivoted back to the
original column names.

Exogenous feature columns follow the same convention. A feature column
`price__store_1` is matched to its corresponding target `sales__store_1` by
shared group suffix. Feature columns without `__` are treated as global and
broadcast to all groups.

!!! example "Interactive Example"
    See [**Panel Data**](/examples/panel_data/) ([View](/examples/panel_data/) | [Editable](/examples/panel_data/edit/)) for a hands-on walkthrough of multi-series forecasting.

## The dual-inheritance design

Every Yohou-Nixtla forecaster inherits from two parent classes:

- **`BaseClassWrapper`** (from `sklearn_wrap`): provides scikit-learn
  compatibility (`clone`, `get_params`, `set_params`). It manages the wrapped
  Nixtla model instance and forwards `**params` to the backend model
  constructor.
- **`BasePointForecaster`** (from Yohou): provides the
  `fit`/`predict`/`observe`/`rewind` lifecycle, panel data detection, and
  target/feature transformer management.

This dual inheritance means forecasters work natively with both Yohou pipelines
(like `DecompositionPipeline` or `ColumnForecaster`) and scikit-learn
meta-estimators (like `GridSearchCV`). Adding a new Nixtla model is a
three-line class definition: set `_estimator_default_class` and optionally
override `__sklearn_tags__` for exogenous feature support.

## Exogenous feature support

Yohou's `BasePointForecaster` defines three exogenous inputs: `X_actual`
(historical observations only), `X_future` (known for future time steps), and
`X_forecast` (vintage-time forecasts). Nixtla backends only support `X_future`,
which maps to statsforecast's `X_df` and neuralforecast's `futr_exog_list`/`futr_df`.

Not all forecasters accept external regressors. Each forecaster declares a
`supports_exogenous` tag. Passing `X_future` to a forecaster that does not
support it raises a `ValueError` immediately, rather than failing silently
deep in the backend. Seven forecasters currently support exogenous features:
`AutoARIMAForecaster`, `ARIMAForecaster`, `HoltWintersForecaster` (Stats), and
`NHITSForecaster`, `MLPForecaster`, `PatchTSTForecaster`, `TimesNetForecaster`
(Neural).

The integration bypasses yohou's step column derivation entirely. Yohou
normally pivots `X_future` into step columns (`col_step_1`, `col_step_2`, ...),
which makes sense for its recursive prediction loop. Nixtla backends handle
multi-step prediction natively, so they expect raw feature columns to remain
consistent between training and prediction. The fit method passes
`X_future=None` to yohou's `_pre_fit` (preventing step column creation), then
merges the raw `X_future` columns into the Nixtla training DataFrame directly.

At predict time, the forecaster validates that `X_future` contains the same
feature columns that were present during training, converts the polars
DataFrame to Nixtla's expected format, and passes it to the backend's predict
call. The `x_future_to_nixtla` conversion handles both single-series and panel
data layouts automatically.

`X_forecast` is explicitly rejected. Nixtla has no equivalent concept for
vintage-time exogenous features, and silently ignoring this parameter would
hide a configuration error. Passing `X_forecast` to any Nixtla forecaster
raises a `ValueError` with a clear message.

## Limitations

1. **Point forecasts only**: Forecasters are wrapped as `BasePointForecaster`.
   Interval and probabilistic forecasts from Nixtla are not yet exposed.
   However, you can wrap any point forecaster in Yohou's
   `SplitConformalForecaster` to get conformal prediction intervals.

2. **Polars-to-pandas overhead**: Each `fit`/`predict` call converts between
   polars and pandas. For very high-frequency prediction loops, this may add
   latency.

3. **Neural dependencies**: Neural forecasters require PyTorch and
   `neuralforecast`, which are large dependencies. Install them only when
   needed via `pip install yohou-nixtla[neural]`.

4. **Fixed model set**: The current release wraps 15 Nixtla models. Custom
   models require subclassing `BaseStatsForecaster` or
   `BaseNeuralForecaster`.

## See Also

- [Getting Started](../tutorials/getting-started.md): install and run your first forecast
- [How to Choose a Forecaster](../how-to/choose-forecaster.md): pick the right model
- [API Reference](../reference/api.md): detailed API documentation
