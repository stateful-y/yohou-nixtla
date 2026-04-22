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

Not all forecasters accept external regressors. By default, the base class
tags forecasters as ignoring exogenous features. Only four forecasters override
this: `AutoARIMAForecaster`, `ARIMAForecaster` (Stats), and
`PatchTSTForecaster`, `TimesNetForecaster` (Neural).

The exogenous data flows through the same conversion pipeline as the target.
During `fit`, exogenous columns are merged into the Nixtla long-format
DataFrame alongside the target. Yohou's `feature_transformer` applies
preprocessing (scaling, encoding) before the data reaches the backend.

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
