# Glossary

Reference terms and definitions for Yohou-Nixtla.

## Base Classes

**`BaseNixtlaForecaster`**
: Abstract base class for all Yohou-Nixtla forecasters. Provides the shared
  `fit`/`predict`/`observe`/`rewind` template and handles data conversion
  between Yohou and Nixtla formats.

**`BaseStatsForecaster`**
: Base class for all `statsforecast` wrappers. Subclasses set
  `_estimator_default_class` to a specific statsforecast model (e.g.,
  `AutoARIMA`).

**`BaseNeuralForecaster`**
: Base class for all `neuralforecast` wrappers. Subclasses set
  `_estimator_default_class` to a specific neuralforecast model (e.g.,
  `NBEATS`).

## Fitted Attributes

**`instance_`**
: The constructed backend model instance (e.g., `AutoARIMA()` or `NBEATS()`).
  Available after `fit`.

**`nixtla_forecaster_`**
: The Nixtla orchestrator wrapping the model instance. Either `StatsForecast`
  or `NeuralForecast`. Available after `fit`.

**`freq_`**
: The inferred or provided frequency string. Set at `fit` time.

**`y_columns_`**
: Original target column names from the training data. Set at `fit` time.

## Lifecycle Methods

**`fit(y, X=None, forecasting_horizon=1)`**
: Train the forecaster on historical data. Must be called before `predict`.

**`predict(forecasting_horizon, X=None)`**
: Generate point forecasts for `forecasting_horizon` steps ahead.

**`observe(y_new)`**
: Append new observations to the forecaster's internal state without
  retraining. Use this when new data arrives in a streaming or online scenario.

**`rewind(y)`**
: Reset the forecaster's internal observation state. Does not retrain the model.

## Data Formats

**Wide format**
: Yohou's native DataFrame format. One row per timestamp, one column per
  variable or series. The `time` column holds timestamps.

**Long format**
: Nixtla's native DataFrame format. One row per (`unique_id`, `ds`) pair,
  with a `y` column for the target value. Used internally by Yohou-Nixtla for
  conversion.

**Panel data**
: Multiple related time series stored together in a single DataFrame. In Yohou,
  identified by the `__` separator in column names (e.g., `sales__store_1`,
  `sales__store_2`).

**`__` (double underscore)**
: Column naming separator in Yohou that signals panel/grouped data. Format:
  `<feature>__<group>`. Both `<feature>` and `<group>` may themselves contain
  single underscores but not `__`.

**`unique_id`**
: Series identifier column in Nixtla's long format. Derived from Yohou's panel
  group identifiers.

**`ds`**
: Datetime column in Nixtla's long format. Corresponds to Yohou's `time` column.

## Frequency Terms

**`freq`**
: Frequency string used by Nixtla (pandas offset aliases, e.g., `"D"`,
  `"MS"`, `"H"`). Yohou-Nixtla auto-infers this from the polars DataFrame's
  `time` column when `freq=None`.

**`infer_freq(y)`**
: Utility function that maps polars interval strings to Nixtla frequency
  aliases. Called automatically at fit time when `freq=None`. See the
  [Configuration guide](../how-to/configure.md) for a full table of
  frequency mappings.

## Parameters

**`season_length`**
: (Stats forecasters) Number of time steps in one seasonal cycle. Set to 12
  for monthly data with annual seasonality, 7 for daily data with weekly
  seasonality, 24 for hourly data with daily seasonality.

**`input_size`**
: (Neural forecasters) Lookback window - how many historical time steps the
  model uses as input for each prediction. Also called the context window.
  Should be at least twice the `forecasting_horizon`.

**`max_steps`**
: (Neural forecasters) Maximum number of gradient-update steps during training.

**`n_jobs`**
: (Stats forecasters) Number of parallel workers for multi-series fitting.
  Set to `-1` to use all available CPU cores.

**`target_transformer`**
: A scikit-learn compatible transformer applied to the target variable before
  fitting and inverse-applied after predicting. Useful for normalizing or
  log-transforming the target.

**`feature_transformer`**
: A scikit-learn compatible transformer applied to exogenous features before
  fitting and predicting.

**`target_as_feature`**
: Whether to include lagged target values as additional exogenous features.
  `"transformed"` uses the target after applying `target_transformer`;
  `"raw"` uses the original scale.

## Model Families

**Stats forecasters**
: Wrappers around `statsforecast` models. Fast, interpretable, suitable for
  classical time series. Includes AutoARIMA, AutoETS, AutoCES, AutoTheta,
  Naive, SeasonalNaive, ARIMA, HoltWinters, Theta, and Croston.

**Neural forecasters**
: Wrappers around `neuralforecast` PyTorch models. Suited for complex patterns
  and large datasets. Includes N-BEATS, N-HiTS, MLP, PatchTST, and TimesNet.
  Require `pip install yohou-nixtla[neural]`.
