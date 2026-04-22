# How to Integrate with Yohou

This guide shows you how to use Yohou-Nixtla forecasters inside Yohou's
ecosystem: pipelines, hyperparameter search, scikit-learn introspection, and
the `observe`/`rewind` lifecycle.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou's `fit`/`predict`/`observe`/`rewind` pattern

## Clone a forecaster

All forecasters are compatible with `sklearn.base.clone`, producing an
unfitted copy with the same hyperparameters:

```python
from sklearn.base import clone
from yohou_nixtla import AutoARIMAForecaster

original = AutoARIMAForecaster(season_length=12, max_p=5)
cloned = clone(original)
# cloned has the same parameters but is not fitted
```

Cross-validation and search utilities rely on this internally.

## Inspect and modify parameters

Use `get_params` and `set_params` to read or change parameters
programmatically:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)

params = forecaster.get_params()
# {"season_length": 12, "max_p": 5, "freq": None, ...}

forecaster.set_params(season_length=4)
```

`get_params(deep=True)` includes nested transformer parameters with the
`<param>__<subparam>` convention, so you can access, for example,
`target_transformer__with_mean`.

## Search hyperparameters with GridSearchCV

Use Yohou's `GridSearchCV` rather than sklearn's. It wraps the time series
`fit`/`predict` lifecycle correctly, including the `forecasting_horizon`
argument:

```python
from yohou.model_selection import GridSearchCV
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster()

param_grid = {
    "season_length": [4, 12],
    "max_p": [2, 5],
    "max_q": [2, 5],
}

search = GridSearchCV(
    forecaster,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
)
search.fit(y, forecasting_horizon=12)

best_forecaster = search.best_estimator_
print(search.best_params_)
```

### Tune nested transformer parameters

Transformer parameters are searchable through the double-underscore convention:

```python
from sklearn.preprocessing import StandardScaler
from yohou.model_selection import GridSearchCV
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(target_transformer=StandardScaler())

param_grid = {
    "season_length": [12],
    "target_transformer__with_mean": [True, False],
}

search = GridSearchCV(forecaster, param_grid, cv=3)
search.fit(y, forecasting_horizon=12)
```

## Use observe, observe_predict, and rewind

Yohou-Nixtla forecasters support Yohou's incremental lifecycle methods for
updating or rolling back the model state as new data arrives:

```python
from yohou_nixtla import AutoARIMAForecaster

forecaster = AutoARIMAForecaster(season_length=12)
forecaster.fit(y_train, forecasting_horizon=12)

# Observe new data without refitting
forecaster.observe(y_new)

# Observe and produce rolling predictions in one call
y_pred = forecaster.observe_predict(y_new)

# Roll back to a previous state
forecaster.rewind(steps=1)
```

`observe` ingests new observations incrementally. `observe_predict` combines
observation with rolling predictions, advancing the window by the forecasting
horizon at each stride. `rewind` rolls the internal state back without
refitting.

## See Also

- [Concepts](../explanation/concepts.md): dual inheritance and sklearn compatibility design
- [API Reference](../reference/api.md): `fit`, `predict`, `get_params`, `set_params` signatures
