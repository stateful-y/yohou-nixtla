# How to Integrate with scikit-learn

This guide shows you how to use Yohou-Nixtla forecasters with scikit-learn's
introspection and hyperparameter search utilities.

## Prerequisites

- Yohou-Nixtla installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with scikit-learn's `clone`, `get_params`, and `set_params`

## Clone a Forecaster

All forecasters are compatible with `sklearn.base.clone`. This produces an
unfitted copy with the same hyperparameters - used internally by cross-validation
and search utilities:

```python
from sklearn.base import clone
from yohou_nixtla import AutoARIMAForecaster

original = AutoARIMAForecaster(season_length=12, max_p=5)
cloned = clone(original)
# cloned has the same parameters but is not fitted
```

## Get and Set Parameters

Use `get_params` and `set_params` to inspect and modify parameters
programmatically:

```python
forecaster = AutoARIMAForecaster(season_length=12)

# Inspect all parameters
params = forecaster.get_params()
# {"season_length": 12, "max_p": 5, "freq": None, ...}

# Update a parameter in-place
forecaster.set_params(season_length=4)
```

`get_params(deep=True)` includes nested transformer parameters with the
`<param>__<subparam>` naming convention.

## Hyperparameter Search with Yohou's GridSearchCV

Use Yohou's `GridSearchCV` rather than sklearn's - it wraps the `fit`/`predict`
lifecycle correctly for time series cross-validation:

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
print(search.best_params_)

best_forecaster = search.best_estimator_
```

## Configure Transformer Parameters in Hyperparameter Search

Transformer parameters nested inside a forecaster are accessible via the double
underscore convention:

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

## See Also

- [Concepts](../explanation/concepts.md) - understand the sklearn compatibility design
- [How to Configure Forecasters](configure.md) - all available parameters for each forecaster
- [API Reference](../reference/api.md) - `fit`, `predict`, `get_params`, `set_params` signatures
