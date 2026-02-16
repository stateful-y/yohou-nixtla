"""Base class for Nixtla forecaster wrappers.

This module provides ``BaseNixtlaForecaster``, the single shared base class
for all Nixtla backend integrations (statsforecast, neuralforecast).

Nixtla forecasters are **pure algorithm wrappers** -- yohou handles feature
transformation (``feature_transformer``, ``target_as_feature``,
``target_transformer``), panel data (per-group transforms via
``BasePanelForecaster``), and exogenous features (through yohou's pipeline).
The Nixtla backend receives already-transformed data in long format for
efficient batch fit/predict.
"""

from __future__ import annotations

import abc
from typing import Any, Self
from typing import cast as typing_cast

import polars as pl
import polars.selectors as cs
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted
from sklearn_wrap.base import BaseClassWrapper
from yohou.point import BasePointForecaster
from yohou.utils import cast
from yohou.utils.panel import dict_to_panel, inspect_locality, select_panel_columns
from yohou.utils.validate_data import validate_forecaster_data
from yohou.utils.validation import check_panel_group_names

from yohou_nixtla._conversion import infer_freq, nixtla_to_yohou, yohou_to_nixtla

__all__ = ["BaseNixtlaForecaster"]


class BaseNixtlaForecaster(BaseClassWrapper, BasePointForecaster, metaclass=abc.ABCMeta):
    """Abstract base class for all Nixtla library forecaster wrappers.

    Provides the shared fit/predict template that:

    1. Delegates feature transformation and panel data handling to yohou
       (via ``_pre_fit``).
    2. Reassembles transformed data into wide format (via ``dict_to_panel``).
    3. Converts to Nixtla long format (via ``yohou_to_nixtla``).
    4. Calls backend-specific ``_fit_backend`` / ``_predict_backend``.
    5. Converts predictions back (via ``nixtla_to_yohou``) and applies
       inverse target transformation.

    Subclasses (``BaseStatsForecaster``, ``BaseNeuralForecaster``)
    only need to implement two abstract methods:

    - ``_fit_backend``: Create and fit the Nixtla orchestrator.
    - ``_predict_backend``: Generate raw predictions from the orchestrator.

    Parameters
    ----------
    feature_transformer : BaseTransformer or None, default=None
        Transformer applied to exogenous features before fitting/predicting.
    target_transformer : BaseTransformer or None, default=None
        Transformer applied to the target before fitting. Inverse-transformed
        after predicting to return forecasts in the original scale.
    target_as_feature : {"transformed", "raw"} or None, default=None
        Whether to include transformed (or raw) lagged target values as
        additional features.
    freq : str or None, default=None
        Frequency string. Auto-inferred from data if None.
    **params : dict
        Parameters forwarded to the wrapped model constructor via
        ``BaseClassWrapper``.

    Attributes
    ----------
    nixtla_forecaster_ : object
        The fitted Nixtla orchestrator (``StatsForecast`` or
        ``NeuralForecast`` depending on subclass).
    freq_ : str
        The inferred or provided frequency string.
    y_columns_ : list of str
        Original target column names from the training data.

    See Also
    --------
    yohou.point.BasePointForecaster : Base class for point forecasters.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "freq": [str, None],
        "target_as_feature": [StrOptions({"transformed", "raw"}), None],
    }

    _estimator_default_class: type | None = None

    # Fitted attributes
    y_columns_: list[str]

    def __init__(
        self,
        *,
        feature_transformer=None,
        target_transformer=None,
        target_as_feature=None,
        freq: str | None = None,
        **params,
    ):
        BaseClassWrapper.__init__(self, **params)
        BasePointForecaster.__init__(
            self,
            feature_transformer=feature_transformer,
            target_transformer=target_transformer,
            target_as_feature=target_as_feature,
        )
        self.freq = freq

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters of this forecaster.

        Extends ``BaseClassWrapper.get_params`` to include ``freq``
        which is not forwarded to the wrapped model.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        params = super().get_params(deep=deep)
        params["freq"] = self.freq
        params["feature_transformer"] = self.feature_transformer
        params["target_transformer"] = self.target_transformer
        params["target_as_feature"] = self.target_as_feature
        return params

    def set_params(self, **params) -> Self:
        """Set the parameters of this forecaster.

        Extends ``BaseClassWrapper.set_params`` to handle ``freq``
        which is not forwarded to the wrapped model.

        Parameters
        ----------
        **params : dict
            Forecaster parameters.

        Returns
        -------
        self

        """
        for name in ("freq", "feature_transformer", "target_transformer", "target_as_feature"):
            if name in params:
                setattr(self, name, params.pop(name))
        super().set_params(**params)
        return self

    def __sklearn_tags__(self):
        """Get estimator tags for this forecaster.

        Returns
        -------
        Tags
            Estimator tags with ``forecaster_type="point"`` and
            ``uses_reduction=False``.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.forecaster_type = "point"
        tags.forecaster_tags.uses_reduction = False
        tags.forecaster_tags.ignores_exogenous = True
        return tags

    def _validate_pre_fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame | None,
        dict[str, list[str]],
        dict[str, list[str]] | None,
    ]:
        """Validate pre-fit inputs without requiring X for exogenous models.

        Nixtla backends manage their own feature engineering (lags,
        exogenous columns) internally, so ``X`` is always optional
        regardless of ``ignores_exogenous``.  This override skips the
        two ``target_as_feature``/``X`` checks that the base class
        enforces.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None
            Feature time series.
        forecasting_horizon : int
            Number of steps to forecast.

        Returns
        -------
        y : pl.DataFrame
            Validated target.
        X : pl.DataFrame or None
            Validated features.
        y_panel_groups : dict[str, list[str]]
            Panel groups from ``y``.
        X_panel_groups : dict[str, list[str]] or None
            Panel groups from ``X``.

        """
        y, X, _ = validate_forecaster_data(self, y, X, reset=True)
        self.fit_forecasting_horizon_ = forecasting_horizon

        _, y_panel_groups = inspect_locality(y)
        X_panel_groups = None
        if X is not None:
            _, X_panel_groups = inspect_locality(X)
            if len(X_panel_groups) and list(X_panel_groups.keys()) != list(y_panel_groups.keys()):
                raise ValueError("`X` and `y` do not have the same local group names.")

        return y, X, y_panel_groups, X_panel_groups

    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        **params,
    ) -> Self:
        """Fit the Nixtla forecaster to the training data.

        Orchestrates the full pipeline:

        1. Yohou preprocessing (validation, panel detection, transformer
           fitting via ``_pre_fit``).
        2. Reassemble panel dicts to wide format via ``dict_to_panel``.
        3. Convert to Nixtla long format via ``yohou_to_nixtla``.
        4. Delegate to ``_fit_backend`` for backend-specific fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with ``time`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features with ``time`` column.
        forecasting_horizon : int, default=1
            Number of steps to forecast.
        **params : dict
            Additional metadata routing parameters.

        Returns
        -------
        self
            Fitted forecaster.

        Raises
        ------
        ValueError
            If ``forecasting_horizon < 1``.

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be a positive integer, got {forecasting_horizon}.")

        # 1. Yohou preprocessing: validation, panel detection, transformer fitting
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # 2. Reassemble panel dicts back to wide DataFrames
        y_wide = dict_to_panel(y_t)
        X_wide = dict_to_panel(X_t)

        # 3. Store original column names for reconstruction
        assert y_wide is not None, "y_wide must not be None after _pre_fit"
        self.y_columns_ = [c for c in y_wide.columns if c != "time"]

        # 4. Infer frequency
        self.freq_ = self.freq if self.freq is not None else infer_freq(y)

        # 5. Convert to Nixtla long format
        nixtla_df = yohou_to_nixtla(y_wide, X_wide)

        # 6. Delegate to backend
        self._fit_backend(nixtla_df, forecasting_horizon)

        return self

    @abc.abstractmethod
    def _fit_backend(self, nixtla_df: Any, forecasting_horizon: int) -> None:
        """Create and fit the backend-specific Nixtla orchestrator.

        Parameters
        ----------
        nixtla_df : pd.DataFrame
            Training data in Nixtla long format (``unique_id``, ``ds``,
            ``y``, plus optional exogenous columns).
        forecasting_horizon : int
            Number of steps to forecast.

        """

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Overrides yohou's recursive prediction loop -- Nixtla backends
        natively handle multi-step horizons, so the backend is called
        directly with the requested horizon.

        When a ``target_transformer`` was provided at construction,
        predictions are automatically inverse-transformed back to the
        original scale (unless ``predict_transformed=True``).

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Future exogenous features.
        forecasting_horizon : int or None, default=None
            Number of steps to forecast. If None, uses the horizon from fit.
        panel_group_names : list of str or None, default=None
            Panel groups to predict.
        predict_transformed : bool, default=False
            Whether to return transformed predictions.
        **params : dict
            Additional metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Point forecasts with ``observed_time`` and ``time`` columns.

        """
        check_is_fitted(self, ["nixtla_forecaster_", "y_columns_"])

        # Validate and normalize panel_group_names
        panel_group_names = check_panel_group_names(self.panel_group_names_, panel_group_names)

        h = forecasting_horizon if forecasting_horizon is not None else self.fit_forecasting_horizon_

        # Call backend (always predicts all groups for batch efficiency)
        forecast_df = self._predict_backend(h, X)

        # Convert back to yohou format (time + value columns)
        y_pred = nixtla_to_yohou(
            forecast_df.reset_index() if hasattr(forecast_df, "index") else forecast_df,
            y_columns=self.y_columns_,
        )

        # Truncate if backend returned more steps than requested (neural models)
        if len(y_pred) > h:
            y_pred = y_pred.head(h)

        # Apply inverse target transform
        if self.target_transformer is not None and not predict_transformed:
            y_pred = self._inverse_transform_predictions(y_pred)
        else:
            # Cast to original dtypes
            y_pred = self._cast_predictions(y_pred)

        # Drop nixtla time, add yohou time columns (observed_time + time)
        y_pred_no_time = y_pred.drop("time")
        result = self._add_time_columns(y_pred_no_time)

        # Filter to requested panel groups (keeps time columns as globals)
        return select_panel_columns(result, panel_group_names, include_global=True)

    @abc.abstractmethod
    def _predict_backend(self, forecasting_horizon: int, X: pl.DataFrame | None = None) -> Any:
        """Generate raw predictions from the backend orchestrator.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to forecast.
        X : pl.DataFrame or None, default=None
            Future exogenous features (already in yohou format).

        Returns
        -------
        pd.DataFrame
            Raw Nixtla predictions in long format.

        """

    def _inverse_transform_predictions(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Apply inverse target transformation to predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions with ``time`` and value columns in transformed space.

        Returns
        -------
        pl.DataFrame
            Predictions with ``time`` and value columns in original space,
            cast to original dtypes.

        """
        if self.panel_group_names_ is None:
            # Standard data
            transformer = typing_cast(Any, self.target_transformer_)
            y_pred_inv = transformer.inverse_transform(X_t=y_pred, X_p=self._y_observed)

            value_cols = cast(y_pred_inv.select(~cs.by_name("time")), self.local_y_schema_)
            return pl.concat([y_pred_inv.select("time"), value_cols], how="horizontal")

        # Panel data: per-group inverse transform
        inv_parts: list[pl.DataFrame] = []
        for group_name in self.panel_group_names_:
            group_cols = [c for c in y_pred.columns if c.startswith(f"{group_name}__")]
            group_df = y_pred.select(["time"] + group_cols)
            transformer = self.target_transformer_[group_name]
            y_obs = self._y_observed[group_name]
            group_inv = transformer.inverse_transform(X_t=group_df, X_p=y_obs)

            schema = {f"{group_name}__{col}": dtype for col, dtype in self.local_y_schema_.items()}
            inv_parts.append(cast(group_inv.select(~cs.by_name("time")), schema))

        return pl.concat([y_pred.select("time"), *inv_parts], how="horizontal")

    def _cast_predictions(self, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Cast prediction values to original dtypes without inverse transform.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions with ``time`` and value columns.

        Returns
        -------
        pl.DataFrame
            Predictions with value columns cast to original dtypes.

        """
        if self.panel_group_names_ is None:
            value_cols = cast(y_pred.select(~cs.by_name("time")), self.local_y_schema_)
        else:
            schema: dict = {}
            for group_name in self.panel_group_names_:
                schema.update({f"{group_name}__{col}": dtype for col, dtype in self.local_y_schema_.items()})
            value_cols = cast(y_pred.select(~cs.by_name("time")), schema)

        return pl.concat([y_pred.select("time"), value_cols], how="horizontal")

    def _predict_one(self, panel_group_names: list[str] | None = None, **params) -> pl.DataFrame:
        """Generate predictions for the fitted forecasting horizon.

        Parameters
        ----------
        panel_group_names : list of str or None, default=None
            Panel groups to predict.  When not ``None``, the output is
            filtered to the requested groups only.
        **params : dict
            Additional metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with ``observed_time`` and ``time`` columns.

        """
        check_is_fitted(self, ["nixtla_forecaster_", "y_columns_"])

        # Validate and normalize panel_group_names
        panel_group_names = check_panel_group_names(self.panel_group_names_, panel_group_names)

        forecast_df = self._predict_backend(self.fit_forecasting_horizon_)

        y_pred = nixtla_to_yohou(
            forecast_df.reset_index() if hasattr(forecast_df, "index") else forecast_df,
            y_columns=self.y_columns_,
        ).drop("time")

        result = self._add_time_columns(y_pred)

        # Filter to requested panel groups (keeps time columns as globals)
        return select_panel_columns(result, panel_group_names, include_global=True)

    def _convert_nixtla_to_yohou(
        self,
        forecast_df: Any,
        reset_index: bool = True,
    ) -> pl.DataFrame:
        """Convert Nixtla forecast output to yohou format with time columns.

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Raw forecast output from Nixtla orchestrator.
        reset_index : bool, default=True
            Whether to reset pandas index before conversion.

        Returns
        -------
        pl.DataFrame
            Predictions with ``observed_time`` and ``time`` columns.

        """
        check_is_fitted(self, ["y_columns_"])

        if reset_index and hasattr(forecast_df, "index"):
            forecast_df = forecast_df.reset_index()

        y_pred = nixtla_to_yohou(
            forecast_df,
            y_columns=self.y_columns_,
        ).drop("time")

        return self._add_time_columns(y_pred)
