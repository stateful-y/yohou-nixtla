"""Data conversion utilities between yohou polars wide format and Nixtla pandas long format.

This module provides functions to convert between yohou's polars wide-format
DataFrames (with a ``time`` column and value columns) and Nixtla's pandas
long-format DataFrames (with ``unique_id``, ``ds``, and ``y`` columns).

Functions
---------
infer_freq
    Infer a Nixtla-compatible frequency string from a yohou polars DataFrame.
yohou_to_nixtla
    Convert yohou polars wide-format to Nixtla pandas long-format.
nixtla_to_yohou
    Convert Nixtla pandas long-format predictions back to yohou polars wide-format.
"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
from yohou.utils.panel import inspect_locality
from yohou.utils.validation import check_interval_consistency

__all__ = ["infer_freq", "yohou_to_nixtla", "nixtla_to_yohou"]

# Mapping from polars interval strings to pandas offset aliases.
# Covers the most common time series frequencies.
_POLARS_TO_PANDAS_FREQ: dict[str, str] = {
    "1s": "s",
    "1m": "min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "D",
    "1w": "W",
    "1mo": "MS",
    "2mo": "2MS",
    "3mo": "QS",
    "6mo": "6MS",
    "1q": "QS",
    "1y": "YS",
}


def infer_freq(y: pl.DataFrame) -> str:
    """Infer a Nixtla-compatible frequency string from a yohou polars DataFrame.

    Examines the ``time`` column to determine the regular interval between
    consecutive time steps, then converts it to a pandas offset alias
    compatible with Nixtla's libraries.

    Parameters
    ----------
    y : pl.DataFrame
        DataFrame with a ``time`` column of datetime type. Must have at least
        2 rows to infer an interval.

    Returns
    -------
    str
        A pandas offset alias string (e.g., ``"D"``, ``"h"``, ``"MS"``).

    Raises
    ------
    ValueError
        If the DataFrame has fewer than 2 rows, if the time intervals are
        inconsistent, or if the interval cannot be mapped to a known
        pandas offset alias.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1),
    ...         end=datetime(2020, 1, 5),
    ...         interval="1d",
    ...         eager=True,
    ...     ),
    ...     "value": [1, 2, 3, 4, 5],
    ... })
    >>> infer_freq(y)
    'D'

    """
    interval_str = check_interval_consistency(y)

    if interval_str in _POLARS_TO_PANDAS_FREQ:
        return _POLARS_TO_PANDAS_FREQ[interval_str]

    # Attempt to parse as a timedelta and match
    raise ValueError(
        f"Cannot map polars interval '{interval_str}' to a Nixtla-compatible "
        f"pandas offset alias. Known mappings: {list(_POLARS_TO_PANDAS_FREQ.keys())}"
    )


def yohou_to_nixtla(
    y: pl.DataFrame,
    X: pl.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert yohou polars wide-format DataFrames to Nixtla pandas long-format.

    Transforms yohou's wide format (one column per time series, shared ``time``
    column) into Nixtla's long format (``unique_id`` / ``ds`` / ``y`` columns).

    For non-panel data, each value column becomes a separate ``unique_id``.
    For panel data (columns with ``__`` separator), each prefixed column
    becomes a separate ``unique_id``.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series in yohou wide format. Must have a ``time`` column
        and one or more value columns.
    X : pl.DataFrame or None, default=None
        Exogenous features in yohou wide format with matching ``time`` column.
        If provided, exogenous columns are joined into the long-format output.

    Returns
    -------
    pd.DataFrame
        Nixtla long-format DataFrame with columns ``unique_id``, ``ds``, ``y``,
        and any exogenous feature columns.

    Raises
    ------
    ValueError
        If ``y`` has no value columns (only ``time``).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> y = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value": [10.0, 20.0],
    ... })
    >>> df = yohou_to_nixtla(y)
    >>> list(df.columns)
    ['unique_id', 'ds', 'y']
    >>> len(df)
    2

    """
    value_cols = [c for c in y.columns if c != "time"]
    if not value_cols:
        raise ValueError("y must have at least one value column besides 'time'.")

    # Build long-format rows: each value column becomes a unique_id
    # (works identically for both panel and global data)
    records: list[pd.DataFrame] = []
    for col_name in value_cols:
        series_df = y.select(["time", col_name]).to_pandas()
        series_df = series_df.rename(columns={"time": "ds", col_name: "y"})
        series_df["unique_id"] = col_name
        records.append(series_df)

    long_df = pd.concat(records, ignore_index=True)

    # Reorder columns: unique_id first
    long_df = long_df[["unique_id", "ds", "y"]]

    # Add exogenous features if provided
    if X is not None:
        long_df = _add_exogenous(long_df, X, value_cols)

    return long_df


def _add_exogenous(
    long_df: pd.DataFrame,
    X: pl.DataFrame,
    _y_value_cols: list[str],
) -> pd.DataFrame:
    """Add exogenous feature columns to the Nixtla long-format DataFrame.

    Parameters
    ----------
    long_df : pd.DataFrame
        Nixtla long-format DataFrame with ``unique_id``, ``ds``, ``y``.
    X : pl.DataFrame
        Exogenous features in yohou wide format.
    _y_value_cols : list of str
        Value column names from y, used to align panel structure.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with exogenous columns added.

    """
    x_cols = [c for c in X.columns if c != "time"]
    if not x_cols:
        return long_df

    _, x_panel_groups = inspect_locality(X)

    if x_panel_groups:
        # Panel exogenous: match by entity prefix, with suffix fallback
        # Two naming conventions exist:
        #   1. <entity>__<variable> (conftest): "group_0__y_0", "group_0__X_0"
        #      → match by prefix "group_0"
        #   2. <variable>__<entity> (standard yohou panel): "target__store1", "feature__store1"
        #      → match by suffix "store1"
        x_long_records: list[pd.DataFrame] = []
        for uid in long_df["unique_id"].unique():
            uid_mask = long_df["unique_id"] == uid
            uid_df = long_df.loc[uid_mask].copy()

            if "__" in uid:
                prefix = uid.split("__", 1)[0]
                suffix = uid.split("__", 1)[1]

                # Strategy 1: prefix matching (e.g., "group_0__*" for uid "group_0__y_0")
                matching_x_cols = [c for c in x_cols if c.startswith(f"{prefix}__")]

                if matching_x_cols:
                    for xc in matching_x_cols:
                        # Use variable part (after __) as feature column name
                        x_suffix = xc.split("__", 1)[1]
                        x_series = X.select(["time", xc]).to_pandas()
                        uid_df = uid_df.merge(
                            x_series.rename(columns={"time": "ds", xc: x_suffix}),
                            on="ds",
                            how="left",
                        )
                else:
                    # Strategy 2: suffix matching (e.g., "*__store1" for uid "target__store1")
                    matching_x_cols = [c for c in x_cols if c.endswith(f"__{suffix}")]
                    for xc in matching_x_cols:
                        # Use variable part (before __) as feature column name
                        x_prefix = xc.split("__", 1)[0]
                        x_series = X.select(["time", xc]).to_pandas()
                        uid_df = uid_df.merge(
                            x_series.rename(columns={"time": "ds", xc: x_prefix}),
                            on="ds",
                            how="left",
                        )
            x_long_records.append(uid_df)
        return pd.concat(x_long_records, ignore_index=True)
    else:
        # Global exogenous: same features for all unique_ids
        x_pandas = X.to_pandas().rename(columns={"time": "ds"})
        return long_df.merge(x_pandas, on="ds", how="left")


def nixtla_to_yohou(
    forecast_df: pd.DataFrame,
    y_columns: list[str],
    observed_time: pl.Series | None = None,
) -> pl.DataFrame:
    """Convert Nixtla pandas long-format predictions to yohou polars wide-format.

    Reconstructs yohou's wide format from Nixtla's prediction output. Each
    ``unique_id`` becomes a column in the output DataFrame, with ``time``
    and optionally ``observed_time`` columns added.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Nixtla long-format prediction DataFrame with ``unique_id``, ``ds``,
        and one or more model prediction columns.
    y_columns : list of str
        Original yohou target column names, used to map ``unique_id`` values
        back to column names and determine column order.
    observed_time : pl.Series or None, default=None
        The ``observed_time`` value to prepend to the output. If None,
        no ``observed_time`` column is added.

    Returns
    -------
    pl.DataFrame
        Yohou wide-format DataFrame with ``time`` column and one column per
        target series. If ``observed_time`` is provided, an ``observed_time``
        column is included.

    Raises
    ------
    ValueError
        If ``forecast_df`` is empty or has no prediction columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> forecast = pd.DataFrame({
    ...     "unique_id": ["value", "value"],
    ...     "ds": [datetime(2020, 1, 6), datetime(2020, 1, 7)],
    ...     "model": [25.0, 30.0],
    ... })
    >>> result = nixtla_to_yohou(forecast, y_columns=["value"])
    >>> list(result.columns)
    ['time', 'value']
    >>> len(result)
    2

    """
    if forecast_df.empty:
        raise ValueError("forecast_df is empty.")

    # Drop spurious columns that may arise from reset_index() on a RangeIndex
    _ignore = {"unique_id", "ds", "index", "level_0"}

    # Identify prediction column(s)  --  exclude metadata columns
    pred_cols = [c for c in forecast_df.columns if c not in _ignore]
    if not pred_cols:
        raise ValueError("forecast_df has no prediction columns (only 'unique_id' and 'ds').")

    # Use the first prediction column as the values
    pred_col = pred_cols[0]

    unique_ids = forecast_df["unique_id"].unique()

    # Pivot: each unique_id becomes a column
    pivot_parts: dict[str, pl.Series] = {}
    time_series: pl.Series | None = None

    for uid in unique_ids:
        uid_data = forecast_df[forecast_df["unique_id"] == uid].sort_values("ds")
        if time_series is None:
            time_series = pl.Series("time", uid_data["ds"].values).cast(pl.Datetime("us"))
        pivot_parts[uid] = pl.Series(uid, uid_data[pred_col].values)

    if time_series is None:  # pragma: no cover
        raise ValueError("No data found in forecast_df.")

    # Build output DataFrame with columns in original order
    result_cols: dict[str, pl.Series] = {"time": time_series}

    for col_name in y_columns:
        if col_name in pivot_parts:
            result_cols[col_name] = pivot_parts[col_name]
        else:
            warnings.warn(
                f"Column '{col_name}' from y_columns was not found in the "
                f"forecast output. Available unique_ids: {list(pivot_parts.keys())}.",
                UserWarning,
                stacklevel=2,
            )

    result = pl.DataFrame(result_cols)

    # Add observed_time if provided
    if observed_time is not None:
        observed_col = pl.Series("observed_time", [observed_time[0]] * len(result)).cast(time_series.dtype)
        result = result.with_columns(observed_col).select(
            ["observed_time", "time"] + [c for c in result.columns if c not in ("time", "observed_time")]
        )

    return result
