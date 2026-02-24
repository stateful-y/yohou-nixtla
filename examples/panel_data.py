# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
#     "yohou-nixtla",
#     "polars>=0.20",
#     "plotly>=5.19",
#     "numpy>=2.0",
# ]
# ///
"""Panel data forecasting with Nixtla wrappers."""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from datetime import datetime, timedelta

    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    from yohou_nixtla.stats import NaiveForecaster

    return (NaiveForecaster, datetime, go, make_subplots, np, pl, timedelta)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Panel Data Forecasting

        Yohou-Nixtla handles panel (grouped) time series automatically.
        Columns with the `__` separator are treated as panel groups.

        ## What You'll Learn

        - How to structure panel data using the `__` column naming convention
        - Fitting a single forecaster across multiple groups simultaneously
        - Generating group-level predictions with the same `fit`/`predict` API

        ## Prerequisites

        Basic familiarity with Yohou-Nixtla's fit/predict API. See [Model Comparison](/examples/model_comparison/) for an introduction.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Generate Panel Data")


@app.cell
def _(datetime, np, pl, timedelta):
    rng = np.random.default_rng(42)
    n = 120

    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )

    y = pl.DataFrame(
        {
            "time": time,
            "sales__store_A": (50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 30) + rng.normal(0, 3, n)).tolist(),
            "sales__store_B": (80 + 15 * np.sin(np.arange(n) * 2 * np.pi / 30) + rng.normal(0, 4, n)).tolist(),
        }
    )

    y.head(10)
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Fit and Predict")


@app.cell
def _(NaiveForecaster, y):
    forecaster = NaiveForecaster()
    forecaster.fit(y[:100], forecasting_horizon=20)
    y_pred = forecaster.predict()
    y_pred
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Visualize Results")


@app.cell
def _(go, make_subplots, y, y_pred):
    cols = [c for c in y.columns if c != "time"]

    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)

    for i, col in enumerate(cols, 1):
        fig.add_trace(
            go.Scatter(x=y["time"].to_list(), y=y[col].to_list(), name=f"{col} actual"),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=y_pred["time"].to_list(),
                y=y_pred[col].to_list(),
                name=f"{col} forecast",
                line={"dash": "dash"},
            ),
            row=1,
            col=i,
        )

    fig.update_layout(title="Panel Forecasting: Two Stores", showlegend=True)
    fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Takeaways

        - **`__` column naming convention** automatically declares panel groups
        - **Single forecaster** handles all groups at once -- no manual looping
        - Predictions preserve the same panel structure as the input data
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - **Model comparison**: See [Model Comparison](/examples/model_comparison/) to compare statistical and neural forecasters side by side
        """
    )


if __name__ == "__main__":
    app.run()
