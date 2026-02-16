# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
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


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Panel Data Forecasting

        Yohou‑Nixtla handles panel (grouped) time series automatically.
        Columns with the ``__`` separator are treated as panel groups.
        """
    )
    return (mo,)


@app.cell
def _():
    from datetime import datetime, timedelta

    import numpy as np
    import polars as pl

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
    return (n, np, pl, rng, time, y)


@app.cell
def _(mo):
    mo.md("## Fit and Predict")


@app.cell
def _(y):
    from yohou_nixtla.stats import NaiveForecaster

    forecaster = NaiveForecaster()
    forecaster.fit(y[:100], forecasting_horizon=20)
    y_pred = forecaster.predict()
    y_pred
    return (y_pred,)


@app.cell
def _(y, y_pred):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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


if __name__ == "__main__":
    app.run()
