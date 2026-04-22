# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly>=5.19",
#     "polars>=0.20",
#     "yohou",
#     "yohou-nixtla",
# ]
# ///
"""Model comparison across statsforecast and neuralforecast."""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    __gallery__ = {
        "title": "How to Compare Forecasters",
        "description": "Fit multiple statistical forecasters on the same dataset and evaluate their accuracy with MAE.",
        "category": "how-to",
    }

    return (mo,)


@app.cell(hide_code=True)
def _():
    from datetime import datetime

    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    from yohou_nixtla.stats import AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster

    return (AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster, datetime, go, np, pl)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Compare Forecasters

        This notebook shows how to compare multiple statistical forecasters on the
        same dataset and evaluate their accuracy. See
        [About Yohou-Nixtla](/pages/explanation/concepts/) for background on the
        available backends.

        **Prerequisites:** Yohou-Nixtla installed and familiarity with the
        fit/predict API ([Getting Started](/pages/tutorials/getting-started/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load Data

        Load the Air Passengers dataset and split into 120 training and 24 test observations.
        """
    )


@app.cell
def _(datetime, np, pl):
    # Generate synthetic monthly airline-style data (trend + seasonality)
    rng = np.random.default_rng(42)
    n = 144
    time = pl.datetime_range(
        start=datetime(1949, 1, 1),
        end=datetime(1960, 12, 1),
        interval="1mo",
        eager=True,
    )
    trend = np.linspace(100, 500, n)
    season = 40 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 10, n)
    y = pl.DataFrame({"time": time, "passengers": trend + season + noise})

    # Split: train on first 120, test on last 24
    y_train = y[:120]
    y_test = y[120:]

    target_col = "passengers"
    return (target_col, y, y_test, y_train)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Fit Forecasters

        Fit three statistical forecasters on the training data with a 24-step horizon.
        If you want to include neural forecasters, add
        [`NBEATSForecaster`](/pages/api/generated/yohou_nixtla.neural.NBEATSForecaster/)
        to the comparison.
        """
    )


@app.cell
def _(AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster, y_train):
    h = 24

    snaive = SeasonalNaiveForecaster(season_length=12)
    snaive.fit(y_train, forecasting_horizon=h)
    pred_snaive = snaive.predict()

    arima = AutoARIMAForecaster()
    arima.fit(y_train, forecasting_horizon=h)
    pred_arima = arima.predict()

    ets = AutoETSForecaster(season_length=12)
    ets.fit(y_train, forecasting_horizon=h)
    pred_ets = ets.predict()

    return (pred_arima, pred_ets, pred_snaive)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Visualize Results

        Plot actual values alongside each model's forecast.
        """
    )


@app.cell
def _(go, pred_arima, pred_ets, pred_snaive, target_col, y):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y["time"].to_list(),
            y=y[target_col].to_list(),
            mode="lines",
            name="Actual",
        )
    )

    models = {
        "SeasonalNaive": pred_snaive,
        "AutoARIMA": pred_arima,
        "AutoETS": pred_ets,
    }
    dashes = ["dash", "dot", "dashdot"]

    for (_name, _pred), _dash in zip(models.items(), dashes, strict=True):
        fig.add_trace(
            go.Scatter(
                x=_pred["time"].to_list(),
                y=_pred[target_col].to_list(),
                mode="lines",
                name=_name,
                line={"dash": _dash},
            )
        )

    fig.update_layout(
        title="Air Passengers: Multi-Backend Forecast Comparison",
        xaxis_title="Time",
        yaxis_title="Passengers",
    )
    fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Evaluate

        Compute MAE on the held-out test set.
        """
    )


@app.cell
def _(mo, np, pred_arima, pred_ets, pred_snaive, target_col, y_test):
    def mae(y_true, y_pred, col):
        """Compute mean absolute error."""
        truth = np.array(y_true[col].to_list())
        pred = np.array(y_pred[col].to_list())
        return float(np.mean(np.abs(truth - pred)))

    results = {
        "SeasonalNaive": mae(y_test, pred_snaive, target_col),
        "AutoARIMA": mae(y_test, pred_arima, target_col),
        "AutoETS": mae(y_test, pred_ets, target_col),
    }

    mo.md(
        "### Test MAE\n\n"
        + "\n".join(f"- **{k}**: {v:.2f}" for k, v in results.items())
    )


if __name__ == "__main__":
    app.run()
