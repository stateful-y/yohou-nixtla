# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
#     "yohou-nixtla",
#     "polars>=0.20",
#     "plotly>=5.19",
# ]
# ///
"""Model comparison across statsforecast and neuralforecast."""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import plotly.graph_objects as go

    from yohou.datasets import load_air_passengers
    from yohou_nixtla.stats import AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster

    return (AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster, go, load_air_passengers, np)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Model Comparison: Stats vs Neural

        Compare forecasters from both Nixtla backends on the same dataset.

        ## What You'll Learn

        - How to use Yohou-Nixtla's unified `fit`/`predict` API across multiple Nixtla backends
        - Comparing `SeasonalNaiveForecaster`, `AutoARIMAForecaster`, and `AutoETSForecaster`
        - Evaluating forecast accuracy with MAE

        ## Prerequisites

        Basic familiarity with time series concepts (trend, seasonality) and the fit/predict pattern.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Load Data")


@app.cell
def _(load_air_passengers):
    y = load_air_passengers()

    # Split: train on first 120, test on last 24
    y_train = y[:120]
    y_test = y[120:]

    target_col = [c for c in y.columns if c != "time"][0]
    return (target_col, y, y_test, y_train)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Fit Forecasters")


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
    mo.md(r"## 3. Visualize Results")


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
    mo.md(r"## 4. Evaluate")


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Takeaways

        - **Unified API** -- Yohou-Nixtla provides the same `fit`/`predict` interface across all Nixtla backends
        - **SeasonalNaive** serves as a simple baseline with no parameter tuning
        - **AutoARIMA** and **AutoETS** automatically select the best model configuration
        - All forecasters return Polars DataFrames with consistent column naming
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - **Panel data**: See [Panel Data Forecasting](/examples/panel_data/) to forecast multiple related time series simultaneously
        """
    )


if __name__ == "__main__":
    app.run()
