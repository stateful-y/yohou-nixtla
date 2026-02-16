# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
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


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Model Comparison: Stats vs Neural

        Compare forecasters from both Nixtla backends on the same dataset.
        """
    )
    return (mo,)


@app.cell
def _():
    import polars as pl
    from yohou.datasets import load_air_passengers

    y = load_air_passengers()

    # Split: train on first 120, test on last 24
    y_train = y[:120]
    y_test = y[120:]

    target_col = [c for c in y.columns if c != "time"][0]
    return (pl, target_col, y, y_test, y_train)


@app.cell
def _(mo):
    mo.md("## Fit forecasters from each backend")


@app.cell
def _(y_train):
    from yohou_nixtla.stats import AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster

    h = 24

    # Stats
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


@app.cell
def _(pl, pred_arima, pred_ets, pred_snaive, target_col, y, y_test):
    import plotly.graph_objects as go

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


@app.cell
def _(mo, pred_arima, pred_ets, pred_snaive, target_col, y_test):
    import numpy as np

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
        "## Test MAE\n\n"
        + "\n".join(f"- **{k}**: {v:.2f}" for k, v in results.items())
    )


if __name__ == "__main__":
    app.run()
