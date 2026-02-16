# API Reference

Complete API reference for Yohou-Nixtla.

## Stats Forecasters

Wrappers around [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/index.html) models for classical statistical time series forecasting.

::: yohou_nixtla.stats.BaseStatsForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.AutoARIMAForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.AutoETSForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.AutoCESForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.AutoThetaForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.NaiveForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.SeasonalNaiveForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.ARIMAForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.HoltWintersForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.ThetaForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.stats.CrostonForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

## Neural Forecasters

Wrappers around [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html) for deep learning-based time series forecasting using PyTorch models.

::: yohou_nixtla.neural.BaseNeuralForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.neural.NBEATSForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.neural.NHITSForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.neural.MLPForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.neural.PatchTSTForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

::: yohou_nixtla.neural.TimesNetForecaster
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters: ["!^_"]

## Data Conversion

Utilities for converting between Yohou's polars wide-format and Nixtla's pandas long-format. These are used internally by all forecasters but can also be called directly.

::: yohou_nixtla._conversion
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      filters: ["!^_"]
