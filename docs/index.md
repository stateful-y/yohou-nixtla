![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Yohou-Nixtla's documentation

Yohou-Nixtla brings the power of Nixtla's forecasting backends (**StatsForecast** and **NeuralForecast**) into the [Yohou](https://yohou.readthedocs.io/) ecosystem. Each backend is wrapped as a scikit-learn-compatible Yohou forecaster with full support for `fit`, `predict`, `observe`, and `rewind`, so you can use classical statistical models and deep learning architectures through a single unified API.

!!! note "Powered by Nixtla"
    Under the hood, Yohou-Nixtla delegates to [Nixtla](https://nixtla.io/) libraries. Data is automatically converted between Yohou's polars wide-format and Nixtla's pandas long-format, so you never need to wrangle DataFrames yourself.

<div class="grid cards" markdown>
-  **Get Started in 5 Minutes**

    ---

    Install Yohou-Nixtla and produce your first forecast

    [Getting Started](pages/tutorials/getting-started.md)

- **Need Help?**

    ---

    Find answers to common questions and troubleshooting tips.

    [Troubleshooting](pages/how-to/troubleshooting.md)

- **Learn the Concepts**

    ---

    Understand the forecasting backends, data conversion, and panel data

    [Concepts](pages/explanation/concepts.md)

- **See It In Action**

    ---

    Compare statistical and neural forecasters on real data

    [Examples](pages/tutorials/examples.md)

</div>

## Key Features

- **15 forecasters, one API**: 10 statistical models (AutoARIMA, AutoETS, Holt-Winters, ...) and 5 neural
architectures (NBEATS, NHITS, PatchTST, ...) all sharing `fit` / `predict` / `observe` / `rewind`.

- **Yohou and Scikit-Learn-compatible**: Every forecaster supports `clone`, `get_params`, and `set_params`.
Use Yohou's `GridSearchCV` for time series hyperparameter search.

- **Panel data out of the box**: Name columns with the `__` separator (`sales__store_1`) and
Yohou-Nixtla fits each group independently in a single call.

- **Automatic data conversion**: Polars wide-format DataFrames are converted to Nixtla's pandas
long-format transparently on every `fit` and `predict` call.

- **Exogenous features**: Pass external regressors through `X` with optional
`actual_transformer` for automatic scaling and preprocessing.

- **Minimal boilerplate**: Adding a new Nixtla model is a three-line class. No glue code,
no manual DataFrame wrangling.

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou-nixtla/blob/main/LICENSE).

## Acknowledgements

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in time series data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

![Made by stateful-y](assets/made_by_stateful-y.png){width=200}
