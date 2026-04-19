![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Yohou-Nixtla's documentation

Yohou-Nixtla brings the power of Nixtla's forecasting backends (**StatsForecast** and **NeuralForecast**) into the Yohou ecosystem. Each backend is wrapped as a scikit-learn-compatible forecaster with full support for `fit`, `predict`, `update`, and `reset`, so you can use classical statistical models and deep learning architectures through a single unified API.

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

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou-nixtla/blob/main/LICENSE).

## Acknowledgements

This project is maintained by [stateful-y](https://stateful-y.io), an ML consultancy specializing in time series data science & engineering. If you're interested in collaborating or learning more about our services, please visit our website.

![Made by stateful-y](assets/made_by_stateful-y.png){width=200}
