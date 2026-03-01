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

    Install → Fit → Predict → Done

    [Getting Started](pages/getting-started.md)

- **Learn the Concepts**

    ---

    Understand the forecasting backends, data conversion, and panel data

    [User Guide](pages/user-guide.md)

- **See It In Action**

    ---

    Compare statistical and neural forecasters on real data

    [Examples](pages/examples.md)

- **API Reference**

    ---

    Complete reference for all forecaster classes and conversion utilities

    [API Reference](pages/api-reference.md)

</div>

## Table of Contents

### [Getting Started](pages/getting-started.md)

Step-by-step guide to installing and using Yohou-Nixtla.

- [Installation](pages/getting-started.md#installation)
- [Basic Usage](pages/getting-started.md#basic-usage)
- [Complete Example](pages/getting-started.md#complete-example)

### [User Guide](pages/user-guide.md)

In-depth documentation on the design, architecture, and core concepts.

- [Core Concepts](pages/user-guide.md#core-concepts)
- [Key Features](pages/user-guide.md#key-features)
- [Best Practices](pages/user-guide.md#best-practices)

### [Examples](pages/examples.md)

Interactive notebooks demonstrating real-world usage.

- [Model Comparison](pages/examples.md)
- [Panel Data Forecasting](pages/examples.md)

### [API Reference](pages/api-reference.md)

Detailed reference for the Yohou-Nixtla API, including all forecaster classes and conversion utilities.

## License

Yohou-Nixtla is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0). You are free to use, modify, and distribute this software under the terms of this license.
