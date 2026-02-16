# Examples

Learn Yohou-Nixtla through focused, interactive examples. Each notebook demonstrates one core concept and is runnable and editable locally or online.

## Getting Started

### Model Comparison ([View](/examples/model_comparison/) | [Editable](/examples/model_comparison/edit/))

**Comparing Statistical and Neural Forecasters**

Start here to see Yohou-Nixtla's unified API in action. This example fits an `AutoARIMAForecaster` and an `NBEATSForecaster` to the classic Air Passengers dataset, then visualizes their predictions side by side. You'll understand the strengths and trade-offs of each backend while using the same `fit`/`predict` interface throughout.

### Panel Data ([View](/examples/panel_data/) | [Editable](/examples/panel_data/edit/))

**Forecasting Multiple Related Time Series**

Forecast multiple related time series simultaneously using Yohou's `__` column naming convention. This example shows how to structure panel data, fit a single forecaster across all groups, and generate group-level predictions, all through the same `fit`/`predict` API. You'll learn the `__` prefix convention for declaring panel columns and see how one forecaster handles all groups at once.

## Next Steps

- **[User Guide](user-guide.md)**: Deep dive into core concepts and best practices
- **[API Reference](api-reference.md)**: Complete documentation on all forecaster classes
- **[Contributing](contributing.md)**: Add your own examples or improve existing ones
