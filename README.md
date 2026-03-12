# estatecnica

This repository is my statistical analysis toolkit for exploring data, testing ideas,
and building reusable helpers for future work. It currently focuses on statistical
tests, time-series analysis, simulation, and lightweight forecasting workflows.

The codebase is being refactored from notebook-oriented exploration into a cleaner
package structure so the utilities are easier to maintain, extend, and reuse.

The notebooks are mainly used as showcase material and as reminders of how the
features should be used.

## Environment

This repository uses `uv` for dependency and environment management.

Common commands:

```bash
uv sync
uv run jupyter lab
uv run python -m unittest tests.time_series.test_transforms
uv run python -m unittest tests.time_series.test_forecast
uv run python -m unittest tests.time_series.test_analysis
```

## Time-Series Usage

### Load the toolkit

```python
import pandas as pd

from estatecnica.time_series import TimeSeriesAnalyzer, TimeSeriesForecaster
from estatecnica.time_series.simulation import ARIMASimulator
```

### Analyze a dated series

```python
df = pd.read_csv("data/AAPL_sample_data.csv")

analyzer = TimeSeriesAnalyzer(
    df,
    date_column="date",
    value_column="close_p",
)

analyzer.basic_statistics()
analyzer.visual_inspection()
analyzer.plot_periodogram()
analyzer.autocorrelation_analysis(find_pq=True)
```

### Visualization examples

```python
# exploration
analyzer.visual_inspection()
analyzer.plot_time_and_lag()
analyzer.plot_time_trend()
analyzer.plot_lag_regression()

# seasonality and calendar patterns
analyzer.seasonal_plot()
analyzer.seasonal_plot(period="year", freq="dayofyear")
analyzer.plot_day_of_week_pattern()
analyzer.plot_month_pattern()
analyzer.plot_week_of_year_pattern()

# correlation and frequency views
analyzer.plot_periodogram()
analyzer.frequency_domain_analysis()
analyzer.autocorrelation_analysis(find_pq=True)

# trend projection
analyzer.trend_analysis(window_size=60, polynomial_order=2)
```

### Run baseline forecasts

```python
forecaster = TimeSeriesForecaster(
    df,
    date_column="date",
    value_column="close_p",
)

results = forecaster.baseline_forecasts()
results["naive"].rmse
results["drift"].rmse

forecaster.plot_pred_basic("naive")
forecaster.plot_pred_basic("season_naive")
forecaster.plot_pred_basic("recent_drift")
```

### Generate synthetic time-series data

```python
simulator = ARIMASimulator(start_date="2024-01-01", n_samples=180, seed=42)
simulated = simulator.generate_arima_data(d=1, p_coeffs=(0.6,), q_coeffs=(0.3,))

analyzer = TimeSeriesAnalyzer(simulated, date_column="date", value_column="value")
analyzer.frequency_domain_analysis()
```

### Run a specific test module

```bash
uv run python -m unittest tests.time_series.test_stats_tests
uv run python -m unittest tests.time_series.test_simulation
```
