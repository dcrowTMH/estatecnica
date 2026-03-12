import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series.forecast import (  # noqa: E402
    TimeSeriesForecaster,
    average_forecast,
    baseline_forecasts,
    drift_forecast,
    fourier_features_forecast,
    naive_forecast,
    recent_drift_forecast,
    seasonal_naive_forecast,
)


class TimeSeriesForecastTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.Series(
            [10.0, 12.0, 14.0, 16.0],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        self.test = pd.Series(
            [18.0, 20.0],
            index=pd.date_range("2024-01-05", periods=2, freq="D"),
        )

    def test_naive_forecast_repeats_last_value(self):
        result = naive_forecast(self.train, self.test)
        self.assertEqual(list(result.predictions), [16.0, 16.0])

    def test_average_and_drift_forecasts_return_named_results(self):
        average = average_forecast(self.train, self.test)
        drift = drift_forecast(self.train, self.test)

        self.assertEqual(average.method, "average")
        self.assertEqual(drift.method, "drift")
        self.assertEqual(list(drift.predictions), [18.0, 20.0])

    def test_seasonal_and_recent_drift_forecasts_work(self):
        seasonal = seasonal_naive_forecast(self.train, self.test, seasonal_period=2)
        recent = recent_drift_forecast(self.train, self.test, window=2)

        self.assertEqual(seasonal.method, "seasonal_naive")
        self.assertEqual(recent.method, "recent_drift")
        self.assertEqual(len(seasonal.predictions), 2)

    def test_forecaster_baseline_set_uses_split(self):
        series = pd.Series(
            range(1, 21),
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
            name="value",
        )
        forecaster = TimeSeriesForecaster(
            series, train_pct=0.6, val_pct=0.2, test_pct=0.2
        )

        results = forecaster.baseline_forecasts()

        self.assertEqual(len(forecaster.train_ts), 12)
        self.assertEqual(len(forecaster.val_ts), 4)
        self.assertEqual(len(forecaster.test_ts), 4)
        self.assertSetEqual(
            set(results.keys()),
            {"naive", "average", "drift", "seasonal_naive", "recent_drift"},
        )

    def test_plot_pred_basic_rejects_unknown_method(self):
        series = pd.Series(
            range(1, 21),
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
            name="value",
        )
        forecaster = TimeSeriesForecaster(series)

        with self.assertRaises(ValueError):
            forecaster.plot_pred_basic("unknown", show=False)

    @unittest.skipIf(find_spec("statsmodels") is None, "statsmodels is not installed")
    def test_fourier_features_forecast_returns_fitted_and_future(self):
        series = pd.Series(
            range(1, 31),
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
            name="value",
        )
        forecaster = TimeSeriesForecaster(series)
        X, dp = forecaster.fourier_features(freq="M", order=2)

        result = fourier_features_forecast(X, forecaster.ts, dp, future_steps=3)

        self.assertEqual(len(result["forecast"]), 3)
        self.assertEqual(len(result["fitted"]), len(forecaster.ts))


if __name__ == "__main__":
    unittest.main()
