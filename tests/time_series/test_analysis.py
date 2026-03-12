import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series.analysis import TimeSeriesAnalyzer  # noqa: E402


class TimeSeriesAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(
            [float(2 * (index + 1)) for index in range(70)],
            index=pd.date_range("2024-01-01", periods=70, freq="D"),
            name="value",
        )
        self.analyzer = TimeSeriesAnalyzer(self.series)

    def test_basic_statistics_returns_summary(self):
        result = self.analyzer.basic_statistics()

        self.assertEqual(result["n_obs"], 70)
        self.assertEqual(result["min"], 2.0)
        self.assertEqual(result["max"], 140.0)

    def test_anova_helpers_return_numeric_results(self):
        day_result = self.analyzer.day_of_week_anova()
        month_result = self.analyzer.month_anova()

        self.assertIsNotNone(day_result)
        self.assertIsNotNone(month_result)
        self.assertIn("p_value", day_result)
        self.assertIn("f_stat", month_result)

    def test_time_and_lag_fits_return_regression_results(self):
        time_fit = self.analyzer.fit_time_trend()
        lag_fit = self.analyzer.fit_lag_regression()

        self.assertAlmostEqual(time_fit.slope, 2.0, places=6)
        self.assertGreater(lag_fit.slope, 0.0)
        self.assertEqual(len(time_fit.fitted), 70)

    def test_trend_analysis_returns_future_projection(self):
        result = self.analyzer.trend_analysis(
            window_size=10, polynomial_order=1, future_steps=5, show=False
        )

        self.assertEqual(len(result["result"].forecast), 5)
        self.assertEqual(len(result["result"].fitted), 70)

    @unittest.skipIf(find_spec("statsmodels") is None, "statsmodels is not installed")
    def test_autocorrelation_analysis_returns_order_suggestion(self):
        result = self.analyzer.autocorrelation_analysis(find_pq=True, show=False)

        self.assertIn("order_suggestion", result)
        self.assertGreaterEqual(result["n_lags"], 1)

    @unittest.skipIf(find_spec("statsmodels") is None, "statsmodels is not installed")
    def test_statsmodels_wrappers_return_results(self):
        suggestion = self.analyzer.suggest_arima_orders(max_lags=2)
        decomposition = self.analyzer.seasonal_decomposition(period=2, show=False)

        self.assertGreaterEqual(suggestion.n_lags_used, 1)
        self.assertIn("seasonal_strength", decomposition)


if __name__ == "__main__":
    unittest.main()
