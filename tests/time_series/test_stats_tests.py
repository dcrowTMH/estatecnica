import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series.stats_tests import (  # noqa: E402
    autocorrelation_ljung_box,
    format_adf_result,
    stationary_test_adf,
    suggest_arima_orders,
)


@unittest.skipIf(find_spec("statsmodels") is None, "statsmodels is not installed")
class TimeSeriesStatsTests(unittest.TestCase):
    def test_stationary_test_adf_returns_structured_result(self):
        rng = np.random.default_rng(42)
        series = pd.Series(
            rng.normal(size=120),
            index=pd.date_range("2024-01-01", periods=120, freq="D"),
        )

        result = stationary_test_adf(series)

        self.assertIsInstance(result.pvalue, float)
        self.assertIn("1%", result.critical_values)
        self.assertIsInstance(format_adf_result(result), dict)

    def test_autocorrelation_ljung_box_returns_dataframe(self):
        rng = np.random.default_rng(0)
        series = pd.Series(
            rng.normal(size=50),
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )

        result = autocorrelation_ljung_box(series, lags=5, return_df=True)

        self.assertTrue(hasattr(result, "columns"))
        self.assertIn("lb_stat", result.columns)
        self.assertIn("lb_pvalue", result.columns)

    def test_suggest_arima_orders_returns_dataclass(self):
        rng = np.random.default_rng(7)
        series = pd.Series(
            rng.normal(size=80),
            index=pd.date_range("2024-01-01", periods=80, freq="D"),
        )

        result = suggest_arima_orders(series, max_lags=10)

        self.assertGreaterEqual(result.p_suggest, 0)
        self.assertGreaterEqual(result.q_suggest, 0)
        self.assertEqual(result.n_lags_used, 10)
        self.assertEqual(len(result.acf_values), 11)


if __name__ == "__main__":
    unittest.main()
