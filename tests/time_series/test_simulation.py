import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series.simulation import (  # noqa: E402
    ARIMASimulator,
    generate_arima_dataframe,
    generate_arima_values,
)


class TimeSeriesSimulationTests(unittest.TestCase):
    def test_generate_arima_values_is_deterministic_for_seed(self):
        first = generate_arima_values(
            n_samples=8, seed=10, p_coeffs=(0.2,), q_coeffs=(0.1,)
        )
        second = generate_arima_values(
            n_samples=8, seed=10, p_coeffs=(0.2,), q_coeffs=(0.1,)
        )

        self.assertTrue(np.allclose(first, second))
        self.assertEqual(len(first), 8)

    def test_generate_arima_dataframe_has_expected_columns(self):
        result = generate_arima_dataframe(start_date="2024-01-01", n_samples=5, seed=3)

        self.assertListEqual(list(result.columns), ["date", "value"])
        self.assertEqual(len(result), 5)
        self.assertEqual(str(result["date"].iloc[0].date()), "2024-01-01")

    def test_arima_simulator_wrapper_returns_dataframe(self):
        simulator = ARIMASimulator(start_date="2024-02-01", n_samples=4, seed=7)

        result = simulator.generate_arima_data(d=1, drift=0.5)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
