import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series.transforms import (  # noqa: E402
    calendar_aggregates,
    split_time_series,
    to_series,
)


class TimeSeriesTransformsTests(unittest.TestCase):
    def test_to_series_accepts_dataframe(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = to_series(df, date_col="date", value_col="value")

        self.assertIsInstance(result, pd.Series)
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))
        self.assertEqual(result.iloc[-1], 4.0)

    def test_calendar_aggregates_returns_all_periods(self):
        series = pd.Series(
            [1, 2, 3, 4, 5, 6, 7],
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
            name="value",
        )

        daily, weekly, monthly, yearly = calendar_aggregates(
            series,
            method="sum",
            week_start="Sunday",
        )

        self.assertEqual(len(daily), 7)
        self.assertEqual(int(weekly.iloc[-1]), 28)
        self.assertEqual(int(monthly.iloc[0]), 28)
        self.assertEqual(int(yearly.iloc[0]), 28)

    def test_split_time_series_respects_percentages(self):
        series = pd.Series(
            range(10),
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )

        train, val, test = split_time_series(
            series, train_pct=0.6, val_pct=0.2, test_pct=0.2
        )

        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()
