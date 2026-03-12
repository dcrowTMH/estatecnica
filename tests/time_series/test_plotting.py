import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from estatecnica.time_series import plotting  # noqa: E402


class TimeSeriesPlottingTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(
            range(1, 31),
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
            name="value",
        )

    def test_visual_inspection_returns_axes_grid(self):
        axes = plotting.visual_inspection(self.series, show=False)
        self.assertEqual(axes.shape, (2, 2))

    def test_seasonal_plot_returns_axis(self):
        ax = plotting.seasonal_plot(self.series, period="week", freq="day", show=False)
        self.assertEqual(ax.get_xlabel(), "day")

    def test_periodogram_and_fft_plot_return_axes(self):
        periodogram_ax = plotting.plot_periodogram(self.series, show=False)
        fft_ax = plotting.plot_frequency_spectrum(self.series, show=False)

        self.assertEqual(periodogram_ax.get_ylabel(), "Variance")
        self.assertEqual(fft_ax.get_ylabel(), "Power")

    def test_calendar_pattern_plots_return_axes(self):
        day_ax = plotting.plot_day_of_week_pattern(self.series, show=False)
        month_ax = plotting.plot_month_pattern(self.series, show=False)
        week_ax = plotting.plot_week_of_year_pattern(self.series, show=False)

        self.assertEqual(day_ax.get_title(), "Average by Day of Week")
        self.assertEqual(month_ax.get_title(), "Monthly Pattern")
        self.assertEqual(week_ax.get_title(), "Week of Year Pattern")

    def test_compute_frequency_spectrum_returns_arrays(self):
        freqs, power, periods = plotting.compute_frequency_spectrum(self.series)

        self.assertEqual(len(freqs), len(power))
        self.assertGreater(len(periods), 0)


if __name__ == "__main__":
    unittest.main()
