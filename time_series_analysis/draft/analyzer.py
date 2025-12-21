"""
Draft TimeSeriesAnalyzer wrapper that composes the helper modules in the `draft/`
directory.

This module is a conservative, lightweight reimplementation that mirrors the
public surface of the original `TimeSeriesAnalyzer` but delegates work to the
small helper modules:
 - draft.transforms
 - draft.stats
 - draft.plots

Purpose:
 - Provide a reviewable intermediary before migrating logic into the original
   module.
 - Make methods small, return values (not only prints) where sensible so tests
   can assert behavior.
 - Keep the original file untouched while enabling notebooks to import from
   `time_series_analysis.draft.analyzer` for validation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .plots import (
    frequency_domain_analysis,
    plot_periodogram,
    seasonal_plot,
    visual_inspection,
)
from .stats import (
    autocorrelation_ljung_box,
    stationary_test_adf,
    suggest_arima_orders,
)

# Local-draft helpers (same directory)
from .transforms import (
    daily_to_weekly_and_yearly,
    split_time_series,
    to_series,
)


class TimeSeriesAnalyzer:
    """
    A compact TimeSeriesAnalyzer that delegates to the draft helper modules.

    Initialization accepts either:
      - a pandas.Series with a DatetimeIndex, or
      - a pandas.DataFrame with a date column and a value column.

    The initializer normalizes inputs to `self.ts` (pd.Series) and creates a
    minimal `self.df` (DataFrame) and sets `self.value_column`.
    """

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        weekly_method: str = "mean",
        week_start: str = "Monday",
    ) -> None:
        # Normalize to a pandas Series
        self.ts = to_series(data, date_col=date_column, value_col=value_column)
        # Determine the name used for the series values
        self.value_column = value_column or (self.ts.name if self.ts.name else "value")
        # Keep a DataFrame view for compatibility with plotting code or regression
        self.df = pd.DataFrame({self.value_column: self.ts.copy()})
        # Add simple time and lag_1 columns similar to original class
        self.df["time"] = np.arange(len(self.df.index))
        self.df["lag_1"] = self.df[self.value_column].shift(1)

        # Aggregations: daily, weekly, monthly, yearly
        self.daily, self.weekly, self.monthly, self.yearly = daily_to_weekly_and_yearly(
            self.ts, method=weekly_method, week_start=week_start
        )

    def basic_statistics(self) -> Dict[str, Any]:
        """Return basic descriptive statistics for the time series."""
        ts = self.ts
        result = {
            "start": ts.index.min(),
            "end": ts.index.max(),
            "n_obs": len(ts),
            "mean": float(ts.mean()),
            "std": float(ts.std()),
            "min": float(ts.min()),
            "max": float(ts.max()),
            "n_missing": int(ts.isna().sum()),
        }
        return result

    # ------------------------
    # Plotting wrappers
    # ------------------------
    def visual_inspection(self, show: bool = True):
        """
        Produce the 2x2 visual inspection plot.

        Returns the matplotlib axes object created by the plotting helper.
        """
        return visual_inspection(
            ts=self.ts,
            value_name=self.value_column,
            monthly=self.monthly,
            weekly=self.weekly,
            show=show,
        )

    def seasonal_plot(
        self,
        period: str = "week",
        freq: str = "day",
        ax: Optional[Any] = None,
        show: bool = True,
    ):
        return seasonal_plot(
            self.ts,
            value_column=self.value_column,
            period=period,
            freq=freq,
            ax=ax,
            show=show,
        )

    def plot_periodogram(
        self, detrend: str = "linear", ax: Optional[Any] = None, show: bool = True
    ):
        return plot_periodogram(self.ts, detrend=detrend, ax=ax, show=show)

    def frequency_domain_analysis(self, frequency: float = 1.0, show: bool = True):
        return frequency_domain_analysis(self.ts, frequency=frequency, show=show)

    # ------------------------
    # Statistical wrappers
    # ------------------------
    def stationary_test_adf(self, significance: float = 0.05):
        """
        Return a structured result from the ADF test.

        Delegates to draft.stats.stationary_test_adf which returns a dataclass.
        """
        return stationary_test_adf(self.ts, significance=significance)

    def autocorrelation_ljung_box(self, lags: int = 10, return_df: bool = True):
        """
        Delegates to the Ljung-Box helper; returns the underlying result.
        """
        return autocorrelation_ljung_box(self.ts, lags=lags, return_df=return_df)

    def suggest_arima_orders(
        self, max_lags: int = 40, fft: bool = False
    ) -> Dict[str, Any]:
        """
        Compute ACF/PACF and return suggested p and q (and diagnostic arrays).

        The returned dict contains:
           - 'p_suggest', 'q_suggest', 'acf', 'pacf', 'conf_interval', 'n_lags_used'
        """
        return suggest_arima_orders(self.ts, max_lags=max_lags, fft=fft)

    # ------------------------
    # Forecasting / utilities
    # ------------------------
    @staticmethod
    def split_time_series(
        ts: pd.Series,
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ):
        """Thin wrapper exposing transforms.split_time_series behavior."""
        return split_time_series(
            ts, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct
        )

    # ------------------------
    # Small convenience helpers (non-invasive)
    # ------------------------
    def day_of_week_anova(self):
        """
        Convenience: perform day-of-week ANOVA and return (f_stat, p_value) if possible.

        This helper uses pandas grouping and scipy.stats.f_oneway internally via plot
        helpers, but returns numeric results (does not plot).
        """
        df = self.ts.to_frame(name=self.value_column)
        df["dow"] = df.index.dayofweek
        groups = [
            df[df["dow"] == i][self.value_column].values
            for i in range(7)
            if len(df[df["dow"] == i]) > 0
        ]
        if len(groups) < 2:
            return None
        try:
            from scipy import stats
        except Exception:
            return None
        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": float(f_stat), "p_value": float(p_value)}

    # ------------------------
    # Representation
    # ------------------------
    def __repr__(self) -> str:
        return f"<TimeSeriesAnalyzerDraft n_obs={len(self.ts)} start={self.ts.index.min()} end={self.ts.index.max()}>"
