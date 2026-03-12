"""High-level analysis facade for time-series workflows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .plotting import (
    compute_frequency_spectrum,
    plot_autocorrelation_diagnostics,
    plot_day_of_week_pattern,
    plot_frequency_spectrum,
    plot_lag_regression,
    plot_month_pattern,
    plot_periodogram,
    plot_time_and_lag,
    plot_time_regression,
    plot_trend_forecast,
    plot_week_of_year_pattern,
    seasonal_decomposition_plot,
    seasonal_plot,
    visual_inspection,
)
from .stats_tests import (
    autocorrelation_ljung_box,
    stationary_test_adf,
    suggest_arima_orders,
)
from .transforms import calendar_aggregates, split_time_series, to_series


@dataclass(frozen=True)
class RegressionFit:
    """Simple regression result for exploratory trend fitting."""

    intercept: float
    slope: float
    fitted: pd.Series


@dataclass(frozen=True)
class TrendAnalysisResult:
    """Trend fit summary including future projection."""

    moving_average: pd.Series
    fitted: pd.Series
    forecast: pd.Series
    coefficients: np.ndarray


def _require_statsmodels():
    if find_spec("statsmodels") is None:
        raise ModuleNotFoundError("statsmodels is required for this analysis method.")

    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
    from statsmodels.tsa.seasonal import seasonal_decompose

    return CalendarFourier, DeterministicProcess, seasonal_decompose


class TimeSeriesAnalyzer:
    """High-level facade that composes the time-series helper modules."""

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        aggregate_method: str = "mean",
        week_start: str = "Monday",
    ) -> None:
        self.ts = to_series(data, date_col=date_column, value_col=value_column)
        self.value_column = value_column or (self.ts.name if self.ts.name else "value")
        self.df = pd.DataFrame({self.value_column: self.ts.copy()})
        self.df["time"] = np.arange(len(self.df.index))
        self.df["lag_1"] = self.df[self.value_column].shift(1)

        self.daily, self.weekly, self.monthly, self.yearly = calendar_aggregates(
            self.ts,
            method=aggregate_method,
            week_start=week_start,
        )

    def basic_statistics(self) -> Dict[str, Any]:
        """Return descriptive statistics for the time series."""
        ts = self.ts
        return {
            "start": ts.index.min(),
            "end": ts.index.max(),
            "n_obs": len(ts),
            "mean": float(ts.mean()),
            "std": float(ts.std()),
            "min": float(ts.min()),
            "max": float(ts.max()),
            "n_missing": int(ts.isna().sum()),
        }

    def visual_inspection(self, show: bool = True):
        """Return the default 2x2 exploratory plot."""
        return visual_inspection(
            ts=self.ts,
            value_name=self.value_column,
            monthly=self.monthly,
            show=show,
        )

    def seasonal_plot(
        self,
        period: str = "week",
        freq: str = "day",
        ax: Optional[Any] = None,
        show: bool = True,
    ):
        """Plot repeating seasonal structure across weeks or years."""
        return seasonal_plot(
            self.ts,
            period=period,
            freq=freq,
            value_name=self.value_column,
            ax=ax,
            show=show,
        )

    def plot_periodogram(
        self, detrend: str = "linear", ax: Optional[Any] = None, show: bool = True
    ):
        """Plot the periodogram of the series."""
        return plot_periodogram(self.ts, detrend=detrend, ax=ax, show=show)

    def frequency_domain_analysis(
        self, frequency: float = 1.0, show: bool = True
    ) -> Dict[str, Any]:
        """Return FFT spectrum data and optionally render the spectrum plot."""
        frequencies, power, dominant_periods = compute_frequency_spectrum(
            self.ts,
            frequency=frequency,
        )
        ax = (
            plot_frequency_spectrum(self.ts, frequency=frequency, show=show)
            if show
            else None
        )
        return {
            "frequencies": frequencies,
            "power": power,
            "dominant_periods": dominant_periods,
            "ax": ax,
        }

    def stationary_test_adf(self, significance: float = 0.05):
        """Run the Augmented Dickey-Fuller test on the series."""
        return stationary_test_adf(self.ts, significance=significance)

    def autocorrelation_ljung_box(self, lags: int = 10, return_df: bool = True):
        """Run the Ljung-Box autocorrelation test."""
        return autocorrelation_ljung_box(self.ts, lags=lags, return_df=return_df)

    def suggest_arima_orders(
        self,
        max_lags: int = 40,
        fft: bool = False,
        segment: str = "daily",
        series: Optional[pd.Series] = None,
    ):
        """Suggest ARIMA p and q values for a selected segment."""
        if series is not None:
            target = to_series(series)
        elif segment == "daily":
            target = self.ts
        elif segment == "weekly":
            target = self.weekly.dropna()
        elif segment == "monthly":
            target = self.monthly.dropna()
        else:
            raise ValueError("segment must be one of: daily, weekly, monthly")
        return suggest_arima_orders(target, max_lags=max_lags, fft=fft)

    def autocorrelation_analysis(
        self,
        diffed_ts: Optional[pd.Series] = None,
        find_pq: bool = False,
        segment: str = "daily",
        show: bool = True,
    ) -> Dict[str, Any]:
        """Plot autocorrelation diagnostics and return summary lag statistics."""
        if diffed_ts is not None:
            target = to_series(diffed_ts)
        elif segment == "daily":
            target = self.ts.dropna()
        elif segment == "weekly":
            target = self.weekly.dropna()
        elif segment == "monthly":
            target = self.monthly.dropna()
        else:
            raise ValueError("segment must be one of: daily, weekly, monthly")

        n_lags = min(40, max(1, len(target) // 2 - 1))
        plots = plot_autocorrelation_diagnostics(target, n_lags=n_lags, show=show)
        lag_correlations = {
            lag: float(self.ts.autocorr(lag=lag))
            for lag in (7, 30, 90, 180, 365)
            if len(self.ts) > lag
        }
        result: Dict[str, Any] = {
            "series": target,
            "n_lags": n_lags,
            "lag_correlations": lag_correlations,
            "plots": plots,
        }
        if find_pq:
            result["order_suggestion"] = suggest_arima_orders(
                target,
                max_lags=n_lags,
                fft=False,
            )
        return result

    def seasonal_decomposition(
        self,
        model: str = "additive",
        period: int = 365,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Decompose the series into trend, seasonal, and residual components."""
        _, _, seasonal_decompose = _require_statsmodels()
        decomposition = seasonal_decompose(self.ts.dropna(), model=model, period=period)
        seasonal_strength = np.var(decomposition.seasonal) / np.var(
            decomposition.seasonal + decomposition.resid
        )
        axes = seasonal_decomposition_plot(decomposition, show=show) if show else None
        return {
            "decomposition": decomposition,
            "seasonal_strength": float(seasonal_strength),
            "axes": axes,
        }

    def day_of_week_anova(self) -> Optional[Dict[str, float]]:
        """Return a day-of-week ANOVA summary if enough groups exist."""
        groups = [
            self.ts[self.ts.index.dayofweek == index].values
            for index in range(7)
            if len(self.ts[self.ts.index.dayofweek == index]) > 0
        ]
        if len(groups) < 2:
            return None
        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": float(f_stat), "p_value": float(p_value)}

    def month_anova(self) -> Optional[Dict[str, float]]:
        """Return a month-of-year ANOVA summary if enough groups exist."""
        groups = [
            self.ts[self.ts.index.month == month].values
            for month in range(1, 13)
            if len(self.ts[self.ts.index.month == month]) > 0
        ]
        if len(groups) < 2:
            return None
        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": float(f_stat), "p_value": float(p_value)}

    def week_of_year_anova(self) -> Optional[Dict[str, float]]:
        """Return a week-of-year ANOVA summary if enough groups exist."""
        weeks = self.ts.index.isocalendar().week.astype(int)
        groups = [
            self.ts[weeks == week].values
            for week in sorted(pd.unique(weeks))
            if len(self.ts[weeks == week]) > 0
        ]
        if len(groups) < 2:
            return None
        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": float(f_stat), "p_value": float(p_value)}

    def plot_day_of_week_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Plot day-of-week averages."""
        return plot_day_of_week_pattern(
            self.ts, value_name=self.value_column, ax=ax, show=show
        )

    def day_of_week_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Compatibility wrapper for the legacy day-of-week plot method."""
        return self.plot_day_of_week_pattern(ax=ax, show=show)

    def plot_month_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Plot month-of-year averages."""
        return plot_month_pattern(
            self.ts, value_name=self.value_column, ax=ax, show=show
        )

    def month_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Compatibility wrapper for the legacy month-of-year plot method."""
        return self.plot_month_pattern(ax=ax, show=show)

    def plot_week_of_year_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Plot week-of-year averages."""
        return plot_week_of_year_pattern(
            self.ts,
            value_name=self.value_column,
            ax=ax,
            show=show,
        )

    def week_of_year_pattern(self, ax: Optional[Any] = None, show: bool = True):
        """Compatibility wrapper for the legacy week-of-year plot method."""
        return self.plot_week_of_year_pattern(ax=ax, show=show)

    def plot_time_and_lag(self, show: bool = True):
        """Plot the time index and lag-1 relationship."""
        return plot_time_and_lag(self.df, value_column=self.value_column, show=show)

    def fit_time_trend(self) -> RegressionFit:
        """Fit a simple linear trend over time using numpy."""
        x = self.df["time"].to_numpy(dtype=float)
        y = self.df[self.value_column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = pd.Series(intercept + slope * x, index=self.df.index)
        return RegressionFit(
            intercept=float(intercept), slope=float(slope), fitted=fitted
        )

    def plot_time_trend(self, show: bool = True):
        """Plot observed values against the fitted time trend."""
        fit = self.fit_time_trend()
        return plot_time_regression(self.ts, fit.fitted, self.value_column, show=show)

    def fit_lag_regression(self) -> RegressionFit:
        """Fit a simple lag-1 regression using numpy."""
        lag_df = self.df[["lag_1", self.value_column]].dropna()
        x = lag_df["lag_1"].to_numpy(dtype=float)
        y = lag_df[self.value_column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = pd.Series(intercept + slope * x, index=lag_df.index)
        return RegressionFit(
            intercept=float(intercept), slope=float(slope), fitted=fitted
        )

    def plot_lag_regression(self, show: bool = True):
        """Plot observed and fitted lag-1 regression values."""
        fit = self.fit_lag_regression()
        lag_series = self.df["lag_1"].dropna()
        return plot_lag_regression(
            lag_series,
            self.ts,
            fit.fitted,
            self.value_column,
            show=show,
        )

    def trend_analysis(
        self,
        window_size: int = 30,
        polynomial_order: int = 3,
        future_steps: int = 90,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Fit a polynomial trend, create a future projection, and optionally plot it."""
        window = max(2, min(window_size, max(2, len(self.ts) // 4)))
        moving_average = self.ts.rolling(
            window=window,
            center=True,
            min_periods=int(np.ceil(window / 2)),
        ).mean()

        x = np.arange(len(self.ts), dtype=float)
        coefficients = np.polyfit(
            x, self.ts.to_numpy(dtype=float), deg=polynomial_order
        )
        fitted = pd.Series(np.polyval(coefficients, x), index=self.ts.index)

        future_index = pd.date_range(
            start=self.ts.index[-1] + pd.Timedelta(days=1),
            periods=future_steps,
            freq="D",
        )
        future_x = np.arange(len(self.ts), len(self.ts) + future_steps, dtype=float)
        forecast = pd.Series(np.polyval(coefficients, future_x), index=future_index)

        axes = (
            plot_trend_forecast(
                self.ts,
                moving_average,
                fitted,
                forecast,
                self.value_column,
                show=show,
            )
            if show
            else None
        )

        result = TrendAnalysisResult(
            moving_average=moving_average,
            fitted=fitted,
            forecast=forecast,
            coefficients=np.asarray(coefficients, dtype=float),
        )
        return {"result": result, "plot": axes}

    def fourier_features(self, freq: str = "A", order: int = 5):
        """Create deterministic Fourier features for the series index."""
        CalendarFourier, DeterministicProcess, _ = _require_statsmodels()
        fourier = CalendarFourier(freq=freq, order=order)
        process = DeterministicProcess(
            index=self.ts.asfreq("D").index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        return process.in_sample(), process

    def fourier_features_capture(self, freq: str = "A", order: int = 5):
        """Compatibility wrapper for the legacy Fourier feature method name."""
        return self.fourier_features(freq=freq, order=order)

    def time_step_lag_linear_regression_plot(self, show: bool = True):
        """Compatibility wrapper for the old combined time/lag scatter plot."""
        return self.plot_time_and_lag(show=show)

    def time_step_linear_regression_fit(self, show: bool = True):
        """Compatibility wrapper returning the time-trend fit."""
        fit = self.fit_time_trend()
        if show:
            self.plot_time_trend(show=show)
        return fit

    def lag_linear_regression_fit(self, show: bool = True):
        """Compatibility wrapper returning the lag regression fit."""
        fit = self.fit_lag_regression()
        if show:
            self.plot_lag_regression(show=show)
        return fit

    @staticmethod
    def split_time_series(
        ts: pd.Series,
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ):
        """Expose the shared chronological split helper."""
        return split_time_series(
            ts, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct
        )

    def __repr__(self) -> str:
        return (
            f"<TimeSeriesAnalyzer n_obs={len(self.ts)} "
            f"start={self.ts.index.min()} end={self.ts.index.max()}>"
        )
