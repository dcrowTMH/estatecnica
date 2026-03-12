"""Reusable plotting helpers for time-series workflows."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def ensure_datetime_index(ts: pd.Series) -> pd.Series:
    """Validate and coerce the input to a Series with a DatetimeIndex."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        try:
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index)
        except Exception as exc:
            raise ValueError(
                "Series index must be datetime-like or convertible to datetime."
            ) from exc
    return ts.sort_index()


def visual_inspection(
    ts: pd.Series,
    value_name: Optional[str] = None,
    monthly: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
):
    """Create a 2x2 exploratory plot for a time series."""
    ts = ensure_datetime_index(ts)
    name = value_name or ts.name or "value"
    if monthly is None:
        monthly = ts.resample("MS").mean()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(monthly.index, monthly.values, color="C0")
    axes[0, 0].set_title("Time Series (Monthly)")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel(name)

    month_groups = ts.groupby(ts.index.month).apply(list)
    axes[0, 1].boxplot(
        [month_groups.get(month, []) for month in range(1, 13)],
        patch_artist=True,
    )
    axes[0, 1].set_title("Monthly Distribution")
    axes[0, 1].set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        rotation=45,
    )
    axes[0, 1].set_ylabel(name)

    dow_groups = [ts[ts.index.dayofweek == index].values for index in range(7)]
    axes[1, 0].boxplot(dow_groups, patch_artist=True)
    axes[1, 0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1, 0].set_title("Day-of-Week Distribution")
    axes[1, 0].set_ylabel(name)

    axes[1, 1].hist(ts.values, bins=30, alpha=0.7, color="C2")
    axes[1, 1].set_title("Value Distribution")
    axes[1, 1].set_xlabel(name)
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    if show:
        plt.show()
    return axes


def seasonal_plot(
    ts: pd.Series,
    period: str = "week",
    freq: str = "day",
    value_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot seasonal lines across weeks or years."""
    ts = ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    df = ts.to_frame(name=name)
    if period == "week":
        df["period"] = df.index.isocalendar().week.astype(int)
    elif period == "year":
        df["period"] = df.index.year
    else:
        raise ValueError("Unsupported period: use 'week' or 'year'")

    if freq == "day":
        df["freq"] = df.index.dayofweek
    elif freq == "dayofyear":
        df["freq"] = df.index.dayofyear
    else:
        raise ValueError("Unsupported freq: use 'day' or 'dayofyear'")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for label, group in df.groupby("period"):
        ordered = group.sort_values("freq")
        ax.plot(ordered["freq"], ordered[name], linewidth=1.2, alpha=0.8)
        if len(ordered) > 0:
            ax.annotate(
                str(label), xy=(ordered["freq"].iloc[-1], ordered[name].iloc[-1])
            )

    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    ax.set_xlabel(freq)
    ax.set_ylabel(name)

    if show:
        plt.show()
    return ax


def seasonal_decomposition_plot(decomposition, show: bool = True):
    """Plot a statsmodels seasonal decomposition result."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=axes[0], title="Original")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    if show:
        plt.show()
    return axes


def plot_periodogram(
    ts: pd.Series,
    detrend: str = "linear",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a periodogram using scipy.signal.periodogram."""
    ts = ensure_datetime_index(ts)
    freqs, power = periodogram(
        ts.values,
        fs=1.0,
        detrend=detrend,
        window="boxcar",
        scaling="spectrum",
    )

    positive = freqs > 0
    freqs = freqs[positive]
    power = power[positive]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.step(freqs, power, color="purple")
    ax.set_xscale("log")
    tick_values = np.array([1 / 365.0, 1 / 180.0, 1 / 90.0, 1 / 30.0, 1 / 7.0, 1.0])
    tick_labels = ["Annual", "Semiannual", "Quarterly", "Monthly", "Weekly", "Daily"]
    mask = (
        (tick_values >= freqs.min()) & (tick_values <= freqs.max())
        if freqs.size
        else []
    )
    if freqs.size:
        selected_ticks = tick_values[mask]
        selected_labels = [label for label, keep in zip(tick_labels, mask) if keep]
        if len(selected_ticks) > 0:
            ax.set_xticks(selected_ticks)
            ax.set_xticklabels(selected_labels, rotation=30)
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram (power spectrum)")

    if show:
        plt.show()
    return ax


def plot_autocorrelation_diagnostics(
    ts: pd.Series,
    n_lags: int,
    show: bool = True,
):
    """Plot ACF and PACF diagnostics for a time series."""
    ts = ensure_datetime_index(ts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(ts.dropna(), lags=n_lags, ax=axes[0], title="Autocorrelation Function")
    plot_pacf(
        ts.dropna(),
        lags=n_lags,
        ax=axes[1],
        title="Partial Autocorrelation Function",
    )
    axes[0].set_xlabel("Lags")
    axes[1].set_xlabel("Lags")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def compute_frequency_spectrum(
    ts: pd.Series,
    frequency: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the positive-frequency FFT spectrum and dominant periods."""
    ts = ensure_datetime_index(ts)
    values = ts.values
    if len(values) < 2:
        raise ValueError("Time series too short for FFT")

    fft_values = fft(values)
    freqs = fftfreq(len(values), d=1.0 / frequency)
    power = np.abs(fft_values) ** 2
    positive = freqs > 0
    pos_freqs = freqs[positive]
    pos_power = power[positive]

    top_count = min(5, len(pos_power))
    if top_count == 0:
        return pos_freqs, pos_power, np.array([])

    top_index = np.argsort(pos_power)[-top_count:]
    dominant_periods = np.array(
        [1.0 / freq for freq in pos_freqs[top_index] if freq != 0]
    )
    return pos_freqs, pos_power, np.sort(dominant_periods)


def plot_frequency_spectrum(
    ts: pd.Series,
    frequency: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a power spectrum using FFT."""
    pos_freqs, pos_power, _ = compute_frequency_spectrum(ts, frequency=frequency)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(pos_freqs, pos_power, linewidth=1.0)
    ax.set_title("Power Spectrum (FFT)")
    ax.set_xlabel("Frequency (cycles per time unit)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()
    return ax


def plot_day_of_week_pattern(
    ts: pd.Series,
    value_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot average values by day of week."""
    ts = ensure_datetime_index(ts)
    name = value_name or ts.name or "value"
    daily_means = ts.groupby(ts.index.dayofweek).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        range(7),
        [daily_means.get(index, np.nan) for index in range(7)],
        color="C0",
        alpha=0.85,
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("Average by Day of Week")
    ax.set_ylabel(name)

    if show:
        plt.show()
    return ax


def plot_month_pattern(
    ts: pd.Series,
    value_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot monthly averages with standard deviation error bars."""
    ts = ensure_datetime_index(ts)
    name = value_name or ts.name or "value"
    monthly_mean = ts.groupby(ts.index.month).mean()
    monthly_std = ts.groupby(ts.index.month).std()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        range(1, 13),
        [monthly_mean.get(month, np.nan) for month in range(1, 13)],
        yerr=[monthly_std.get(month, np.nan) for month in range(1, 13)],
        alpha=0.8,
        capsize=3,
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        rotation=45,
    )
    ax.set_title("Monthly Pattern")
    ax.set_ylabel(name)

    if show:
        plt.show()
    return ax


def plot_week_of_year_pattern(
    ts: pd.Series,
    value_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot average values by ISO week of year."""
    ts = ensure_datetime_index(ts)
    name = value_name or ts.name or "value"
    weeks = ts.index.isocalendar().week.astype(int)
    weekly_mean = ts.groupby(weeks).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        weekly_mean.index, weekly_mean.values, marker=".", linestyle="-", color="teal"
    )
    ax.set_title("Week of Year Pattern")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel(name)
    ax.set_xlim(1, 53)

    if show:
        plt.show()
    return ax


def plot_time_and_lag(
    df: pd.DataFrame,
    value_column: str,
    show: bool = True,
):
    """Plot time vs value and lag_1 vs value."""
    if "time" not in df.columns or "lag_1" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'time' and 'lag_1' columns for this plot"
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(df["time"], df[value_column], ".", color="0.25")
    axes[0].set_title(f"Time Plot of {value_column}")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(value_column)

    lag_df = df[["lag_1", value_column]].dropna()
    axes[1].plot(lag_df["lag_1"], lag_df[value_column], ".", color="0.25")
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Lag Plot (Shift = 1) of {value_column}")
    axes[1].set_xlabel("lag_1")
    axes[1].set_ylabel(value_column)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_time_regression(
    y: pd.Series,
    y_pred: pd.Series,
    value_name: str,
    show: bool = True,
):
    """Plot observed and fitted values over time."""
    if not isinstance(y, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y and y_pred must be pandas Series")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y.index, y.values, ".", color="0.25", label="Observed")
    ax.plot(y_pred.index, y_pred.values, linewidth=3, label="Fitted")
    ax.set_title(f"{value_name} - Observed vs Fitted")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_name)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_trend_forecast(
    observed: pd.Series,
    moving_average: pd.Series,
    fitted: pd.Series,
    forecast: pd.Series,
    value_name: str,
    show: bool = True,
):
    """Plot observed values, moving average, fitted trend, and future trend forecast."""
    if not isinstance(observed, pd.Series):
        raise TypeError("observed must be a pandas Series")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(observed.index, observed.values, ".", color="0.25", label="Observed")
    if moving_average is not None:
        ax.plot(
            moving_average.index,
            moving_average.values,
            linewidth=2,
            label="Moving Average",
        )
    ax.plot(fitted.index, fitted.values, linewidth=3, label="Trend")
    ax.plot(
        forecast.index, forecast.values, linewidth=3, color="C3", label="Trend Forecast"
    )
    ax.set_title(f"{value_name} - Trend Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_name)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_lag_regression(
    X: pd.Series,
    y: pd.Series,
    y_pred: pd.Series,
    value_name: str,
    show: bool = True,
):
    """Plot lag values against observed and predicted series."""
    if (
        not isinstance(X, pd.Series)
        or not isinstance(y, pd.Series)
        or not isinstance(y_pred, pd.Series)
    ):
        raise TypeError("X, y, and y_pred must be pandas Series")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X.values, y.loc[X.index].values, ".", color="0.25", label="Observed")
    ax.plot(X.values, y_pred.loc[X.index].values, linewidth=2, label="Predicted")
    ax.set_xlabel(X.name if X.name else "lag")
    ax.set_ylabel(value_name)
    ax.set_title(f"Lag regression: {value_name}")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_forecast_predictions(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    pred: Optional[pd.Series],
    value_name: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    """Plot train, validation, test, and prediction series on one axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if train is not None and len(train) > 0:
        ax.plot(train.index, train.values, color="blue", label="Training Data")
    if val is not None and len(val) > 0:
        ax.plot(val.index, val.values, color="orange", label="Validation Data")
    if test is not None and len(test) > 0:
        ax.plot(test.index, test.values, color="green", label="Test Data")
    if pred is not None and len(pred) > 0:
        ax.plot(pred.index, pred.values, color="red", label="Prediction")

    ax.set_title(title or f"{value_name} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_name)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


__all__ = [
    "compute_frequency_spectrum",
    "ensure_datetime_index",
    "plot_autocorrelation_diagnostics",
    "plot_day_of_week_pattern",
    "plot_forecast_predictions",
    "plot_frequency_spectrum",
    "plot_lag_regression",
    "plot_month_pattern",
    "plot_periodogram",
    "plot_time_and_lag",
    "plot_time_regression",
    "plot_trend_forecast",
    "plot_week_of_year_pattern",
    "seasonal_decomposition_plot",
    "seasonal_plot",
    "visual_inspection",
]
