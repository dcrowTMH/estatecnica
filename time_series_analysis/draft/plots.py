```Python Code/estatecnica/time_series_analysis/draft/plots.py#L1-300
"""
Draft plotting helper utilities for time series analysis.

This module provides a set of focused plotting functions that operate on
pandas Series (time series) or lightweight derived series (monthly/weekly).
Each function is deliberately small and returns the matplotlib Axes
object to make testing and composition easier.

Design goals:
- Accept a pd.Series with a DatetimeIndex.
- Avoid side-effects where reasonable (provide `show` flag).
- Return axes for further customization or testing.
- Keep visual defaults simple and consistent with seaborn/matplotlib.
"""

from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram
from scipy.fft import fft, fftfreq

# consistent plotting defaults used across functions
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)


def _ensure_datetime_index(ts: pd.Series) -> pd.Series:
    """Validate and coerce the input to a pd.Series with a DatetimeIndex."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        try:
            ts = ts.copy()
            ts.index = pd.to_datetime(ts.index)
        except Exception:
            raise ValueError("Series index must be datetime-like or convertible to datetime")
    return ts


def visual_inspection(
    ts: pd.Series,
    value_name: Optional[str] = None,
    monthly: Optional[pd.Series] = None,
    weekly: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
) -> plt.Axes:
    """
    Create a 2x2 set of exploratory plots:
      - time series (monthly)
      - monthly boxplot (distribution by month)
      - day-of-week boxplot (weekly buckets)
      - histogram of raw values

    Args:
        ts: daily time series pd.Series.
        value_name: label used on axes (falls back to ts.name or 'value').
        monthly: optional precomputed monthly series. If None, computed from ts.
        weekly: optional precomputed weekly series. If None, computed from ts.
        figsize: figure size.
        show: whether to call plt.show().

    Returns:
        The top-level Axes array object (2x2 numpy array of Axes).
    """
    ts = _ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    if monthly is None:
        monthly = ts.resample("MS").mean()
    if weekly is None:
        weekly = ts.resample("W-MON").mean()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Time series (monthly)
    axes[0, 0].plot(monthly.index, monthly.values, color="C0")
    axes[0, 0].set_title("Time Series (Monthly)")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel(name)

    # Monthly boxplot (distribution by month)
    # build monthly groups for boxplot: group by month number using the original daily data
    month_groups = ts.groupby(ts.index.month).apply(list)
    axes[0, 1].boxplot([month_groups.get(m, []) for m in range(1, 13)], patch_artist=True)
    axes[0, 1].set_title("Monthly Distribution")
    axes[0, 1].set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=45,
    )
    axes[0, 1].set_ylabel(name)

    # Day-of-week boxplot (use daily values grouped by dayofweek)
    dow_groups = [ts[ts.index.dayofweek == i].values for i in range(7)]
    axes[1, 0].boxplot(dow_groups, patch_artist=True)
    axes[1, 0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1, 0].set_title("Day-of-Week Distribution")
    axes[1, 0].set_ylabel(name)

    # Histogram of values
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
    palette: Optional[Sequence[str]] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Create a seasonal lineplot across a chosen period.

    Examples:
      - period='week', freq='day' produces one line per week across days of week
      - period='year', freq='dayofyear' produces lines per year across days

    Args:
        ts: time series pd.Series
        period: 'week' or 'year' (controls the hue grouping)
        freq: 'day' (dayofweek) or 'dayofyear'
        value_name: label for the y-axis (falls back to ts.name)
        ax: optional existing axis to draw on
        palette: optional palette to pass to seaborn
        show: whether to call plt.show()

    Returns:
        matplotlib Axes
    """
    ts = _ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    df = ts.to_frame(name=name)
    if period == "week":
        df["period"] = df.index.isocalendar().week
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

    if palette is None:
        palette = sns.color_palette("husl", n_colors=df["period"].nunique())

    sns.lineplot(x="freq", y=name, hue="period", data=df, ci=None, ax=ax, palette=palette, legend=False)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    ax.set_xlabel(freq)
    ax.set_ylabel(name)

    # annotate the end of each line with its period label
    # this is a best-effort approach and may overlap if many lines are present
    for line, lab in zip(ax.lines, df["period"].unique()):
        try:
            y_ = line.get_ydata()[-1]
            ax.annotate(
                str(lab),
                xy=(1, y_),
                xytext=(6, 0),
                color=line.get_color(),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=10,
                va="center",
            )
        except Exception:
            continue

    if show:
        plt.show()
    return ax


def plot_periodogram(
    ts: pd.Series,
    detrend: str = "linear",
    fs_days_per_year: float = 365.0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a periodogram (variance spectrum) using scipy.signal.periodogram.

    Args:
        ts: pd.Series (datetime index).
        detrend: detrend option forwarded to periodogram.
        fs_days_per_year: used to scale frequency units; typical value 365.
        ax: optional matplotlib Axes
        show: whether to call plt.show()

    Returns:
        matplotlib Axes
    """
    ts = _ensure_datetime_index(ts)

    # approximate sampling frequency: samples/day
    # if the series is daily, fs = 1 sample/day; scale to "per year" if needed
    fs = 1.0  # samples per day for daily series
    freqs, power = periodogram(ts.values, fs=fs, detrend=detrend, window="boxcar", scaling="spectrum")

    # Show positive frequencies only
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    power = power[pos_mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.step(freqs, power, color="purple")
    ax.set_xscale("log")
    # use human-readable xticks that map approximate cycles per year/month/week
    ticks = np.array([1 / 365.0, 1 / 180.0, 1 / 90.0, 1 / 30.0, 1 / 7.0, 1 / 1.0])
    labels = ["Annual", "Semiannual", "Quarterly", "Monthly", "Weekly", "Daily"]
    # keep only ticks within the range
    ticks = ticks[(ticks >= freqs.min()) & (ticks <= freqs.max())]
    labels = [lab for t, lab in zip(ticks, labels)][: len(ticks)]
    if len(ticks) > 0:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram (power spectrum)")

    if show:
        plt.show()
    return ax


def day_of_week_pattern(ts: pd.Series, value_name: Optional[str] = None, show: bool = True) -> plt.Axes:
    """
    Plot average values by day of week and perform a one-way ANOVA across weekdays.

    Returns the axis and prints short summary of ANOVA test.
    """
    ts = _ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    df = ts.to_frame(name=name)
    df["dow"] = df.index.dayofweek
    daily_means = df.groupby("dow")[name].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(7), daily_means.values, color="C0", alpha=0.85)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title("Average by Day of Week")
    ax.set_ylabel(name)

    # ANOVA: prepare groups and run test (skip empty groups)
    groups = [df[df["dow"] == i][name].values for i in range(7) if len(df[df["dow"] == i]) > 0]
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"Day-of-Week ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        except Exception as exc:
            print(f"ANOVA failed: {exc}")
    else:
        print("Not enough groups to perform ANOVA (need at least two non-empty groups).")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


def month_pattern(ts: pd.Series, value_name: Optional[str] = None, show: bool = True) -> plt.Axes:
    """
    Plot monthly averages with error bars (std) and perform an ANOVA across months.
    """
    ts = _ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    df = ts.to_frame(name=name)
    df["month"] = df.index.month
    monthly_mean = df.groupby("month")[name].mean()
    monthly_std = df.groupby("month")[name].std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, 13), monthly_mean.values, yerr=monthly_std.values, alpha=0.8, capsize=3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    ax.set_title("Monthly Pattern")
    ax.set_ylabel(name)

    # ANOVA across months
    groups = [df[df["month"] == m][name].values for m in range(1, 13) if len(df[df["month"] == m]) > 0]
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"Monthly ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        except Exception as exc:
            print(f"ANOVA failed: {exc}")
    else:
        print("Not enough data to perform monthly ANOVA.")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


def week_of_year_pattern(ts: pd.Series, value_name: Optional[str] = None, show: bool = True) -> plt.Axes:
    """
    Plot and test for patterns across weeks of the year.
    """
    ts = _ensure_datetime_index(ts)
    name = value_name or ts.name or "value"

    df = ts.to_frame(name=name)
    # isocalendar().week returns an int-like object (pandas >=1.1 returns DataFrame)
    df["week_of_year"] = df.index.isocalendar().week

    weekly_mean = df.groupby("week_of_year")[name].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_mean.index, weekly_mean.values, marker=".", linestyle="-", color="teal")
    ax.set_title("Week of Year Pattern")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel(name)
    ax.set_xlim(1, 53)

    # ANOVA
    unique_weeks = sorted(df["week_of_year"].unique())
    groups = [df[df["week_of_year"] == w][name].values for w in unique_weeks if len(df[df["week_of_year"] == w]) > 0]
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"Week-of-Year ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        except Exception as exc:
            print(f"ANOVA failed: {exc}")
    else:
        print("Not enough weekly groups for ANOVA.")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


def frequency_domain_analysis(ts: pd.Series, frequency: float = 1.0, show: bool = True) -> plt.Axes:
    """
    Plot a power spectrum (FFT) using numpy. Returns the plot Axes.

    Args:
        ts: pd.Series
        frequency: sampling frequency (e.g., 1 sample/day -> frequency=1.0)
        show: whether to call plt.show()
    """
    ts = _ensure_datetime_index(ts)
    vals = ts.values
    n = len(vals)
    if n < 2:
        raise ValueError("Time series too short for FFT")

    fft_vals = fft(vals)
    freqs = fftfreq(n, d=1.0 / frequency)
    power = np.abs(fft_vals) ** 2

    pos = freqs > 0
    pos_freqs = freqs[pos]
    pos_power = power[pos]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pos_freqs, pos_power, linewidth=1.0)
    ax.set_title("Power Spectrum (FFT)")
    ax.set_xlabel("Frequency (cycles per time unit)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

    # Show top few dominant periods (convert freq -> period if freq>0)
    if pos_power.size:
        top_k = min(5, pos_power.size)
        idx = np.argsort(pos_power)[-top_k:]
        dominant_freqs = pos_freqs[idx]
        # handle divide-by-zero safety
        dominant_periods = np.array([1.0 / f if f != 0 else np.nan for f in dominant_freqs])
        print("Dominant periods (in time units):", np.sort(dominant_periods))

    plt.tight_layout()
    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------
# Forecast / regression plotting helpers
# These functions were moved here from the draft forecast module so plotting
# logic is centralized and reusable by analyzer and forecast classes.
# Each function returns the (fig, ax) pair or axes so callers can test them.
# ---------------------------------------------------------------------
def plot_time_and_lag(df: pd.DataFrame, value_column: str, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time vs value with regression and lag_1 vs value with regression.
    Accepts a DataFrame that contains 'time', 'lag_1', and the target `value_column`.
    Returns (fig, axes) where axes is a 2-element array.
    """
    if "time" not in df.columns or "lag_1" not in df.columns:
        raise ValueError("DataFrame must contain 'time' and 'lag_1' columns for this plot")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df["time"], df[value_column], color="0.75")
    sns.regplot(x="time", y=value_column, data=df, ci=None, scatter_kws=dict(color="0.25"), ax=axes[0])
    axes[0].set_title(f"Time Plot of {value_column}")

    axes[1].plot(df["lag_1"], df[value_column], ".", color="0.25")
    sns.regplot(x="lag_1", y=value_column, data=df, ci=None, scatter_kws=dict(color="0.25"), ax=axes[1])
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Lag Plot (Shift = 1) of {value_column}")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_time_regression(y: pd.Series, y_pred: pd.Series, value_name: str, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed series `y` and fitted values `y_pred` on the same axes.
    """
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


def plot_lag_regression(X: pd.Series, y: pd.Series, y_pred: pd.Series, value_name: str, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot lag (X) against observed (y) and predicted (y_pred).
    `X` should be a pandas Series containing lag values (e.g., lag_1).
    """
    if not isinstance(X, pd.Series) or not isinstance(y, pd.Series) or not isinstance(y_pred, pd.Series):
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
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training, validation, test series and predictions on a single axes.

    Returns (fig, ax).
    """
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

    if title:
        ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_name)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_fourier_forecast(y: pd.Series, y_pred: pd.Series, y_fore: pd.Series, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed seasonal fit and forecast produced by a fourier/deterministic design.

    y      : historical observed series
    y_pred : fitted values on the training period
    y_fore : forecasted values (future index)
    """
    if not isinstance(y, pd.Series) or not isinstance(y_pred, pd.Series) or not isinstance(y_fore, pd.Series):
        raise TypeError("y, y_pred, and y_fore must be pandas Series")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y.index, y.values, ".", color="0.25", label="Observed")
    ax.plot(y_pred.index, y_pred.values, linewidth=2, label="Seasonal Fit")
    ax.plot(y_fore.index, y_fore.values, linewidth=2, label="Seasonal Forecast", color="C3")
    ax.set_title("Fourier Seasonal Fit & Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(y.name if y.name else "value")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
