"""Input normalization and time-series transformation helpers."""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

AggMethod = Literal["mean", "sum", "median", "min", "max", "std", "count"]
TimeSeriesInput = Union[pd.Series, pd.DataFrame]


def to_series(
    data: TimeSeriesInput,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    copy: bool = True,
) -> pd.Series:
    """Normalize a Series or DataFrame into a Series with a DatetimeIndex."""
    if isinstance(data, pd.Series):
        series = data.copy() if copy else data
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception as exc:
                raise ValueError(
                    "Series index is not datetime-like and could not be parsed."
                ) from exc
        return series.sort_index()

    if isinstance(data, pd.DataFrame):
        df = data.copy() if copy else data

        if date_col is not None:
            if date_col not in df.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame")
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True).set_index(date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            candidate = None
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.datetime64):
                    candidate = col
                    break
            if candidate is None:
                raise ValueError(
                    "Could not determine a datetime index. Provide `date_col`."
                )
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.sort_values(candidate).reset_index(drop=True).set_index(candidate)

        if value_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = numeric_cols[0] if numeric_cols else df.columns[0]

        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in DataFrame")

        series = df[value_col].copy() if copy else df[value_col]
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("After processing, series index is not DatetimeIndex")
        return series.sort_index()

    raise TypeError("Input must be a pandas Series or DataFrame")


def resample_series(ts: pd.Series, freq: str, method: AggMethod) -> pd.Series:
    """Resample a time series using a named aggregation method."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise TypeError("ts must use a DatetimeIndex")

    if method == "mean":
        return ts.resample(freq).mean()
    if method == "sum":
        return ts.resample(freq).sum()
    if method == "median":
        return ts.resample(freq).median()
    if method == "min":
        return ts.resample(freq).min()
    if method == "max":
        return ts.resample(freq).max()
    if method == "std":
        return ts.resample(freq).std()
    if method == "count":
        return ts.resample(freq).count()

    raise ValueError(f"Unknown aggregation method: {method}")


def calendar_aggregates(
    data: TimeSeriesInput,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    method: AggMethod = "mean",
    week_start: Literal["Monday", "Sunday"] = "Monday",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return daily, weekly, monthly, and yearly aggregates for a time series."""
    series = to_series(data, date_col=date_col, value_col=value_col)
    weekly_freq = "W-SUN" if week_start == "Sunday" else "W-MON"

    daily = resample_series(series, "D", method)
    weekly = resample_series(series, weekly_freq, method)
    monthly = resample_series(series, "MS", method)
    yearly = resample_series(series, "YS", method)

    return daily, weekly, monthly, yearly


def daily_to_weekly_and_yearly(
    data: TimeSeriesInput,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    method: AggMethod = "mean",
    week_start: Literal["Monday", "Sunday"] = "Monday",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Backward-compatible alias for calendar aggregates."""
    return calendar_aggregates(
        data,
        date_col=date_col,
        value_col=value_col,
        method=method,
        week_start=week_start,
    )


def split_time_series(
    data: TimeSeriesInput,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Split a time series chronologically into train, validation, and test sets."""
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-9:
        raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

    series = to_series(data) if not isinstance(data, pd.Series) else to_series(data)
    if len(series) == 0:
        raise ValueError("Time series is empty")

    train_end = int(len(series) * train_pct)
    val_end = int(len(series) * (train_pct + val_pct))

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    return train, val, test


__all__ = [
    "AggMethod",
    "TimeSeriesInput",
    "calendar_aggregates",
    "daily_to_weekly_and_yearly",
    "resample_series",
    "split_time_series",
    "to_series",
]
