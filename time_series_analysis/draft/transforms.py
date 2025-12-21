# Python Code/estatecnica/time_series_analysis/draft/transforms.py
"""
Draft transforms helper utilities for time series analysis.

This module provides functions to:
- normalize inputs into a pandas Series with a DatetimeIndex
- resample/aggregate series with common methods
- create daily/weekly/monthly/yearly aggregates
- split a series into chronological train/val/test splits

These are lightweight, well-documented helpers intended to be imported
by an analyzer or forecasting module. This is a 'draft' module — keep
function signatures stable and return pandas objects for easy chaining.
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

AggMethod = Literal["mean", "sum", "median", "min", "max", "std", "count"]


def to_series(
    data: Union[pd.Series, pd.DataFrame],
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    copy: bool = True,
) -> pd.Series:
    """
    Normalize input into a pandas Series with a DatetimeIndex.

    Args:
        data: pd.Series (preferred) or pd.DataFrame.
        date_col: if `data` is a DataFrame and the date is in a column, provide its name.
        value_col: if `data` is a DataFrame, provide the column name that contains values;
                   if None, the first numeric column is used.
        copy: whether to operate on a copy of the data

    Returns:
        pd.Series with a pd.DatetimeIndex and values.

    Raises:
        TypeError if input types are unsupported.
        ValueError if a datetime index cannot be established or value column cannot be found.
    """
    if isinstance(data, pd.Series):
        series = data.copy() if copy else data
        # Try to ensure index is datetime
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                raise ValueError(
                    "Series index is not datetime-like and could not be parsed."
                )
        return series

    if isinstance(data, pd.DataFrame):
        df = data.copy() if copy else data
        # If a date_col provided, use it for index
        if date_col is not None:
            if date_col not in df.columns:
                raise ValueError(f"date_col '{date_col}' not found in DataFrame")
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df = df.set_index(date_col)
        else:
            # If index is datetime, keep it
            if isinstance(df.index, pd.DatetimeIndex):
                pass
            else:
                # try to infer a datetime-like column automatically
                candidate = None
                for col in df.columns:
                    if np.issubdtype(df[col].dtype, np.datetime64):
                        candidate = col
                        break
                if candidate is not None:
                    df[candidate] = pd.to_datetime(df[candidate])
                    df = df.sort_values(candidate).reset_index(drop=True)
                    df = df.set_index(candidate)
                else:
                    raise ValueError(
                        "Could not determine a datetime index. Provide `date_col`."
                    )

        # determine value column
        if value_col is None:
            # choose first column that is numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                # fallback: first column
                value_col = df.columns[0]
            else:
                value_col = numeric_cols[0]
        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in DataFrame")
        series = df[value_col].copy() if copy else df[value_col]
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("After processing, series index is not DatetimeIndex")
        return series

    raise TypeError("Input must be a pandas Series or DataFrame")


def _resample_method(ts: pd.Series, freq: str, method: AggMethod) -> pd.Series:
    """
    Internal utility to resample a time series using named aggregation methods.

    Args:
        ts: pd.Series with DatetimeIndex
        freq: pandas resample frequency string (e.g., 'D', 'W-SUN', 'MS', 'YS')
        method: aggregation method name

    Returns:
        resampled pd.Series
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

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


def daily_to_weekly_and_yearly(
    ts: Union[pd.Series, pd.DataFrame],
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    method: AggMethod = "mean",
    week_start: Literal["Monday", "Sunday"] = "Monday",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Convert a daily series into daily, weekly, monthly, and yearly aggregated series.

    This function is tolerant of both Series and DataFrame inputs and uses `to_series`
    to normalize the input.

    Args:
        ts: pd.Series or pd.DataFrame
        date_col: optional date column if passing a DataFrame
        value_col: optional value column if passing a DataFrame
        method: aggregation method used for resampling
        week_start: 'Monday' or 'Sunday' defines weekly anchor (W-MON or W-SUN)

    Returns:
        tuple of (daily, weekly, monthly, yearly) pd.Series. Frequencies:
            - daily: 'D'
            - weekly: 'W-MON' or 'W-SUN'
            - monthly: 'MS' (month start)
            - yearly: 'YS' (year start)
    """
    s = to_series(ts, date_col=date_col, value_col=value_col)

    freq_weekly = "W-SUN" if week_start == "Sunday" else "W-MON"
    freq_monthly = "MS"
    freq_yearly = "YS"
    freq_daily = "D"

    daily = _resample_method(s, freq_daily, method)
    weekly = _resample_method(s, freq_weekly, method)
    monthly = _resample_method(s, freq_monthly, method)
    yearly = _resample_method(s, freq_yearly, method)

    return daily, weekly, monthly, yearly


def split_time_series(
    ts: Union[pd.Series, pd.DataFrame],
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split a time series (chronologically) into train/validation/test parts.

    If a DataFrame is passed, the function will convert it to a Series using `to_series`.

    Args:
        ts: pd.Series or pd.DataFrame with DatetimeIndex
        train_pct/val_pct/test_pct: fractions that must sum to 1.0

    Returns:
        (train, val, test) pandas Series

    Raises:
        ValueError if percentages don't sum to 1 or if series is too short.
    """
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-9:
        raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

    s = to_series(ts) if not isinstance(ts, pd.Series) else ts
    n = len(s)
    if n == 0:
        raise ValueError("Time series is empty")

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = s.iloc[:train_end]
    val = s.iloc[train_end:val_end]
    test = s.iloc[val_end:]

    return train, val, test


# Expose a compact public API for the draft module
__all__ = [
    "to_series",
    "daily_to_weekly_and_yearly",
    "split_time_series",
    "_resample_method",
]
