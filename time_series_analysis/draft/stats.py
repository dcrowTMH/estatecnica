# Python Code/estatecnica/time_series_analysis/draft/stats.py

"""
Lightweight stats helper utilities for time series analysis (draft).

This module provides:
- ADF test wrapper that returns a structured result
- Ljung-Box wrapper
- Utilities to compute ACF/PACF and suggest AR/MA orders based on cutoff heuristics

The functions return data structures (dataclasses / dicts) rather than printing,
so they are easier to test and to use from other modules or notebooks.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

# These imports assume the user's environment has statsmodels installed.
# If not available in a draft environment, callers should handle ImportError.
from statsmodels.tsa.stattools import acf, adfuller, pacf


@dataclass
class ADFResult:
    adf_stat: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: Dict[str, float]
    is_stationary: bool


def stationary_test_adf(ts: pd.Series, significance: float = 0.05) -> ADFResult:
    """
    Run the Augmented Dickey-Fuller (ADF) test on a time series.

    Args:
        ts: Time series (pd.Series) with a datetime index or numeric index.
        significance: significance level to decide stationarity (default 0.05).

    Returns:
        ADFResult dataclass describing the test output.

    Raises:
        ValueError if the series is too short or contains no non-NaN values.
        Any exception raised by statsmodels.adfuller will propagate after being wrapped.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    if ts_clean.size < 3:
        raise ValueError(
            "Time series is too short for ADF test (need >= 3 non-NaN values)"
        )

    try:
        result = adfuller(ts_clean)
    except Exception as exc:
        raise RuntimeError(f"ADF test failed: {exc}") from exc

    adf_stat, pvalue, usedlag, nobs, critical_values_dict = (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
    )
    is_stationary = pvalue <= significance

    return ADFResult(
        adf_stat=float(adf_stat),
        pvalue=float(pvalue),
        usedlag=int(usedlag),
        nobs=int(nobs),
        critical_values={k: float(v) for k, v in critical_values_dict.items()},
        is_stationary=bool(is_stationary),
    )


def autocorrelation_ljung_box(
    ts: pd.Series, lags: Union[int, Tuple[int, ...]] = 10, return_df: bool = True
) -> Any:
    """
    Run the Ljung-Box test for autocorrelation.

    Args:
        ts: time series (pd.Series).
        lags: integer or tuple/list of lag integers to compute.
        return_df: if True, returns a pandas.DataFrame (statsmodels behavior),
                   otherwise returns structured arrays.

    Returns:
        The object returned by statsmodels.stats.diagnostic.acorr_ljungbox:
        - DataFrame when return_df=True
        - tuple of arrays when return_df=False

    Raises:
        TypeError/ValueError for invalid inputs.
        RuntimeError if the underlying test fails.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    if ts_clean.size < 2:
        raise ValueError("Time series is too short for Ljung-Box test")

    try:
        lb_res = acorr_ljungbox(ts_clean, lags=lags, return_df=return_df)
    except Exception as exc:
        raise RuntimeError(f"Ljung-Box test failed: {exc}") from exc

    return lb_res


def _suggest_cutoff(
    values: np.ndarray, conf_interval: float, consecutive_insignificant: int = 2
) -> int:
    """
    Internal helper: suggest a cutoff lag index given a sequence of acf/pacf values
    (lag-0 should typically be excluded before calling this).

    Heuristic:
    - Find the first index i such that values[i:i+consecutive_insignificant] are all
      within +/- conf_interval. Return i (which corresponds to lag = i).

    Args:
        values: 1-D numpy array of ACF/PACF values (lag-0 excluded).
        conf_interval: threshold for significance, e.g. 1.96/sqrt(N).
        consecutive_insignificant: how many consecutive insignificant lags to require.

    Returns:
        int: suggested cutoff lag (>= 0). Defaults to 1 if no clear cutoff is found.
    """
    # Defensive cast
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0

    for i in range(n):
        if i + consecutive_insignificant > n:
            break
        seq = values[i : i + consecutive_insignificant]
        if np.all(np.abs(seq) <= conf_interval):
            # Return the lag corresponding to position 'i' (since lag-0 was excluded)
            return int(i)
    return 1


def suggest_arima_orders(
    ts: pd.Series,
    max_lags: int = 40,
    fft: bool = False,
    conf_multiplier: float = 1.96,
    consecutive_insignificant: int = 2,
) -> Dict[str, Any]:
    """
    Compute ACF/PACF and suggest initial AR (p) and MA (q) orders for ARIMA.

    Args:
        ts: input time series (pd.Series).
        max_lags: maximum number of lags to compute (caps at len(ts)//2 - 1).
        fft: whether to use FFT-based acf computation (fast).
        conf_multiplier: multiplier for the confidence interval (1.96 ~ 95%).
        consecutive_insignificant: used by the cutoff heuristic.

    Returns:
        dict with keys:
          - 'p_suggest': suggested p (from PACF cutoff)
          - 'q_suggest': suggested q (from ACF cutoff)
          - 'acf': full acf array (including lag-0)
          - 'pacf': full pacf array (including lag-0)
          - 'conf_interval': computed conf interval used
          - 'n_lags_used': nlags used in computation

    Notes:
        The suggestions are heuristics (starting points). Always validate via model diagnostics.
    """
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    n = len(ts_clean)
    if n < 5:
        raise ValueError("Time series is too short to suggest ARIMA orders reliably")

    # Choose number of lags
    n_lags = min(max_lags, max(1, n // 2 - 1))
    if n_lags < 1:
        n_lags = 1

    # Confidence interval for ACF/PACF sampling variability (approx)
    conf_interval = conf_multiplier / np.sqrt(n)

    try:
        acf_vals = acf(ts_clean, nlags=n_lags, fft=fft)
    except Exception as exc:
        raise RuntimeError(f"ACF computation failed: {exc}") from exc

    try:
        pacf_vals = pacf(ts_clean, nlags=n_lags, method="ywm")
    except Exception:
        # fallback to default pacf method if ywm fails for some inputs
        pacf_vals = pacf(ts_clean, nlags=n_lags)

    # exclude lag-0 when applying the cutoff heuristic
    acf_tail = acf_vals[1:] if len(acf_vals) > 1 else np.array([])
    pacf_tail = pacf_vals[1:] if len(pacf_vals) > 1 else np.array([])

    q_suggest = _suggest_cutoff(acf_tail, conf_interval, consecutive_insignificant)
    p_suggest = _suggest_cutoff(pacf_tail, conf_interval, consecutive_insignificant)

    return {
        "p_suggest": int(p_suggest),
        "q_suggest": int(q_suggest),
        "acf": np.asarray(acf_vals, dtype=float),
        "pacf": np.asarray(pacf_vals, dtype=float),
        "conf_interval": float(conf_interval),
        "n_lags_used": int(n_lags),
    }


# Optional tiny CLI-friendly formatter utilities (non-printing core functions)
def format_adf_result(result: ADFResult) -> Dict[str, Any]:
    """
    Convert ADFResult dataclass to a plain dict (useful for JSON serialization or logging).
    """
    return {
        "adf_stat": result.adf_stat,
        "pvalue": result.pvalue,
        "usedlag": result.usedlag,
        "nobs": result.nobs,
        "critical_values": result.critical_values,
        "is_stationary": result.is_stationary,
    }
