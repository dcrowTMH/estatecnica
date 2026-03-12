"""Statistical testing helpers for time-series analysis."""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ADFResult:
    """Structured result for an Augmented Dickey-Fuller test."""

    adf_stat: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: Dict[str, float]
    is_stationary: bool


@dataclass(frozen=True)
class ArimaOrderSuggestion:
    """Heuristic ARIMA order suggestion based on ACF and PACF cutoffs."""

    p_suggest: int
    q_suggest: int
    acf_values: np.ndarray
    pacf_values: np.ndarray
    conf_interval: float
    n_lags_used: int


def _require_statsmodels():
    if find_spec("statsmodels") is None:
        raise ModuleNotFoundError(
            "statsmodels is required for time-series statistical tests."
        )

    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, adfuller, pacf

    return acorr_ljungbox, acf, adfuller, pacf


def stationary_test_adf(ts: pd.Series, significance: float = 0.05) -> ADFResult:
    """Run the Augmented Dickey-Fuller test on a time series."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    if ts_clean.size < 3:
        raise ValueError("Time series is too short for ADF test (need >= 3 values)")

    try:
        _, _, adfuller, _ = _require_statsmodels()
        result = adfuller(ts_clean)
    except Exception as exc:
        raise RuntimeError(f"ADF test failed: {exc}") from exc

    return ADFResult(
        adf_stat=float(result[0]),
        pvalue=float(result[1]),
        usedlag=int(result[2]),
        nobs=int(result[3]),
        critical_values={k: float(v) for k, v in result[4].items()},
        is_stationary=bool(result[1] <= significance),
    )


def autocorrelation_ljung_box(
    ts: pd.Series,
    lags: Union[int, Sequence[int]] = 10,
    return_df: bool = True,
) -> Any:
    """Run the Ljung-Box test for autocorrelation."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    if ts_clean.size < 2:
        raise ValueError("Time series is too short for Ljung-Box test")

    try:
        acorr_ljungbox, _, _, _ = _require_statsmodels()
        return acorr_ljungbox(ts_clean, lags=lags, return_df=return_df)
    except Exception as exc:
        raise RuntimeError(f"Ljung-Box test failed: {exc}") from exc


def suggest_cutoff(
    values: np.ndarray,
    conf_interval: float,
    consecutive_insignificant: int = 2,
) -> int:
    """Suggest a cutoff lag after the first insignificant sequence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0

    for index in range(values.size):
        if index + consecutive_insignificant > values.size:
            break
        window = values[index : index + consecutive_insignificant]
        if np.all(np.abs(window) <= conf_interval):
            return int(index)

    return 1


def suggest_arima_orders(
    ts: pd.Series,
    max_lags: int = 40,
    fft: bool = False,
    conf_multiplier: float = 1.96,
    consecutive_insignificant: int = 2,
) -> ArimaOrderSuggestion:
    """Suggest initial ARIMA p and q values from PACF and ACF cutoffs."""
    if not isinstance(ts, pd.Series):
        raise TypeError("ts must be a pandas Series")

    ts_clean = ts.dropna()
    n_obs = len(ts_clean)
    if n_obs < 5:
        raise ValueError("Time series is too short to suggest ARIMA orders reliably")

    n_lags = min(max_lags, max(1, n_obs // 2 - 1))
    conf_interval = conf_multiplier / np.sqrt(n_obs)
    _, acf, _, pacf = _require_statsmodels()

    try:
        acf_values = acf(ts_clean, nlags=n_lags, fft=fft)
    except Exception as exc:
        raise RuntimeError(f"ACF computation failed: {exc}") from exc

    try:
        pacf_values = pacf(ts_clean, nlags=n_lags, method="ywm")
    except Exception:
        pacf_values = pacf(ts_clean, nlags=n_lags)

    q_suggest = suggest_cutoff(
        acf_values[1:] if len(acf_values) > 1 else np.array([]),
        conf_interval,
        consecutive_insignificant,
    )
    p_suggest = suggest_cutoff(
        pacf_values[1:] if len(pacf_values) > 1 else np.array([]),
        conf_interval,
        consecutive_insignificant,
    )

    return ArimaOrderSuggestion(
        p_suggest=int(p_suggest),
        q_suggest=int(q_suggest),
        acf_values=np.asarray(acf_values, dtype=float),
        pacf_values=np.asarray(pacf_values, dtype=float),
        conf_interval=float(conf_interval),
        n_lags_used=int(n_lags),
    )


def format_adf_result(result: ADFResult) -> Dict[str, Any]:
    """Convert an ADF result dataclass into a plain dictionary."""
    return {
        "adf_stat": result.adf_stat,
        "pvalue": result.pvalue,
        "usedlag": result.usedlag,
        "nobs": result.nobs,
        "critical_values": result.critical_values,
        "is_stationary": result.is_stationary,
    }


__all__ = [
    "ADFResult",
    "ArimaOrderSuggestion",
    "autocorrelation_ljung_box",
    "format_adf_result",
    "stationary_test_adf",
    "suggest_arima_orders",
    "suggest_cutoff",
]
