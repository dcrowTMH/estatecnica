"""Synthetic time-series generation helpers."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def generate_arima_values(
    n_samples: int,
    d: int = 0,
    p_coeffs: Sequence[float] = (),
    q_coeffs: Sequence[float] = (),
    drift: float = 0.0,
    seed: int = 42,
    burn_in: int = 100,
) -> np.ndarray:
    """Generate synthetic ARIMA-like values with simple AR and MA terms."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if d < 0:
        raise ValueError("d must be non-negative")

    p_coeffs = list(p_coeffs)
    q_coeffs = list(q_coeffs)
    p_order = len(p_coeffs)
    q_order = len(q_coeffs)
    total_samples = n_samples + burn_in

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=total_samples)
    series = np.zeros(total_samples, dtype=float)

    for index in range(max(p_order, q_order), total_samples):
        ar_term = (
            sum(
                p_coeffs[offset] * series[index - offset - 1]
                for offset in range(p_order)
            )
            if p_order
            else 0.0
        )
        ma_term = (
            sum(
                q_coeffs[offset] * noise[index - offset - 1]
                for offset in range(q_order)
            )
            if q_order
            else 0.0
        )
        series[index] = ar_term + ma_term + noise[index]

    values = series[burn_in:]
    for _ in range(d):
        values = np.cumsum(values)

    if drift != 0:
        values = values + np.arange(n_samples) * drift

    return values


def values_to_dataframe(
    values: Sequence[float], start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """Convert a sequence of values to a daily date-indexed DataFrame."""
    dates = pd.date_range(start=start_date, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "value": values})


def generate_arima_dataframe(
    start_date: str = "2020-01-01",
    n_samples: int = 500,
    d: int = 0,
    p_coeffs: Sequence[float] = (),
    q_coeffs: Sequence[float] = (),
    drift: float = 0.0,
    seed: int = 42,
    burn_in: int = 100,
) -> pd.DataFrame:
    """Generate ARIMA-like synthetic data as a DataFrame."""
    values = generate_arima_values(
        n_samples=n_samples,
        d=d,
        p_coeffs=p_coeffs,
        q_coeffs=q_coeffs,
        drift=drift,
        seed=seed,
        burn_in=burn_in,
    )
    return values_to_dataframe(values, start_date=start_date)


@dataclass
class ARIMASimulator:
    """Convenience wrapper for generating synthetic ARIMA-like time series."""

    start_date: str = "2020-01-01"
    n_samples: int = 500
    seed: int = 42
    burn_in: int = 100

    def generate_arima_data(
        self,
        d: int = 0,
        p_coeffs: Sequence[float] = (),
        q_coeffs: Sequence[float] = (),
        drift: float = 0.0,
    ) -> pd.DataFrame:
        """Generate a DataFrame with `date` and `value` columns."""
        return generate_arima_dataframe(
            start_date=self.start_date,
            n_samples=self.n_samples,
            d=d,
            p_coeffs=p_coeffs,
            q_coeffs=q_coeffs,
            drift=drift,
            seed=self.seed,
            burn_in=self.burn_in,
        )


__all__ = [
    "ARIMASimulator",
    "generate_arima_dataframe",
    "generate_arima_values",
    "values_to_dataframe",
]
