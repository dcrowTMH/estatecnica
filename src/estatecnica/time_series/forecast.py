"""Forecasting utilities for time-series workflows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .analysis import TimeSeriesAnalyzer
from .plotting import plot_forecast_predictions
from .stats_tests import stationary_test_adf
from .transforms import split_time_series


@dataclass(frozen=True)
class ForecastResult:
    """Prediction series and error summary for a forecast method."""

    method: str
    predictions: pd.Series
    rmse: float


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Return RMSE between two aligned series."""
    if len(actual) == 0 or len(predicted) == 0:
        return float("nan")
    actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
    if len(actual_aligned) == 0:
        return float("nan")
    error = actual_aligned.to_numpy(dtype=float) - predicted_aligned.to_numpy(
        dtype=float
    )
    return float(np.sqrt(np.mean(error**2)))


def naive_forecast(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Predict all future values as the last observed training value."""
    if len(train) == 0:
        predictions = pd.Series([], index=test.index, dtype=float)
        return ForecastResult("naive", predictions, float("nan"))
    predictions = pd.Series(train.iloc[-1], index=test.index, dtype=float)
    return ForecastResult(
        "naive", predictions, root_mean_squared_error(test, predictions)
    )


def average_forecast(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Predict all future values as the mean of the training set."""
    if len(train) == 0:
        predictions = pd.Series([], index=test.index, dtype=float)
        return ForecastResult("average", predictions, float("nan"))
    predictions = pd.Series(float(train.mean()), index=test.index, dtype=float)
    return ForecastResult(
        "average", predictions, root_mean_squared_error(test, predictions)
    )


def drift_forecast(train: pd.Series, test: pd.Series) -> ForecastResult:
    """Forecast with a drift estimated from the full training window."""
    if len(train) < 2:
        return naive_forecast(train, test)
    slope = (train.iloc[-1] - train.iloc[0]) / max(1, len(train) - 1)
    horizon = np.arange(1, len(test) + 1)
    predictions = pd.Series(
        train.iloc[-1] + horizon * slope, index=test.index, dtype=float
    )
    return ForecastResult(
        "drift", predictions, root_mean_squared_error(test, predictions)
    )


def seasonal_naive_forecast(
    train: pd.Series,
    test: pd.Series,
    seasonal_period: int = 7,
) -> ForecastResult:
    """Repeat the last observed season across the forecast horizon."""
    if len(train) == 0:
        predictions = pd.Series([], index=test.index, dtype=float)
        return ForecastResult("seasonal_naive", predictions, float("nan"))
    if len(train) < seasonal_period:
        fallback = naive_forecast(train, test)
        return ForecastResult("seasonal_naive", fallback.predictions, fallback.rmse)

    last_season = train.iloc[-seasonal_period:].to_numpy(dtype=float)
    repeated = np.tile(last_season, int(np.ceil(len(test) / seasonal_period)))[
        : len(test)
    ]
    predictions = pd.Series(repeated, index=test.index, dtype=float)
    return ForecastResult(
        "seasonal_naive", predictions, root_mean_squared_error(test, predictions)
    )


def recent_drift_forecast(
    train: pd.Series,
    test: pd.Series,
    window: int = 30,
) -> ForecastResult:
    """Forecast with drift estimated from the most recent window."""
    if len(train) == 0:
        predictions = pd.Series([], index=test.index, dtype=float)
        return ForecastResult("recent_drift", predictions, float("nan"))
    window = min(window, len(train))
    recent = train.iloc[-window:]
    if len(recent) < 2:
        fallback = naive_forecast(train, test)
        return ForecastResult("recent_drift", fallback.predictions, fallback.rmse)

    slope = (recent.iloc[-1] - recent.iloc[0]) / max(1, len(recent) - 1)
    horizon = np.arange(1, len(test) + 1)
    predictions = pd.Series(
        train.iloc[-1] + horizon * slope, index=test.index, dtype=float
    )
    return ForecastResult(
        "recent_drift", predictions, root_mean_squared_error(test, predictions)
    )


def find_d_parameter(
    series: pd.Series,
    max_d: int = 3,
) -> tuple[Optional[int], Optional[pd.Series]]:
    """Estimate a differencing order by repeated ADF testing."""
    current = series.dropna().copy()
    if len(current) < 3:
        return None, None

    d = 0
    for _ in range(max_d + 1):
        try:
            adf_result = stationary_test_adf(current)
            if adf_result.is_stationary:
                return d, current
        except Exception:
            pass

        if d >= max_d:
            return None, None
        current = current.diff().dropna()
        d += 1

    return d, current


def baseline_forecasts(train: pd.Series, test: pd.Series) -> Dict[str, ForecastResult]:
    """Compute the standard baseline forecast set."""
    return {
        "naive": naive_forecast(train, test),
        "average": average_forecast(train, test),
        "drift": drift_forecast(train, test),
        "seasonal_naive": seasonal_naive_forecast(train, test),
        "recent_drift": recent_drift_forecast(train, test),
    }


def _require_statsmodels_deterministic():
    if find_spec("statsmodels") is None:
        raise ModuleNotFoundError(
            "statsmodels is required for Fourier-feature forecasting."
        )


def fourier_features_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    dp,
    future_steps: int = 90,
) -> Dict[str, pd.Series]:
    """Fit a linear model on deterministic/Fourier features and forecast future values."""
    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError("X must be a pandas DataFrame or Series")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    X_df = X.to_frame() if isinstance(X, pd.Series) else X.copy()
    coefficients, _, _, _ = np.linalg.lstsq(
        X_df.to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        rcond=None,
    )
    fitted = pd.Series(
        X_df.to_numpy(dtype=float) @ coefficients,
        index=y.index,
    )
    X_future = dp.out_of_sample(steps=future_steps)
    forecast = pd.Series(
        X_future.to_numpy(dtype=float) @ coefficients,
        index=X_future.index,
    )
    return {"fitted": fitted, "features_future": X_future, "forecast": forecast}


class TimeSeriesForecaster(TimeSeriesAnalyzer):
    """Forecaster facade that combines analysis context with baseline methods."""

    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None,
        aggregate_method: str = "mean",
        week_start: str = "Monday",
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ) -> None:
        super().__init__(
            data,
            date_column=date_column,
            value_column=value_column,
            aggregate_method=aggregate_method,
            week_start=week_start,
        )
        self.train_ts, self.val_ts, self.test_ts = split_time_series(
            self.ts,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
        )
        self._refresh_baselines()

    def _refresh_baselines(self) -> None:
        """Populate compatibility attributes for baseline results."""
        results = baseline_forecasts(self.train_ts, self.test_ts)
        self._baseline_results = results

        self.naive_pred = results["naive"].predictions
        self.naive_rmse = results["naive"].rmse

        self.aver_pred = results["average"].predictions
        self.aver_rmse = results["average"].rmse

        self.sim_drift_pred = results["drift"].predictions
        self.sim_drift_rmse = results["drift"].rmse

        self.seasonal_week_pred = results["seasonal_naive"].predictions
        self.seasonal_week_rmse = results["seasonal_naive"].rmse

        self.recent_drift_pred = results["recent_drift"].predictions
        self.recent_drift_rmse = results["recent_drift"].rmse

    def baseline_forecasts(self) -> Dict[str, ForecastResult]:
        """Return the standard baseline forecast set for the current split."""
        return dict(self._baseline_results)

    def plot_forecast(
        self,
        result: ForecastResult,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """Plot a forecast result against the train/validation/test split."""
        display_title = title or f"{result.method} Forecast (RMSE: {result.rmse:.2f})"
        return plot_forecast_predictions(
            train=self.train_ts,
            val=self.val_ts,
            test=self.test_ts,
            pred=result.predictions,
            value_name=self.value_column,
            title=display_title,
            show=show,
        )

    def find_d_parameter(
        self, max_d: int = 3
    ) -> tuple[Optional[int], Optional[pd.Series]]:
        """Estimate differencing order from the training portion of the series."""
        return find_d_parameter(self.train_ts, max_d=max_d)

    def plot_pred_basic(self, method: str = "naive", show: bool = True):
        """Compatibility wrapper for plotting a named baseline method."""
        results = self.baseline_forecasts()
        normalized = "drift" if method.lower() == "sim_drift" else method.lower()
        normalized = "seasonal_naive" if normalized == "season_naive" else normalized
        if normalized not in results:
            raise ValueError(f"Unknown forecast method: {method}")
        return self.plot_forecast(results[normalized], show=show)

    def fourier_features_forecast(
        self, X: pd.DataFrame, y: pd.Series, dp, future_steps: int = 90
    ):
        """Convenience wrapper around the module-level Fourier forecast helper."""
        _require_statsmodels_deterministic()
        return fourier_features_forecast(X, y, dp, future_steps=future_steps)


__all__ = [
    "ForecastResult",
    "TimeSeriesForecaster",
    "TimeSeriesForcastor",
    "average_forecast",
    "baseline_forecasts",
    "drift_forecast",
    "find_d_parameter",
    "fourier_features_forecast",
    "naive_forecast",
    "recent_drift_forecast",
    "root_mean_squared_error",
    "seasonal_naive_forecast",
]

TimeSeriesForcastor = TimeSeriesForecaster
