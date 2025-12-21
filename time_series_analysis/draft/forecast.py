```Python Code/estatecnica/time_series_analysis/draft/forecast.py#L1-300
"""
Draft forecasting utilities and a lightweight forecastor class.

This module provides a draft `TimeSeriesForcastorDraft` that inherits from
the `TimeSeriesAnalyzerDraft` in the draft analyzer module. It implements
several simple baseline forecasting methods (naive, average, drift, seasonal)
and a few helper plotting utilities. The implementation is conservative:
- Keeps methods small and testable.
- Returns predictions as pandas Series with appropriate index.
- Uses mean_squared_error for RMSE calculation.

This file is intended to live in the `draft/` area and is safe to iterate on.
"""
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

from .analyzer import TimeSeriesAnalyzerDraft

# small plotting defaults used by forecast plotting functions
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)


class TimeSeriesForcastorDraft(TimeSeriesAnalyzerDraft):
    """
    Forecastor that builds on the draft analyzer.

    The class computes simple baseline forecasts during initialization so that
    notebooks can inspect them immediately. It is intentionally conservative:
    baseline forecasts are implemented as static methods and return (pred_series, rmse).
    """

    def __init__(self, data, date_column: Optional[str] = None, value_column: Optional[str] = None):
        super().__init__(data, date_column=date_column, value_column=value_column)
        # split the series into train/val/test using the analyzer's helper
        self.train_ts, self.val_ts, self.test_ts = self.split_time_series(self.ts)

        # compute baseline forecasts and store them
        self.naive_pred, self.naive_rmse = self.naive_forecast(self.train_ts, self.test_ts)
        self.aver_pred, self.aver_rmse = self.average_forecast(self.train_ts, self.test_ts)
        self.sim_drift_pred, self.sim_drift_rmse = self.simple_drift_forecast(self.train_ts, self.test_ts)
        self.seasonal_week_pred, self.seasonal_week_rmse = self.seasonal_naive_forecast(self.train_ts, self.test_ts)
        # recent drift uses a default window
        try:
            self.recent_drift_pred, self.recent_drift_rmse = self.recent_drift_forecast(self.train_ts, self.test_ts)
        except Exception:
            self.recent_drift_pred, self.recent_drift_rmse = pd.Series([], dtype=float), float("nan")

    # ---------------------
    # plotting helpers
    # ---------------------
    def time_step_lag_linear_regression_plot(self, show: bool = True):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(self.df["time"], self.df[self.value_column], color="0.75")
        sns.regplot(x="time", y=self.value_column, data=self.df, ci=None, scatter_kws=dict(color="0.25"), ax=axes[0])
        axes[0].set_title(f"Time Plot of {self.value_column}")

        sns.regplot(x="lag_1", y=self.value_column, data=self.df, ci=None, scatter_kws=dict(color="0.25"), ax=axes[1])
        axes[1].set_aspect("equal")
        axes[1].set_title(f"Lag Plot (Shift = 1) of {self.value_column}")

        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def time_step_linear_regression_fit(self):
        X = self.df.loc[:, ["time"]]
        y = self.df.loc[:, self.value_column]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        # Delegate plotting to the centralized plotting helper
        fig, ax = plot_time_regression(y, y_pred, self.value_column, show=True)
        # Access the coefficients and intercept
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficients: {model.coef_}")
        return model, y, y_pred

    def lag_linear_regression_fit(self):
        X = self.df.loc[:, ["lag_1"]].dropna()
        y = self.df.loc[:, self.value_column]
        y, X = y.align(X, join="inner")

        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)

        fig, ax = plt.subplots()
        ax.plot(X["lag_1"], y, ".", color="0.25")
        ax.plot(X["lag_1"], y_pred)
        ax.set(aspect="equal", ylabel=f"{self.value_column}", xlabel="lag_1", title=f"Lag Plot of {self.value_column}")
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficients: {model.coef_}")
        return model, y, y_pred

    def plot_pred_basic(self, method: str = "naive"):
        method = method.lower()
        if method == "naive":
            pred_index = self.naive_pred.index
            pred_values = self.naive_pred.values
            title = f"Naive Forecast vs. Actual Data (RMSE: {self.naive_rmse:.2f})"
        elif method == "average":
            pred_index = self.aver_pred.index
            pred_values = self.aver_pred.values
            title = f"Average Forecast vs. Actual Data (RMSE: {self.aver_rmse:.2f})"
        elif method == "sim_drift":
            pred_index = self.sim_drift_pred.index
            pred_values = self.sim_drift_pred.values
            title = f"Simple drift Forecast vs. Actual Data (RMSE: {self.sim_drift_rmse:.2f})"
        elif method == "season_naive":
            pred_index = self.seasonal_week_pred.index
            pred_values = self.seasonal_week_pred.values
            title = f"Seasonal naive (7d) Forecast vs. Actual Data (RMSE: {self.seasonal_week_rmse:.2f})"
        elif method == "recent_drift":
            pred_index = self.recent_drift_pred.index
            pred_values = self.recent_drift_pred.values
            title = f"Recent drift (window) Forecast vs. Actual Data (RMSE: {self.recent_drift_rmse:.2f})"
        else:
            raise ValueError(f"Unknown method '{method}'")

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_ts.index, self.train_ts.values, color="blue", label="Training Data")
        plt.plot(self.val_ts.index, self.val_ts.values, color="orange", label="Validation Data")
        plt.plot(self.test_ts.index, self.test_ts.values, color="green", label="Test Data")
        if len(pred_index) > 0:
            plt.plot(pred_index, pred_values, color="red", label="Prediction")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(self.value_column)
        plt.grid(True)
        plt.legend()
        plt.show()

    # ---------------------
    # forecasting baselines
    # ---------------------
    @staticmethod
    def naive_forecast(train: pd.Series, test: pd.Series) -> Tuple[pd.Series, float]:
        """Predict every value as the last observed value in the training set."""
        if len(train) == 0:
            preds = pd.Series([], index=test.index)
            return preds, float("nan")
        last_value = train.iloc[-1]
        preds = pd.Series(last_value, index=test.index)
        rmse = np.sqrt(mean_squared_error(test, preds)) if len(test) > 0 else float("nan")
        return preds, float(rmse)

    @staticmethod
    def average_forecast(train: pd.Series, test: pd.Series) -> Tuple[pd.Series, float]:
        """Predict all future values as the mean of the training set."""
        if len(train) == 0:
            preds = pd.Series([], index=test.index)
            return preds, float("nan")
        mean_value = train.mean()
        preds = pd.Series(mean_value, index=test.index)
        rmse = np.sqrt(mean_squared_error(test, preds)) if len(test) > 0 else float("nan")
        return preds, float(rmse)

    @staticmethod
    def simple_drift_forecast(train: pd.Series, test: pd.Series) -> Tuple[pd.Series, float]:
        """
        Forecast with a simple drift computed from first and last training values.
        Predictions: last_value + h * slope where slope = (last-first)/(n-1)
        """
        if len(train) < 2:
            # Not enough data to compute slope, fallback to naive
            return TimeSeriesForcastorDraft.naive_forecast(train, test)
        slope = (train.iloc[-1] - train.iloc[0]) / max(1, (len(train) - 1))
        h = np.arange(1, len(test) + 1)
        last_value = train.iloc[-1]
        preds = pd.Series(last_value + h * slope, index=test.index)
        rmse = np.sqrt(mean_squared_error(test, preds)) if len(test) > 0 else float("nan")
        return preds, float(rmse)

    @staticmethod
    def seasonal_naive_forecast(train: pd.Series, test: pd.Series, seasonal_period: int = 7) -> Tuple[pd.Series, float]:
        """
        Seasonal naive: repeat the last observed season (e.g., last 7 days).
        """
        if len(train) == 0:
            return pd.Series([], index=test.index), float("nan")
        if len(train) < seasonal_period:
            # not enough history for a full season, fallback to naive
            return TimeSeriesForcastorDraft.naive_forecast(train, test)
        last_season = train.iloc[-seasonal_period:]
        num_repeats = int(np.ceil(len(test) / seasonal_period))
        repeated = np.tile(last_season.values, num_repeats)[: len(test)]
        preds = pd.Series(repeated, index=test.index)
        rmse = np.sqrt(mean_squared_error(test, preds)) if len(test) > 0 else float("nan")
        return preds, float(rmse)

    @staticmethod
    def recent_drift_forecast(train: pd.Series, test: pd.Series, window: int = 30) -> Tuple[pd.Series, float]:
        """
        Compute drift using recent `window` days. If the training series is shorter than
        the window, raise a ValueError to force caller awareness.
        """
        if len(train) == 0:
            return pd.Series([], index=test.index), float("nan")
        if len(train) < window:
            # fallback to using entire train for slope
            window = len(train)
        recent = train.iloc[-window:]
        if len(recent) < 2:
            return TimeSeriesForcastorDraft.naive_forecast(train, test)
        slope = (recent.iloc[-1] - recent.iloc[0]) / max(1, (len(recent) - 1))
        h = np.arange(1, len(test) + 1)
        last_value = train.iloc[-1]
        preds = pd.Series(last_value + h * slope, index=test.index)
        rmse = np.sqrt(mean_squared_error(test, preds)) if len(test) > 0 else float("nan")
        return preds, float(rmse)

    @staticmethod
    def fourier_features_forecast(X, y, dp, future_steps: int = 90):
        """
        Fit a linear model with provided deterministic features (dp). Returns predictions and forecast.
        This mirrors the simple seasonal linear approach in the original script.
        """
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        X_fore = dp.out_of_sample(steps=future_steps)
        y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
        ax = y.plot(color="0.25", style=".", title="Seasonal Forecast")
        _ = y_pred.plot(ax=ax, label="Seasonal")
        _ = y_fore.plot(ax=ax, label="Seasonal Forecast", color="C3")
        ax.legend()
        return y_pred, X_fore, y_fore

    # ---------------------
    # utility helpers
    # ---------------------
    @staticmethod
    def find_d_parameter(series: pd.Series, max_d: int = 3) -> Tuple[Optional[int], Optional[pd.Series]]:
        """
        Try differencing up to max_d to find minimal `d` that makes the series stationary
        according to the ADF test. Returns (d, differenced_series) on success, (None, None) on failure.
        """
        current = series.dropna().copy()
        if len(current) < 3:
            return None, None
        d = 0
        for _ in range(max_d + 1):
            try:
                adf_result = adfuller(current)
                pvalue = adf_result[1]
            except Exception:
                pvalue = 1.0
            if pvalue <= 0.05:
                return d, current
            if d < max_d:
                current = current.diff().dropna()
                d += 1
            else:
                return None, None
        return d, current

    @staticmethod
    def split_time_series(ts: pd.Series, train_pct: float = 0.7, val_pct: float = 0.15, test_pct: float = 0.15) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Provide the same split semantics as the analyzer transform helper for convenience.
        Delegates to the parent's static splitter if available.
        """
        # Use the parent's helper if present
        try:
            return TimeSeriesAnalyzerDraft.split_time_series(ts, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct)
        except Exception:
            # fallback: manual split
            if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-9:
                raise ValueError("train/val/test percentages must sum to 1.0")
            n = len(ts)
            train_end = int(n * train_pct)
            val_end = int(n * (train_pct + val_pct))
            return ts.iloc[:train_end], ts.iloc[train_end:val_end], ts.iloc[val_end:]
