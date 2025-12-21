import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

from .time_series_analyzer import TimeSeriesAnalyzer


class TimeSeriesForcastor(TimeSeriesAnalyzer):
    def __init__(self, data, date_column, value_column):
        super().__init__(data, date_column, value_column)
        self.train_ts, self.val_ts, self.test_ts = self.__class__.split_time_series(
            self.ts
        )
        # perform all the baseline model
        self.naive_pred, self.naive_rmse = self.__class__.naive_forecast(
            self.train_ts, self.test_ts
        )
        self.aver_pred, self.aver_rmse = self.__class__.average_forecast(
            self.train_ts, self.test_ts
        )
        self.sim_drift_pred, self.sim_drift_rmse = self.__class__.simple_drift_forecast(
            self.train_ts, self.test_ts
        )
        self.seasonal_week_pred, self.seasonal_week_rmse = (
            self.__class__.seasonal_naive_forecast(self.train_ts, self.test_ts)
        )
        self.recent_drift_pred, self.recent_drift_rmse = (
            self.__class__.recent_drift_forecast(self.train_ts, self.test_ts)
        )

    def time_step_lag_linear_regression_plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].plot(self.df["time"], self.df[self.value_column], color="0.75")
        sns.regplot(
            x="time",
            y=self.value_column,
            data=self.df,
            ci=None,
            scatter_kws=dict(color="0.25"),
            ax=axes[0],
        )
        axes[0].set_title(f"Time Plot of {self.value_column}")

        sns.regplot(
            x="lag_1",
            y=self.value_column,
            data=self.df,
            ci=None,
            scatter_kws=dict(color="0.25"),
            ax=axes[1],
        )
        axes[1].set_aspect("equal")
        axes[1].set_title(f"Lag Plot (Shift = 1) of {self.value_column}")

    def time_step_linear_regression_fit(self):
        X = self.df.loc[:, ["time"]]
        y = self.df.loc[:, self.value_column]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        # plot the model got fitted
        ax = y.plot(**plot_params)
        ax = y_pred.plot(ax=ax, linewidth=3)
        ax.set_title(f"Time Plot of {self.value_column}")
        # Access the coefficients and intercept
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficients: {model.coef_}")
        return model, y, y_pred

    def lag_linear_regression_fit(self):
        X = self.df.loc[:, ["lag_1"]].dropna()
        y = self.df.loc[:, self.value_column]
        # drop the target value that's from the blank
        y, X = y.align(X, join="inner")

        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        # plot the model got fitted
        fig, ax = plt.subplots()
        ax.plot(X["lag_1"], y, ".", color="0.25")
        ax.plot(X["lag_1"], y_pred)
        ax.set(
            aspect="equal",
            ylabel=f"{self.value_column}",
            xlabel="lag_1",
            title=f"Lag Plot of {self.value_column}",
        )
        # Access the coefficients and intercept
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficients: {model.coef_}")
        return model, y, y_pred

    def plot_pred_basic(self, method="naive"):
        if method.lower() == "naive":
            pred_index = self.naive_pred.index
            pred_values = self.naive_pred.values
            title = f"Naive Forecast vs. Actual Data (RMSE: {self.naive_rmse:.2f})"
        elif method.lower() == "average":
            pred_index = self.aver_pred.index
            pred_values = self.aver_pred.values
            title = f"Average Forecast vs. Actual Data (RMSE: {self.aver_rmse:.2f})"
        elif method.lower() == "sim_drift":
            pred_index = self.sim_drift_pred.index
            pred_values = self.sim_drift_pred.values
            title = f"Simple drift Forecast vs. Actual Data (RMSE: {self.sim_drift_rmse:.2f})"
        elif method.lower() == "season_naive":
            pred_index = self.seasonal_week_pred.index
            pred_values = self.seasonal_week_pred.values
            title = f"Seasonal drift (7days) Forecast vs. Actual Data (RMSE: {self.seasonal_week_rmse:.2f})"
        elif method.lower() == "recent_drift":
            pred_index = self.recent_drift_pred.index
            pred_values = self.recent_drift_pred.values
            title = f"Recent drift (7days) Forecast vs. Actual Data (RMSE: {self.recent_drift_rmse:.2f})"

        plt.figure(figsize=(10, 6))
        # plot the original data in different colors
        # train data
        plt.plot(
            self.train_ts.index,
            self.train_ts.values,
            color="blue",
            label="Training Data",
        )
        # validation data
        plt.plot(
            self.val_ts.index,
            self.val_ts.values,
            color="orange",
            label="Validation Data",
        )
        # test data
        plt.plot(
            self.test_ts.index, self.test_ts.values, color="green", label="Test Data"
        )
        # pred data
        plt.plot(pred_index, pred_values, color="red", label="Prediction")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(self.value_column)
        plt.grid(True)
        plt.show()

    def find_d_parameter(self, max_d=3):
        """ """
        print("\n--- Finding Optimal 'd' Parameter ---")
        d = 0
        current_series = self.train_ts.dropna()

        for _ in range(max_d + 1):
            # Run the ADF test on the current series
            adf_result = adfuller(current_series)
            p_value = adf_result[1]

            print(f"Testing with d = {d}...")
            print(f"  ADF p-value: {p_value:.4f}")

            # Check if the series is stationary
            if p_value <= 0.05:
                print(f"  Result: Series is stationary. Optimal d = {d}")
                return d, current_series

            # If not stationary and we haven't reached max_d, difference the series
            if d < max_d:
                print("  Result: Series is non-stationary. Differencing...")
                d += 1
                current_series = current_series.diff().dropna()
            else:
                print(
                    f"  Result: Could not achieve stationarity after {max_d} differencing steps."
                )
                return None, None  # Or handle as an error

        return d, current_series

    @staticmethod
    def naive_forecast(train, test):
        "Forecasts the next value as the last observed value."
        # The last value of the training set is the first prediction
        last_value = train.iloc[-1]

        # All predictions for the test set are this same last value
        predictions = pd.Series(last_value, index=test.index)

        # Calculate error
        rmse = np.sqrt(mean_squared_error(test, predictions))
        return predictions, rmse

    @staticmethod
    def average_forecast(train, test):
        "Forecast all the future value as the mean of training data"
        # calculate the mean from all the train values
        mean_values = train.mean()

        # assign the mean value for all the predictions
        predictions = pd.Series(mean_values, index=test.index)

        # compare both the prediction value & the true value
        rmse = np.sqrt(mean_squared_error(test, predictions))
        return predictions, rmse

    @staticmethod
    def simple_drift_forecast(train, test):
        "Forecase using the last value with a simple trend component (slope calculated from the last & first value)"
        if len(train) < 2:
            drift = 0
            print("Not enough train data for calculating the drift")
        else:
            slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)

        h = np.arange(1, len(test) + 1)

        last_value = train.iloc[-1]

        predictions = last_value + h * slope
        predictions = pd.Series(predictions, index=test.index)

        rmse = np.sqrt(mean_squared_error(test, predictions))

        return predictions, rmse

    @staticmethod
    def seasonal_naive_forecast(train, test, seasonal_period=7):
        last_season = train.iloc[-seasonal_period:]

        num_repeats = int(np.ceil(len(test) / seasonal_period))
        repeated_season = np.tile(last_season, num_repeats)

        predictions = pd.Series(repeated_season[: len(test)], index=test.index)

        rmse = np.sqrt(mean_squared_error(test, predictions))
        return predictions, rmse

    @staticmethod
    def recent_drift_forecast(train, test, window=30):
        if len(train) < window:
            raise ValueError(
                "Window size cannot be larger than the training data length."
            )

        recent_data = train.iloc[-window:]

        if len(recent_data) < 2:
            slope = 0
        else:
            slope = (recent_data.iloc[-1] - recent_data.iloc[0]) / (
                len(recent_data) - 1
            )

        h = np.arange(1, len(test) + 1)

        last_value = train.iloc[-1]
        predictions = last_value + h * slope
        predictions = pd.Series(predictions, index=test.index)

        rmse = np.sqrt(mean_squared_error(test, predictions))

        return predictions, rmse

    @staticmethod
    def fourier_features_forecast(X, y, dp, future_steps=90):
        """ """
        model = LinearRegression(fit_intercept=False)

        _ = model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        X_fore = dp.out_of_sample(steps=future_steps)
        y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
        ax = y.plot(color="0.25", style=".", title="Tunnel Traffic - Seasonal Forecast")
        ax = y_pred.plot(ax=ax, label="Seasonal")
        ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color="C3")
        _ = ax.legend()
        return y_pred, X_fore, y_fore

    @staticmethod
    def split_time_series(ts, train_pct=0.7, val_pct=0.15, test_pct=0.15):
        """
        Splits a time series chronologically into training, validation, and test sets.

        Args:
            ts (pd.Series): The time series data with a DatetimeIndex.
            train_pct (float): Percentage of data for the training set.
            val_pct (float): Percentage of data for the validation set.
            test_pct (float): Percentage of data for the test set.

        Returns:
            tuple: A tuple containing (train, val, test) pandas Series.
        """
        if round(train_pct + val_pct + test_pct, 10) != 1.0:
            raise ValueError("Percentages must sum to 1.0")

        n = len(ts)

        # Calculate split points
        train_end_idx = int(n * train_pct)
        val_end_idx = int(n * (train_pct + val_pct))

        # Split the data
        train_data = ts.iloc[:train_end_idx]
        val_data = ts.iloc[train_end_idx:val_end_idx]
        test_data = ts.iloc[val_end_idx:]

        print("--- Time Series Split ---")
        print(
            f"Training Set:   {len(train_data)} points ({train_data.index.min().date()} to {train_data.index.max().date()})"
        )
        print(
            f"Validation Set: {len(val_data)} points ({val_data.index.min().date()} to {val_data.index.max().date()})"
        )
        print(
            f"Test Set:       {len(test_data)} points ({test_data.index.min().date()} to {test_data.index.max().date()})"
        )

        return train_data, val_data, test_data
