```Python Code/estatecnica/time_series_analysis/draft/tests/smoke_test.py#L1-300
#!/usr/bin/env python3
"""
Smoke tests for the draft time series helper modules.

This script is intentionally lightweight and conservative:
- It adjusts sys.path so the `estatecnica` package (under "Python Code")
  can be imported when the script is executed in-place.
- It uses the sample AAPL CSV included in the repository for basic checks.
- It does not modify any production files; it only imports from the `draft/`
  folder and exercises a handful of functions to ensure they import and run.

Run:
    python3 smoke_test.py

You can use the script as a quick sanity check after edits to the draft modules.
"""
from pathlib import Path
import sys
import traceback

import pandas as pd


def ensure_repo_on_path(target_pkg_name="estatecnica"):
    """
    Walk up from this file until we find a directory named `target_pkg_name`,
    then add its parent to sys.path so that `import estatecnica...` works.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == target_pkg_name:
            repo_root = parent.parent  # the directory that contains `estatecnica`
            sys.path.insert(0, str(repo_root))
            return parent  # return the path to the package directory
    raise RuntimeError(f"Could not find a parent directory named '{target_pkg_name}'")


def load_sample_data(estatecnica_pkg_path: Path):
    """
    Load sample AAPL CSV included in the repository. Returns a pd.Series with DatetimeIndex.
    """
    data_path = estatecnica_pkg_path / "data" / "AAPL_sample_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found at: {data_path}")
    df = pd.read_csv(data_path, parse_dates=True)
    # Try to infer date column
    date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        df = df.set_index(date_col)
    else:
        # if no date-like column, try the first column as index
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(axis=0, subset=[df.index.name])
    # Pick a numeric column as value
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise RuntimeError("No numeric column found in sample data for testing")
    series = df[numeric_cols[0]].astype(float)
    series.name = numeric_cols[0]
    return series


def run_smoke_tests():
    print("SMOKE TEST: starting")
    estatecnica_pkg = ensure_repo_on_path("estatecnica")
    print(f"Found estatecnica package at: {estatecnica_pkg}")

    try:
        # Import draft modules from the package
        from estatecnica.time_series_analysis.draft.analyzer import TimeSeriesAnalyzerDraft
        from estatecnica.time_series_analysis.draft import transforms, stats, plots
        from estatecnica.time_series_analysis.draft.forecast import TimeSeriesForcastorDraft
    except Exception:
        traceback.print_exc()
        raise

    # Load sample data
    ts = load_sample_data(estatecnica_pkg)
    print(f"Loaded sample series: name={ts.name}, n_obs={len(ts)}")

    # 1) Test transforms.to_series and daily/weekly aggregations
    try:
        daily, weekly, monthly, yearly = transforms.daily_to_weekly_and_yearly(ts, method="mean")
        assert len(daily) > 0 and len(monthly) > 0, "Aggregates look empty"
        print("TRANSFORMS: daily/monthly aggregation successful")
    except Exception:
        print("TRANSFORMS: FAILED")
        traceback.print_exc()
        raise

    # 2) Test stats.adf wrapper
    try:
        adf_res = stats.stationary_test_adf(ts)
        print("STATS: ADF completed, pvalue=", adf_res.pvalue)
    except Exception:
        print("STATS: ADF FAILED")
        traceback.print_exc()
        raise

    # 3) Test suggest_arima_orders (quick run)
    try:
        orders = stats.suggest_arima_orders(ts, max_lags=20)
        print("STATS: suggest_arima_orders -> p_suggest, q_suggest =", orders.get("p_suggest"), orders.get("q_suggest"))
    except Exception:
        print("STATS: suggest_arima_orders FAILED")
        traceback.print_exc()
        raise

    # 4) Test draft analyzer wrapper
    try:
        analyzer = TimeSeriesAnalyzerDraft(ts)
        stats_summary = analyzer.basic_statistics()
        print("ANALYZER: basic_statistics ->", {k: stats_summary[k] for k in ["n_obs", "mean"]})
        # call ADF via analyzer
        adf_via_analyzer = analyzer.stationary_test_adf()
        print("ANALYZER: ADF pvalue via wrapper:", getattr(adf_via_analyzer, "pvalue", None))
    except Exception:
        print("ANALYZER: FAILED")
        traceback.print_exc()
        raise

    # 5) Test plotting helpers (do not show windows)
    try:
        ax = plots.visual_inspection(ts, value_name=ts.name, monthly=monthly, weekly=weekly, show=False)
        print("Visual inspection produced axes")
    except Exception as e:
        print("Visual inspection failed:", e)

    try:
        ax2 = plots.plot_periodogram(ts, show=False)
        print("Periodogram returned axes")
    except Exception as e:
        print("Periodogram failed:", e)

    # 6) Test Ljung-Box via stats wrapper
    try:
        lb = stats.autocorrelation_ljung_box(ts, lags=5, return_df=True)
        print("STATS: Ljung-Box result shape:", getattr(lb, "shape", "n/a"))
    except Exception:
        print("STATS: Ljung-Box FAILED")
        traceback.print_exc()
        raise

    # 7) Test split_time_series utility
    try:
        train, val, test = transforms.split_time_series(ts, train_pct=0.7, val_pct=0.15, test_pct=0.15)
        print(f"SPLIT: train/val/test lengths = {len(train)}/{len(val)}/{len(test)}")
        assert len(train) + len(val) + len(test) == len(ts)
    except Exception:
        print("SPLIT: FAILED")
        traceback.print_exc()
        raise

    # 8) Test forecastor and forecast plotting (using draft forecast)
    try:
        forecastor = TimeSeriesForcastorDraft(ts)
        print("FORECAST: naive RMSE:", getattr(forecastor, "naive_rmse", None))
        print("FORECAST: average RMSE:", getattr(forecastor, "aver_rmse", None))
        print("FORECAST: seasonal RMSE:", getattr(forecastor, "seasonal_week_rmse", None))
        # Plot predictions using centralized plot helper (non-blocking)
        fig, ax = plots.plot_forecast_predictions(forecastor.train_ts, forecastor.val_ts, forecastor.test_ts, forecastor.naive_pred, forecastor.value_column, title="Naive Forecast (draft)", show=False)
        print("FORECAST: plot_forecast_predictions produced axes")
        # call forecastor plotting methods (which delegate to plots) with show=False where supported
        try:
            fig2, axes2 = forecastor.time_step_lag_linear_regression_plot(show=False)
            print("FORECAST: time_step_lag_linear_regression_plot produced axes")
        except TypeError:
            # fallback if signature differs
            _ = forecastor.time_step_lag_linear_regression_plot()
            print("FORECAST: time_step_lag_linear_regression_plot called (no-show)")
        try:
            model, y, y_pred, plot_info = forecastor.time_step_linear_regression_fit()
            print("FORECAST: linear regression fit returned model and prediction (plot_info present)")
        except ValueError:
            # older signature fallback
            model, y, y_pred = forecastor.time_step_linear_regression_fit()
            print("FORECAST: linear regression fit returned model and prediction")
        try:
            model2, y2, y2_pred, lag_plot_info = forecastor.lag_linear_regression_fit()
            print("FORECAST: lag regression fit returned model and prediction (plot_info present)")
        except ValueError:
            model2, y2, y2_pred = forecastor.lag_linear_regression_fit()
            print("FORECAST: lag regression fit returned model and prediction")
    except Exception:
        print("FORECAST: FAILED")
        traceback.print_exc()
        raise

    print("SMOKE TEST: all checks passed")


if __name__ == "__main__":
    try:
        run_smoke_tests()
    except Exception as exc:
        print("SMOKE TEST: encountered errors.")
        sys.exit(2)
    else:
        print("SMOKE TEST: success.")
        sys.exit(0)
