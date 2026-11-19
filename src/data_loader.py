"""
data_loader.py — Load, validate and return the clean dataset.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_PATH, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

# ── MWh-only columns (kJ columns are redundant; excluded as features) ─────────
MWH_COLS = [
    "solar_MWh", "wind_MWh", "demand_MWh",
    "solar_forecast_MWh", "wind_forecast_MWh", "demand_forecast_MWh",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, parse dates, sort chronologically."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df.index = df["time"]
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic integrity checks; return a summary dict."""
    summary = {}

    # Shape
    summary["rows"], summary["cols"] = df.shape

    # Date range
    summary["start"] = str(df["time"].min().date())
    summary["end"]   = str(df["time"].max().date())

    # Frequency
    diffs = df["time"].diff().dropna()
    summary["daily_gaps"]       = int((diffs != pd.Timedelta("1 day")).sum())
    summary["missing_values"]   = df.isnull().sum().to_dict()
    summary["duplicate_rows"]   = int(df.duplicated().sum())

    # renewable_share verification
    rs_calc = (df["solar_MWh"] + df["wind_MWh"]) / df["demand_MWh"]
    summary["renewable_share_max_diff"] = float((df["renewable_share"] - rs_calc).abs().max())

    # MWh vs kJ consistency (relative error < 10 ppm → consistent)
    mwh_to_kj = 3_600_000
    for col in ["solar", "wind", "demand"]:
        rel = ((df[f"{col}_kJ"] - df[f"{col}_MWh"] * mwh_to_kj) / df[f"{col}_kJ"]).abs().max()
        summary[f"{col}_kJ_rel_err"] = float(rel)

    return summary


def get_splits(df: pd.DataFrame):
    """Return (train, val, test) DataFrames using chronological boundaries."""
    df_tz = df.copy()
    df_tz.index = pd.DatetimeIndex(df_tz.index)

    train = df_tz.loc[TRAIN_START:TRAIN_END]
    val   = df_tz.loc[VAL_START:VAL_END]
    test  = df_tz.loc[TEST_START:TEST_END]
    return train, val, test


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived columns (no data leakage)."""
    df = df.copy()
    t = df["time"].dt
    df["day_of_year"]  = t.dayofyear
    df["month"]        = t.month
    df["week_of_year"] = t.isocalendar().week.astype(int)
    df["day_of_week"]  = t.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = t.quarter
    # Cyclical encoding of month and day_of_year to handle circularity
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def add_lag_features(df: pd.DataFrame, target: str, lags: list) -> pd.DataFrame:
    """Add lag features for a target column.  Safe — only uses past values."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, windows: list) -> pd.DataFrame:
    """Add rolling mean and std.  min_periods=1 to avoid leading NaNs."""
    df = df.copy()
    for w in windows:
        df[f"{target}_roll_mean{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std{w}"]  = df[target].shift(1).rolling(w, min_periods=1).std().fillna(0)
    return df

