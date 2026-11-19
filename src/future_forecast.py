"""
future_forecast.py — Use best-performing model to forecast beyond Dec 2018.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, LAG_DAYS, ROLLING_WIN
from data_loader import add_calendar_features, add_lag_features, add_rolling_features


CALENDAR_FEATS = [
    "day_of_year","month","week_of_year","day_of_week",
    "is_weekend","quarter","month_sin","month_cos","doy_sin","doy_cos",
]


def _get_feature_cols(target: str) -> list:
    lag_cols  = [f"{target}_lag{l}"       for l in LAG_DAYS]
    roll_cols = [f"{target}_roll_mean{w}" for w in ROLLING_WIN] + \
                [f"{target}_roll_std{w}"  for w in ROLLING_WIN]
    return CALENDAR_FEATS + lag_cols + roll_cols


def _build_future_frame(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """
    Build a rolling future frame by appending predicted values day-by-day.
    This allows lag features to be computed correctly for each future step.
    """
    future_df = df.copy()
    last_date = df.index[-1]

    # Use UTC-aware timestamps consistent with the dataset
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta("1D"),
        periods=horizon,
        freq="D",
        tz="UTC",
    )
    return future_df, future_dates


def iterative_forecast(model, df: pd.DataFrame, target: str,
                        scaler, horizon: int = 90) -> pd.Series:
    """
    Multi-step ahead forecast by iteratively predicting one day at a time
    and feeding the prediction back as a lag feature for the next step.
    Works for sklearn-style models (Linear Regression, RF, XGBoost).
    """
    from data_loader import add_calendar_features, add_lag_features, add_rolling_features
    import numpy as np

    history = df[[target, "time"]].copy()
    feat_cols = _get_feature_cols(target)
    predictions = []
    last_date = df["time"].iloc[-1]

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)

        # Build a single-row feature frame
        row = pd.DataFrame({"time": [next_date], target: [np.nan]})
        row["time"] = pd.to_datetime(row["time"], utc=True)

        # Append to history and compute features
        temp = pd.concat([history, row], ignore_index=True)
        temp = add_calendar_features(temp)
        temp = add_lag_features(temp, target, LAG_DAYS)
        temp = add_rolling_features(temp, target, ROLLING_WIN)

        last_row = temp.iloc[[-1]]
        available = [c for c in feat_cols if c in last_row.columns]
        missing   = [c for c in feat_cols if c not in last_row.columns]
        if missing:
            for m in missing:
                last_row[m] = 0.0

        X = last_row[feat_cols].values

        if scaler is not None:
            X = scaler.transform(X)

        pred = float(model.predict(X)[0])
        pred = max(pred, 0)  # energy cannot be negative
        predictions.append(pred)

        # Feed prediction back into history
        history = pd.concat([
            history,
            pd.DataFrame({"time": [next_date], target: [pred]})
        ], ignore_index=True)

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta("1D"),
        periods=horizon, freq="D", tz="UTC",
    )
    return pd.Series(predictions, index=future_dates, name=f"{target}_forecast")


def holtwinters_future(hw_fit, horizon: int = 90) -> np.ndarray:
    """Use a fitted Holt-Winters model to forecast future steps directly."""
    return hw_fit.forecast(horizon)


def plot_future_forecast(df: pd.DataFrame,
                          solar_future: pd.Series,
                          wind_future: pd.Series,
                          horizon: int = 90,
                          demand_mean: float = None,
                          save: bool = True):
    """
    Plot historical + future solar, wind, combined renewable,
    and estimated renewable share.
    """
    # Combined renewable
    combined_future = solar_future + wind_future
    combined_hist   = df["solar_MWh"] + df["wind_MWh"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # ── Solar ─────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df.index[-180:], df["solar_MWh"][-180:],
            color=PALETTE["solar"], lw=1, label="Historical (Solar)", alpha=0.8)
    ax.plot(solar_future.index, solar_future.values,
            color=PALETTE["pred"], lw=1.5, ls="--", label=f"Forecast ({horizon} days)")
    ax.axvline(df.index[-1], color="gray", ls=":", lw=1.5)
    ax.set_title(f"Solar Generation Forecast — Next {horizon} Days", fontsize=11, fontweight="bold")
    ax.set_ylabel("MWh"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Wind ──────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df.index[-180:], df["wind_MWh"][-180:],
            color=PALETTE["wind"], lw=1, label="Historical (Wind)", alpha=0.8)
    ax.plot(wind_future.index, wind_future.values,
            color=PALETTE["pred"], lw=1.5, ls="--", label=f"Forecast ({horizon} days)")
    ax.axvline(df.index[-1], color="gray", ls=":", lw=1.5)
    ax.set_title(f"Wind Generation Forecast — Next {horizon} Days", fontsize=11, fontweight="bold")
    ax.set_ylabel("MWh"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Combined renewable ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(df.index[-180:], combined_hist[-180:],
            color="#2D6A4F", lw=1, label="Historical (Solar + Wind)", alpha=0.8)
    ax.plot(combined_future.index, combined_future.values,
            color=PALETTE["pred"], lw=1.5, ls="--", label=f"Forecast ({horizon} days)")
    ax.axvline(df.index[-1], color="gray", ls=":", lw=1.5)
    ax.set_title(f"Combined Renewable Generation Forecast", fontsize=11, fontweight="bold")
    ax.set_ylabel("MWh"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Renewable share estimate ───────────────────────────────────────────────
    ax = axes[3]
    if demand_mean is not None:
        rs_hist   = (combined_hist[-180:] / df["demand_MWh"][-180:]) * 100
        rs_future = (combined_future / demand_mean) * 100
        ax.plot(df.index[-180:], rs_hist,
                color="#8338EC", lw=1, label="Historical Renewable Share", alpha=0.8)
        ax.plot(rs_future.index, rs_future.values,
                color=PALETTE["pred"], lw=1.5, ls="--", label="Estimated Renewable Share")
        ax.axhline(rs_hist.mean(), color="gray", ls=":", lw=1)
        ax.set_ylabel("Renewable Share (%)"); ax.legend(fontsize=9)
        ax.set_title("Estimated Renewable Share (using mean demand)", fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Demand forecast not available — renewable share not estimated",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.grid(alpha=0.3)

    # Grey out forecast region
    for ax in axes:
        ax.axvspan(df.index[-1], combined_future.index[-1],
                   alpha=0.05, color="purple")

    fig.suptitle(f"Future Renewable Energy Forecast — {horizon} Days Beyond 2018-12-30",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "future_forecast.png"), dpi=DPI)
    plt.show(); plt.close()

    # Save forecast values
    out = pd.DataFrame({
        "date":           combined_future.index.date,
        "solar_forecast": solar_future.values,
        "wind_forecast":  wind_future.values,
        "combined_renewable": combined_future.values,
    })
    if demand_mean is not None:
        out["renewable_share_pct"] = (combined_future.values / demand_mean * 100).round(2)
    return out

