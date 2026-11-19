"""
forecast_comparison.py — Compare existing (pre-built) forecasts vs actual values.
This section is kept COMPLETELY SEPARATE from models we train ourselves.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, RESULTS_DIR


COMPARISONS = [
    ("Solar",  "solar_MWh",  "solar_forecast_MWh",  PALETTE["solar"],  "Solar Generation (MWh)"),
    ("Wind",   "wind_MWh",   "wind_forecast_MWh",   PALETTE["wind"],   "Wind Generation (MWh)"),
    ("Demand", "demand_MWh", "demand_forecast_MWh", PALETTE["demand"], "Electricity Demand (MWh)"),
]


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_forecast_metrics(actual: pd.Series, forecast: pd.Series, name: str) -> dict:
    err   = actual - forecast
    pct   = err / actual * 100
    mae   = err.abs().mean()
    rmse  = np.sqrt((err ** 2).mean())
    mape  = pct.abs().mean()
    bias  = err.mean()
    r2    = 1 - ((err**2).sum() / ((actual - actual.mean())**2).sum())
    return {
        "Variable": name,
        "MAE (MWh)":   round(mae,  1),
        "RMSE (MWh)":  round(rmse, 1),
        "MAPE (%)":    round(mape, 3),
        "Bias (MWh)":  round(bias, 1),
        "R²":          round(r2,   4),
        # bias = actual - forecast:
        #   positive bias → forecast < actual → forecast UNDER-estimated
        #   negative bias → forecast > actual → forecast OVER-estimated
        "Bias Direction": "Under-forecast" if bias > 0 else "Over-forecast",
    }


def all_forecast_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, actual_col, fc_col, _, _ in COMPARISONS:
        rows.append(compute_forecast_metrics(df[actual_col], df[fc_col], name))
    return pd.DataFrame(rows).set_index("Variable")


# ── Actual vs Forecast line chart ─────────────────────────────────────────────
def plot_actual_vs_forecast(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Existing Forecasts vs Actual Values (2015–2018)",
                 fontsize=14, fontweight="bold")

    for ax, (name, actual_col, fc_col, color, ylabel) in zip(axes, COMPARISONS):
        ax.plot(df.index, df[actual_col], color=color,              lw=0.9, alpha=0.85, label="Actual")
        ax.plot(df.index, df[fc_col],     color=PALETTE["forecast"],lw=0.9, alpha=0.7,
                ls="--", label="Existing Forecast")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{name}: Actual vs Existing Forecast", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "07_actual_vs_forecast.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Error distribution plots ──────────────────────────────────────────────────
def plot_error_distributions(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Existing Forecast Error Distributions", fontsize=13, fontweight="bold")

    for col_idx, (name, actual_col, fc_col, color, _) in enumerate(COMPARISONS):
        err = df[actual_col] - df[fc_col]
        pct = err / df[actual_col] * 100

        # Absolute error distribution
        axes[0, col_idx].hist(err, bins=40, color=color, alpha=0.75, edgecolor="white")
        axes[0, col_idx].axvline(err.mean(), color="red",   ls="--", lw=1.5,
                                 label=f"Mean={err.mean():.0f}")
        axes[0, col_idx].axvline(0,          color="black", ls="-",  lw=1)
        axes[0, col_idx].set_title(f"{name} — Abs Error", fontsize=10, fontweight="bold")
        axes[0, col_idx].set_xlabel("Error (MWh)")
        axes[0, col_idx].legend(fontsize=8)
        axes[0, col_idx].grid(alpha=0.3)

        # Percentage error distribution
        axes[1, col_idx].hist(pct, bins=40, color=color, alpha=0.75, edgecolor="white")
        axes[1, col_idx].axvline(pct.mean(), color="red",   ls="--", lw=1.5,
                                 label=f"Mean={pct.mean():.1f}%")
        axes[1, col_idx].axvline(0,          color="black", ls="-",  lw=1)
        axes[1, col_idx].set_title(f"{name} — % Error", fontsize=10, fontweight="bold")
        axes[1, col_idx].set_xlabel("Error (%)")
        axes[1, col_idx].legend(fontsize=8)
        axes[1, col_idx].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "08_error_distributions.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Error over time ───────────────────────────────────────────────────────────
def plot_error_over_time(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Forecast Error Over Time (Actual − Forecast)", fontsize=13, fontweight="bold")

    for ax, (name, actual_col, fc_col, color, _) in zip(axes, COMPARISONS):
        err = df[actual_col] - df[fc_col]
        ax.bar(df.index, err, color=color, alpha=0.5, width=1.0)
        ax.axhline(0,        color="black", lw=1)
        ax.axhline(err.mean(), color="red", ls="--", lw=1.2,
                   label=f"Mean Bias = {err.mean():.0f} MWh")
        ax.set_ylabel(f"{name} Error\n(MWh)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "09_error_over_time.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Scatter: Actual vs Forecast ───────────────────────────────────────────────
def plot_scatter_actual_vs_forecast(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Scatter: Actual vs Existing Forecast", fontsize=13, fontweight="bold")

    for ax, (name, actual_col, fc_col, color, xlabel) in zip(axes, COMPARISONS):
        ax.scatter(df[fc_col], df[actual_col], color=color, alpha=0.35, s=10)
        lo = min(df[actual_col].min(), df[fc_col].min())
        hi = max(df[actual_col].max(), df[fc_col].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.5, ls="--", label="Perfect forecast")
        ax.set_xlabel(f"{name} Forecast (MWh)", fontsize=9)
        ax.set_ylabel(f"{name} Actual (MWh)",   fontsize=9)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "10_scatter_actual_vs_forecast.png"), dpi=DPI)
    plt.show(); plt.close()


def run_all(df: pd.DataFrame):
    """Run the full existing-forecast comparison suite."""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  SECTION 3: Existing Forecast Evaluation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    metrics = all_forecast_metrics(df)
    print(metrics.to_string())
    metrics.to_csv(os.path.join(RESULTS_DIR, "existing_forecast_metrics.csv"))

    plot_actual_vs_forecast(df)
    plot_error_distributions(df)
    plot_error_over_time(df)
    plot_scatter_actual_vs_forecast(df)
    return metrics
