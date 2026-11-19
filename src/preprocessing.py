"""
preprocessing.py — Stationarity tests, decomposition, ACF/PACF, transformations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings, os, sys
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, FIGSIZE_WIDE, FIGSIZE_TALL


# ── ADF Stationarity Test ──────────────────────────────────────────────────────
def adf_test(series: pd.Series, name: str = "") -> dict:
    """Run Augmented Dickey-Fuller test; return result dict."""
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "variable"  : name,
        "adf_stat"  : round(result[0], 4),
        "p_value"   : round(result[1], 6),
        "lags_used" : result[2],
        "n_obs"     : result[3],
        "critical_1%"  : round(result[4]["1%"],  4),
        "critical_5%"  : round(result[4]["5%"],  4),
        "critical_10%" : round(result[4]["10%"], 4),
        "stationary"   : result[1] < 0.05,
    }
    return out


def run_adf_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run ADF on solar_MWh and wind_MWh (original and first-differenced)."""
    rows = []
    for col in ["solar_MWh", "wind_MWh"]:
        rows.append(adf_test(df[col], f"{col} (original)"))
        rows.append(adf_test(df[col].diff().dropna(), f"{col} (1st diff)"))
    return pd.DataFrame(rows)


# ── Rolling Mean / Std ─────────────────────────────────────────────────────────
def plot_rolling_stats(series: pd.Series, col: str, window: int = 30, save: bool = True):
    roll_mean = series.rolling(window).mean()
    roll_std  = series.rolling(window).std()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(series.index, series, alpha=0.4, color=PALETTE.get(col.split("_")[0], "steelblue"),
            label="Daily")
    ax.plot(roll_mean.index, roll_mean, color="black", lw=1.8, label=f"{window}-day Rolling Mean")
    ax.fill_between(series.index,
                    roll_mean - roll_std, roll_mean + roll_std,
                    alpha=0.15, color="gray", label=f"±1 Std Dev")
    ax.set_title(f"{col} — Rolling Statistics (window={window} days)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("MWh")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"rolling_{col}.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Seasonal Decomposition ─────────────────────────────────────────────────────
def plot_decomposition(series: pd.Series, col: str, period: int = 365,
                       model: str = "additive", save: bool = True):
    dec = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    palette_key = col.split("_")[0]
    color = PALETTE.get(palette_key, "steelblue")

    components = [
        (dec.observed,  "Observed",  color),
        (dec.trend,     "Trend",     "black"),
        (dec.seasonal,  "Seasonal",  "#E63946"),
        (dec.resid,     "Residual",  "gray"),
    ]
    for ax, (data, label, c) in zip(axes, components):
        ax.plot(data.index, data, color=c, lw=0.9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"Seasonal Decomposition — {col} ({model.capitalize()}, period={period})",
                      fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"decomp_{col}.png"), dpi=DPI)
    plt.show(); plt.close()


# ── ACF / PACF ────────────────────────────────────────────────────────────────
def plot_acf_pacf(series: pd.Series, col: str, lags: int = 60, save: bool = True):
    acf_vals  = acf(series.dropna(),  nlags=lags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=lags, method="ywm")
    conf      = 1.96 / np.sqrt(len(series))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, vals, title in zip(axes,
                                [acf_vals, pacf_vals],
                                ["ACF", "PACF"]):
        ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.7, width=0.5)
        ax.axhline(conf,  color="red", ls="--", lw=1)
        ax.axhline(-conf, color="red", ls="--", lw=1)
        ax.axhline(0,     color="black", lw=0.8)
        ax.set_title(f"{title} — {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lag (days)"); ax.set_ylabel("Correlation")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"acf_pacf_{col}.png"), dpi=DPI)
    plt.show(); plt.close()

