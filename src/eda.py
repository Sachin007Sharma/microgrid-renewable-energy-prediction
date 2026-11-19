"""
eda.py — Exploratory Data Analysis visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, FIGSIZE_WIDE, FIGSIZE_TALL, FIGSIZE_SQ


# ── 1. Overview line plots ─────────────────────────────────────────────────────
def plot_overview(df: pd.DataFrame, save: bool = True):
    """4-panel overview: solar, wind, demand, renewable share."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    series_cfg = [
        ("solar_MWh",       "Solar Generation (MWh)",       PALETTE["solar"]),
        ("wind_MWh",        "Wind Generation (MWh)",         PALETTE["wind"]),
        ("demand_MWh",      "Electricity Demand (MWh)",      PALETTE["demand"]),
        ("renewable_share", "Renewable Share (%)",           "#8338EC"),
    ]
    for ax, (col, ylabel, color) in zip(axes, series_cfg):
        values = df[col] * 100 if col == "renewable_share" else df[col]
        ax.plot(df.index, values, color=color, lw=0.9, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        # Shade years alternately
        for year in [2015, 2017]:
            ax.axvspan(pd.Timestamp(f"{year}-01-01", tz="UTC"),
                       pd.Timestamp(f"{year}-12-31", tz="UTC"),
                       alpha=0.04, color="gray")

    axes[0].set_title("Microgrid Daily Time Series Overview (2015–2018)",
                      fontsize=14, fontweight="bold")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "01_overview.png"), dpi=DPI)
    plt.show(); plt.close()


# ── 2. Monthly box plots ───────────────────────────────────────────────────────
def plot_monthly_boxplots(df: pd.DataFrame, save: bool = True):
    """Seasonal patterns by month for solar, wind, demand."""
    df_plot = df.copy()
    df_plot["month"] = df_plot["time"].dt.month
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, color, title in zip(
        axes,
        ["solar_MWh", "wind_MWh", "demand_MWh"],
        [PALETTE["solar"], PALETTE["wind"], PALETTE["demand"]],
        ["Solar Generation (MWh)", "Wind Generation (MWh)", "Demand (MWh)"],
    ):
        groups = [df_plot.loc[df_plot["month"]==m, col].values for m in range(1, 13)]
        bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Monthly Distribution of Energy Variables (2015–2018)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "02_monthly_boxplots.png"), dpi=DPI)
    plt.show(); plt.close()


# ── 3. Annual trend bar ───────────────────────────────────────────────────────
def plot_annual_totals(df: pd.DataFrame, save: bool = True):
    df_plot = df.copy()
    df_plot["year"] = df_plot["time"].dt.year
    yearly = df_plot.groupby("year")[["solar_MWh","wind_MWh","demand_MWh"]].sum() / 1e6  # TWh

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(yearly))
    w = 0.25
    ax.bar(x - w, yearly["solar_MWh"],  w, label="Solar (TWh)",  color=PALETTE["solar"])
    ax.bar(x,     yearly["wind_MWh"],   w, label="Wind (TWh)",   color=PALETTE["wind"])
    ax.bar(x + w, yearly["demand_MWh"], w, label="Demand (TWh)", color=PALETTE["demand"])
    ax.set_xticks(x); ax.set_xticklabels(yearly.index)
    ax.set_xlabel("Year"); ax.set_ylabel("Energy (TWh)")
    ax.set_title("Annual Energy Totals (2015–2018)", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "03_annual_totals.png"), dpi=DPI)
    plt.show(); plt.close()


# ── 4. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, save: bool = True):
    cols = ["solar_MWh","wind_MWh","demand_MWh","renewable_share"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
                center=0, ax=ax, linewidths=0.5, square=True,
                annot_kws={"size": 12})
    ax.set_title("Correlation Matrix — MWh Variables", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "04_correlation.png"), dpi=DPI)
    plt.show(); plt.close()


# ── 5. Descriptive statistics table ──────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["solar_MWh","wind_MWh","demand_MWh","renewable_share"]
    desc = df[cols].describe().T
    desc["cv%"] = (desc["std"] / desc["mean"] * 100).round(2)
    desc.columns = ["count","mean","std","min","25%","50%","75%","max","cv%"]
    return desc.round(2)


# ── 6. Renewable share histogram ─────────────────────────────────────────────
def plot_renewable_share(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    rs_pct = df["renewable_share"] * 100

    # Line chart
    axes[0].plot(df.index, rs_pct, color="#8338EC", lw=0.8, alpha=0.8)
    axes[0].axhline(rs_pct.mean(), color="red", ls="--", lw=1.5,
                    label=f"Mean = {rs_pct.mean():.1f}%")
    axes[0].set_title("Daily Renewable Share Over Time", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Renewable Share (%)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Distribution
    axes[1].hist(rs_pct, bins=40, color="#8338EC", alpha=0.75, edgecolor="white")
    axes[1].axvline(rs_pct.mean(), color="red", ls="--", lw=1.5,
                    label=f"Mean = {rs_pct.mean():.1f}%")
    axes[1].set_title("Distribution of Renewable Share", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Renewable Share (%)"); axes[1].set_ylabel("Frequency")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "05_renewable_share.png"), dpi=DPI)
    plt.show(); plt.close()


# ── 7. Outlier visualisation ──────────────────────────────────────────────────
def plot_outliers(df: pd.DataFrame, save: bool = True):
    cols   = ["solar_MWh","wind_MWh","demand_MWh"]
    labels = ["Solar (MWh)","Wind (MWh)","Demand (MWh)"]
    colors = [PALETTE["solar"], PALETTE["wind"], PALETTE["demand"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, label, color in zip(axes, cols, labels, colors):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black", lw=2),
                   flierprops=dict(marker="o", color="red", alpha=0.5, markersize=4))
        ax.set_title(f"Outlier Box-Plot: {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel("MWh"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Outlier Detection via IQR (1.5×IQR rule)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "06_outliers.png"), dpi=DPI)
    plt.show(); plt.close()

