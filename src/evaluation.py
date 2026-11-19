"""
evaluation.py — Compute MAE, RMSE, MAPE, R² and produce comparison visuals.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, RESULTS_DIR


# ── Metric Functions ──────────────────────────────────────────────────────────
def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def mape(actual, pred, eps: float = 1.0):
    """MAPE — skips points where |actual| < eps to avoid division issues."""
    mask = np.abs(actual) >= eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def r2(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def evaluate(actual: np.ndarray, pred: np.ndarray, model_name: str, target: str) -> dict:
    """Return a metrics dict for one model × one target."""
    return {
        "Model":       model_name,
        "Target":      target,
        "MAE (MWh)":   round(mae(actual, pred),   1),
        "RMSE (MWh)":  round(rmse(actual, pred),  1),
        "MAPE (%)":    round(mape(actual, pred),   3),
        "R²":          round(r2(actual, pred),     4),
    }


def build_results_table(results_list: list) -> pd.DataFrame:
    df = pd.DataFrame(results_list)
    return df.sort_values(["Target", "RMSE (MWh)"]).reset_index(drop=True)


# ── Individual model: Actual vs Predicted ─────────────────────────────────────
def plot_model_predictions(actual: pd.Series, predictions: dict,
                            target: str, save: bool = True):
    """
    predictions: {model_name: np.ndarray} — all aligned to the same test index.
    actual: pd.Series with DatetimeIndex.
    """
    n = len(predictions)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    color = PALETTE["solar"] if "solar" in target else PALETTE["wind"]

    for ax, (model_name, preds) in zip(axes, predictions.items()):
        if preds is None:
            ax.text(0.5, 0.5, f"{model_name} — Not available",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        pred_series = pd.Series(preds, index=actual.index[:len(preds)])
        ax.plot(actual.index, actual.values,  color=PALETTE["actual"], lw=1.2, label="Actual",
                alpha=0.9)
        ax.plot(pred_series.index, pred_series.values, color=PALETTE["pred"], lw=1.0,
                ls="--", label=f"{model_name} Prediction", alpha=0.85)
        m = evaluate(actual.values[:len(preds)], preds, model_name, target)
        ax.set_title(f"{model_name} — {target}  |  MAE={m['MAE (MWh)']:.0f}  "
                     f"RMSE={m['RMSE (MWh)']:.0f}  MAPE={m['MAPE (%)']:.2f}%  R²={m['R²']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylabel("MWh")

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Model Predictions vs Actual — {target}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"predictions_{target}.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Comparison bar chart ──────────────────────────────────────────────────────
def plot_model_comparison(results_df: pd.DataFrame, target: str, save: bool = True):
    df_t = results_df[results_df["Target"] == target].copy()
    if df_t.empty:
        return

    metrics = ["MAE (MWh)", "RMSE (MWh)", "MAPE (%)"]
    n = len(metrics)
    color = PALETTE["solar"] if "solar" in target else PALETTE["wind"]

    fig, axes = plt.subplots(1, n, figsize=(16, 5))
    fig.suptitle(f"Model Comparison — {target}", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals = df_t.set_index("Model")[metric].sort_values()
        bars = ax.barh(vals.index, vals.values, color=color, alpha=0.75, edgecolor="white")
        # Highlight best (smallest)
        bars[0].set_color("#023E8A"); bars[0].set_alpha(1.0)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel(metric)
        for bar, val in zip(bars, vals.values):
            ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=8)
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"comparison_{target}.png"), dpi=DPI)
    plt.show(); plt.close()


# ── R² comparison ─────────────────────────────────────────────────────────────
def plot_r2_comparison(results_df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("R² Comparison Across All Models", fontsize=13, fontweight="bold")

    for ax, target in zip(axes, ["solar_MWh", "wind_MWh"]):
        df_t = results_df[results_df["Target"] == target].sort_values("R²", ascending=True)
        color = PALETTE["solar"] if "solar" in target else PALETTE["wind"]
        bars = ax.barh(df_t["Model"], df_t["R²"], color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlim(min(-0.1, df_t["R²"].min() - 0.05), 1.05)
        ax.set_title(f"R² — {target}", fontsize=11, fontweight="bold")
        ax.set_xlabel("R² Score (higher = better)")
        for bar, val in zip(bars, df_t["R²"]):
            ax.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, "comparison_r2.png"), dpi=DPI)
    plt.show(); plt.close()


# ── Residual plots ────────────────────────────────────────────────────────────
def plot_residuals(actual: np.ndarray, predictions: dict,
                   target: str, save: bool = True):
    n = sum(1 for v in predictions.values() if v is not None)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).ravel()
    color = PALETTE["solar"] if "solar" in target else PALETTE["wind"]

    idx = 0
    for model_name, preds in predictions.items():
        if preds is None:
            continue
        n_pts = min(len(actual), len(preds))
        resid = actual[:n_pts] - preds[:n_pts]
        ax = axes[idx]
        ax.scatter(preds[:n_pts], resid, alpha=0.35, s=10, color=color)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(resid.mean(), color="red", ls="--", lw=1.2,
                   label=f"Mean={resid.mean():.0f}")
        ax.set_title(f"Residuals — {model_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted (MWh)"); ax.set_ylabel("Residual (MWh)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        idx += 1

    # Hide unused subplots
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Residual Plots — {target}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIG_DIR, f"residuals_{target}.png"), dpi=DPI)
    plt.show(); plt.close()

