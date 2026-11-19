"""
models_statistical.py — ARIMA, SARIMA, Holt-Winters Exponential Smoothing.
Uses only the target variable's own history (no leakage from other columns).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, sys, time
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, RESULTS_DIR


# ── Shared: build train series from split ─────────────────────────────────────
def _series(train: pd.DataFrame, val: pd.DataFrame, target: str) -> pd.Series:
    """Combine train+val as the fitting series; test is held out."""
    s = pd.concat([train[target], val[target]])
    return s.asfreq("D").fillna(method="ffill")


# ── Naive baseline ────────────────────────────────────────────────────────────
def naive_forecast(train: pd.DataFrame, val: pd.DataFrame,
                   test: pd.DataFrame, target: str) -> np.ndarray:
    """Predict each day = previous day's actual (lag-1)."""
    all_actual = pd.concat([train[target], val[target], test[target]])
    preds = all_actual.shift(1)
    return preds.loc[test.index].values


# ── ARIMA ─────────────────────────────────────────────────────────────────────
def fit_arima(train: pd.DataFrame, val: pd.DataFrame,
              test: pd.DataFrame, target: str,
              order: tuple = (2, 1, 2)) -> np.ndarray:
    """
    Fit ARIMA on train+val, forecast over test horizon.
    order (p,d,q) derived from ACF/PACF: d=1 for stationarity,
    p=2 (AR terms at lag1,2), q=2 (MA correction).
    """
    train_series = _series(train, val, target)
    t0 = time.time()
    model = ARIMA(train_series, order=order, freq="D")
    fit   = model.fit()
    preds = fit.forecast(steps=len(test))
    print(f"  ARIMA{order} [{target}] fitted in {time.time()-t0:.1f}s")
    return np.array(preds)


# ── SARIMA ────────────────────────────────────────────────────────────────────
def fit_sarima(train: pd.DataFrame, val: pd.DataFrame,
               test: pd.DataFrame, target: str,
               order: tuple = (1, 1, 1),
               seasonal_order: tuple = (1, 0, 1, 7)) -> np.ndarray:
    """
    Fit SARIMA with weekly seasonality (s=7).
    Annual (s=365) would be more representative but is computationally infeasible
    for gradient-based optimisation with 900+ obs.
    """
    train_series = _series(train, val, target)
    t0 = time.time()
    model = SARIMAX(train_series, order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    freq="D")
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=len(test))
    print(f"  SARIMA{order}x{seasonal_order} [{target}] fitted in {time.time()-t0:.1f}s")
    return np.array(preds)


# ── Holt-Winters ──────────────────────────────────────────────────────────────
def fit_holtwinters(train: pd.DataFrame, val: pd.DataFrame,
                    test: pd.DataFrame, target: str,
                    seasonal_periods: int = 365,
                    trend: str = "add",
                    seasonal: str = "add") -> np.ndarray:
    """
    Holt-Winters with additive trend + additive seasonality.
    Works well for solar (strong annual cycle) and wind (moderate cycle).
    Uses annual period=365.
    """
    train_series = _series(train, val, target)
    t0 = time.time()
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit   = model.fit(optimized=True)
    preds = fit.forecast(len(test))
    print(f"  Holt-Winters [{target}] fitted in {time.time()-t0:.1f}s")
    return np.array(preds)

