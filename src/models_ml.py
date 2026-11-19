"""
models_ml.py — Linear Regression, Random Forest, XGBoost.
Feature engineering uses only lag/calendar/rolling features (no data leakage).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, LAG_DAYS, ROLLING_WIN
from data_loader import add_calendar_features, add_lag_features, add_rolling_features


# ── Feature Builder ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build feature matrix for a given target column.
    Only uses: lag features, rolling stats, calendar features.
    Does NOT use: *_forecast_MWh, *_kJ, renewable_share, the other target.
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target, LAG_DAYS)
    df = add_rolling_features(df, target, ROLLING_WIN)
    return df


CALENDAR_FEATS = [
    "day_of_year","month","week_of_year","day_of_week",
    "is_weekend","quarter","month_sin","month_cos","doy_sin","doy_cos",
]


def get_feature_cols(target: str) -> list:
    lag_cols  = [f"{target}_lag{l}"       for l in LAG_DAYS]
    roll_cols = [f"{target}_roll_mean{w}" for w in ROLLING_WIN] + \
                [f"{target}_roll_std{w}"  for w in ROLLING_WIN]
    return CALENDAR_FEATS + lag_cols + roll_cols


def prepare_Xy(df: pd.DataFrame, target: str):
    df_feat = build_features(df, target).dropna()
    feat_cols = get_feature_cols(target)
    X = df_feat[feat_cols].values
    y = df_feat[target].values
    idx = df_feat.index
    return X, y, idx


# ── Linear Regression ─────────────────────────────────────────────────────────
def fit_linear_regression(train: pd.DataFrame, val: pd.DataFrame,
                           test: pd.DataFrame, target: str):
    """StandardScaler + OLS Linear Regression."""
    tv = pd.concat([train, val])  # train+val for fitting
    X_tv, y_tv, _ = prepare_Xy(tv,   target)
    X_te, y_te, idx_te = prepare_Xy(test, target)

    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_te_s = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tv_s, y_tv)
    preds = model.predict(X_te_s)

    print(f"  Linear Regression [{target}] fitted. Features: {X_tv_s.shape[1]}")
    return preds, idx_te, model, scaler


# ── Random Forest ─────────────────────────────────────────────────────────────
def fit_random_forest(train: pd.DataFrame, val: pd.DataFrame,
                      test: pd.DataFrame, target: str,
                      n_estimators: int = 200, max_depth: int = None,
                      random_state: int = 42):
    """Random Forest Regressor — no scaling needed."""
    tv = pd.concat([train, val])
    X_tv, y_tv, _ = prepare_Xy(tv,   target)
    X_te, y_te, idx_te = prepare_Xy(test, target)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_tv, y_tv)
    preds = model.predict(X_te)

    feat_cols = get_feature_cols(target)
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"  Random Forest [{target}] top feature: {importances.index[0]} ({importances.iloc[0]:.3f})")
    return preds, idx_te, model, importances


# ── XGBoost ───────────────────────────────────────────────────────────────────
def fit_xgboost(train: pd.DataFrame, val: pd.DataFrame,
                test: pd.DataFrame, target: str,
                n_estimators: int = 300, learning_rate: float = 0.05,
                max_depth: int = 5, random_state: int = 42):
    """XGBoost Regressor with early-stopping on validation."""
    X_tr, y_tr, _ = prepare_Xy(train, target)
    X_va, y_va, _ = prepare_Xy(val,   target)
    X_te, y_te, idx_te = prepare_Xy(test, target)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    # Fit on train only; val used for early stopping
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              verbose=False)
    # Re-fit on full train+val with best n_estimators
    best_n = model.best_iteration + 1
    tv = pd.concat([train, val])
    X_tv, y_tv, _ = prepare_Xy(tv, target)
    final_model = xgb.XGBRegressor(
        n_estimators=best_n,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method="hist",
        verbosity=0,
    )
    final_model.fit(X_tv, y_tv)
    preds = final_model.predict(X_te)

    feat_cols = get_feature_cols(target)
    importances = pd.Series(final_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"  XGBoost [{target}] best n_estimators={best_n}, top feature: {importances.index[0]}")
    return preds, idx_te, final_model, importances


# ── Feature Importance Plot ────────────────────────────────────────────────────
def plot_feature_importance(importances: pd.Series, model_name: str,
                             target: str, top_n: int = 12, save: bool = True):
    top = importances.head(top_n)
    color = PALETTE["solar"] if "solar" in target else PALETTE["wind"]

    fig, ax = plt.subplots(figsize=(9, 5))
    top.sort_values().plot.barh(ax=ax, color=color, alpha=0.8)
    ax.set_title(f"Feature Importance — {model_name} [{target}]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    if save:
        fname = f"feat_importance_{model_name.lower().replace(' ','_')}_{target}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=DPI)
    plt.show(); plt.close()

