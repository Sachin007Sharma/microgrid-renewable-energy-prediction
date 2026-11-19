"""
models_dl.py — LSTM and GRU deep learning models for time-series forecasting.
Uses only the target variable's own history + calendar features (no leakage).
Works on both TensorFlow/Keras (standard) and falls back gracefully if unavailable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import FIG_DIR, PALETTE, DPI, DL_LOOKBACK, DL_EPOCHS, DL_BATCH, DL_UNITS
from data_loader import add_calendar_features, add_lag_features, add_rolling_features

# ── Try TensorFlow import ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. LSTM/GRU models will be skipped.")


CALENDAR_FEATS = [
    "day_of_year","month","week_of_year","day_of_week",
    "is_weekend","quarter","month_sin","month_cos","doy_sin","doy_cos",
]


# ── Sequence builder ──────────────────────────────────────────────────────────
def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Convert tabular arrays into (samples, lookback, features) format."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _build_feature_matrix(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build feature matrix: target + calendar features."""
    df = add_calendar_features(df)
    df = add_lag_features(df, target, [1, 2, 7])
    df = add_rolling_features(df, target, [7, 30])
    return df


def _prepare_all(train, val, test, target: str):
    """Prepare X/y arrays for train, val, test."""
    tv = pd.concat([train, val, test])   # full concat for consistent column order
    tv = _build_feature_matrix(tv, target).dropna()

    lag_cols  = [f"{target}_lag{l}" for l in [1, 2, 7]]
    roll_cols = ([f"{target}_roll_mean{w}" for w in [7, 30]] +
                 [f"{target}_roll_std{w}"  for w in [7, 30]])
    feat_cols = CALENDAR_FEATS + lag_cols + roll_cols

    # Feature matrix (target + derived features as predictors)
    # Target is also included as a feature within the lookback window
    feature_cols_with_target = [target] + feat_cols
    feature_cols_with_target = [c for c in feature_cols_with_target if c in tv.columns]

    train_idx = tv.index[tv.index.isin(train.index)]
    val_idx   = tv.index[tv.index.isin(val.index)]
    test_idx  = tv.index[tv.index.isin(test.index)]

    X_all = tv[feature_cols_with_target].values
    y_all = tv[target].values

    # Split by position (preserves chronological order)
    n_tr = len(train_idx)
    n_va = len(val_idx)
    n_te = len(test_idx)

    X_tr = X_all[:n_tr];             y_tr = y_all[:n_tr]
    X_va = X_all[n_tr:n_tr+n_va];   y_va = y_all[n_tr:n_tr+n_va]
    X_te = X_all[n_tr+n_va:];       y_te = y_all[n_tr+n_va:]
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_idx, feature_cols_with_target


def _normalize(X_tr, X_va, X_te, y_tr):
    """Min-max normalise X; StandardScaler for y."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    xscaler = MinMaxScaler()
    yscaler = StandardScaler()
    X_tr_n = xscaler.fit_transform(X_tr)
    X_va_n = xscaler.transform(X_va)
    X_te_n = xscaler.transform(X_te)
    y_tr_n = yscaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    return X_tr_n, X_va_n, X_te_n, y_tr_n, xscaler, yscaler


# ── LSTM ──────────────────────────────────────────────────────────────────────
def fit_lstm(train: pd.DataFrame, val: pd.DataFrame,
             test: pd.DataFrame, target: str,
             lookback: int = DL_LOOKBACK,
             units: int = DL_UNITS,
             epochs: int = DL_EPOCHS,
             batch_size: int = DL_BATCH) -> tuple:
    """
    Two-layer stacked LSTM.
    Returns (predictions_array, test_index).
    """
    if not TF_AVAILABLE:
        print(f"  LSTM [{target}] SKIPPED — TensorFlow not available.")
        return None, None

    X_tr, y_tr, X_va, y_va, X_te, y_te, test_idx, _ = _prepare_all(train, val, test, target)
    X_tr_n, X_va_n, X_te_n, y_tr_n, xscaler, yscaler = _normalize(X_tr, X_va, X_te, y_tr)

    # Build sequences
    # Combine tr+va for training sequences; use last lookback days of tr for seeding va
    X_tv_n = np.concatenate([X_tr_n, X_va_n])
    y_tv_n = np.concatenate([y_tr_n, yscaler.transform(y_va.reshape(-1,1)).ravel()])

    X_seq_tv, y_seq_tv = make_sequences(X_tv_n, y_tv_n, lookback)
    X_seq_te, _        = make_sequences(np.concatenate([X_tv_n[-lookback:], X_te_n]),
                                        np.concatenate([y_tv_n[-lookback:],
                                                        yscaler.transform(y_te.reshape(-1,1)).ravel()]),
                                        lookback)

    n_feats = X_tr_n.shape[1]
    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Input(shape=(lookback, n_feats)),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(units // 2),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
    ]
    # Split tv sequences 80/20 for internal val during fitting
    split = int(0.8 * len(X_seq_tv))
    history = model.fit(
        X_seq_tv[:split], y_seq_tv[:split],
        validation_data=(X_seq_tv[split:], y_seq_tv[split:]),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb, verbose=0,
    )

    y_pred_scaled = model.predict(X_seq_te, verbose=0).ravel()
    y_pred = yscaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

    print(f"  LSTM [{target}] trained {len(history.history['loss'])} epochs, "
          f"final val_loss={history.history['val_loss'][-1]:.4f}")

    return y_pred, test_idx[-len(y_pred):]


# ── GRU ───────────────────────────────────────────────────────────────────────
def fit_gru(train: pd.DataFrame, val: pd.DataFrame,
            test: pd.DataFrame, target: str,
            lookback: int = DL_LOOKBACK,
            units: int = DL_UNITS,
            epochs: int = DL_EPOCHS,
            batch_size: int = DL_BATCH) -> tuple:
    """
    Two-layer stacked GRU. GRU is computationally lighter than LSTM;
    often achieves similar accuracy with fewer parameters.
    """
    if not TF_AVAILABLE:
        print(f"  GRU [{target}] SKIPPED — TensorFlow not available.")
        return None, None

    X_tr, y_tr, X_va, y_va, X_te, y_te, test_idx, _ = _prepare_all(train, val, test, target)
    X_tr_n, X_va_n, X_te_n, y_tr_n, xscaler, yscaler = _normalize(X_tr, X_va, X_te, y_tr)

    X_tv_n = np.concatenate([X_tr_n, X_va_n])
    y_tv_n = np.concatenate([y_tr_n, yscaler.transform(y_va.reshape(-1,1)).ravel()])

    X_seq_tv, y_seq_tv = make_sequences(X_tv_n, y_tv_n, lookback)
    X_seq_te, _        = make_sequences(np.concatenate([X_tv_n[-lookback:], X_te_n]),
                                        np.concatenate([y_tv_n[-lookback:],
                                                        yscaler.transform(y_te.reshape(-1,1)).ravel()]),
                                        lookback)

    n_feats = X_tr_n.shape[1]
    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Input(shape=(lookback, n_feats)),
        layers.GRU(units, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(units // 2),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
    ]
    split = int(0.8 * len(X_seq_tv))
    history = model.fit(
        X_seq_tv[:split], y_seq_tv[:split],
        validation_data=(X_seq_tv[split:], y_seq_tv[split:]),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb, verbose=0,
    )

    y_pred_scaled = model.predict(X_seq_te, verbose=0).ravel()
    y_pred = yscaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

    print(f"  GRU [{target}] trained {len(history.history['loss'])} epochs, "
          f"final val_loss={history.history['val_loss'][-1]:.4f}")

    return y_pred, test_idx[-len(y_pred):]

