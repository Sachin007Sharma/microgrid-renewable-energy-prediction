
"""
main.py — Single command-line entrypoint to run the entire Microgrid Renewable Energy Prediction pipeline.
Executes all phases: Data Loading, EDA, Existing Forecast Comparison, Preprocessing,
Model Training (Statistical, ML, DL), Evaluation, and 90-day Future Forecasting.
All figures are saved to outputs/figures/ and CSV summaries to outputs/results/.
"""

import os
import sys
import warnings

# Use non-interactive backend to safely render and save all plots headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add src/ to Python module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

from config import DATA_PATH, TARGETS, RESULTS_DIR, FIG_DIR, LAG_DAYS, ROLLING_WIN
from data_loader import load_data, validate_data, get_splits
import eda
from forecast_comparison import run_all as run_forecast_comparison
import preprocessing
from models_statistical import naive_forecast, fit_arima, fit_sarima, fit_holtwinters
from models_ml import fit_linear_regression, fit_random_forest, fit_xgboost, plot_feature_importance
from models_dl import fit_lstm, fit_gru, TF_AVAILABLE
from evaluation import (
    evaluate, build_results_table, plot_model_predictions,
    plot_model_comparison, plot_r2_comparison, plot_residuals
)
from future_forecast import iterative_forecast, plot_future_forecast


def main():
    print("=" * 75)
    print("🔋 MICROGRID RENEWABLE ENERGY PREDICTION PIPELINE")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load and Validate Dataset
    # ---------------------------------------------------------
    print("\n[Phase 1] Loading and Validating Dataset...")
    df = load_data(DATA_PATH)
    val_summary = validate_data(df)
    print(f"  ✓ Rows: {val_summary['rows']}, Columns: {val_summary['cols']}")
    print(f"  ✓ Date Range: {val_summary['start']} to {val_summary['end']}")
    print(f"  ✓ Gaps: {val_summary['daily_gaps']}, Duplicates: {val_summary['duplicate_rows']}")
    print(f"  ✓ Renewable Share Formula verified (max diff: {val_summary['renewable_share_max_diff']:.2e})")

    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ---------------------------------------------------------
    print("\n[Phase 2] Generating Exploratory Data Analysis (EDA) Figures...")
    desc = eda.descriptive_stats(df)
    print(desc.to_string())
    eda.plot_overview(df, save=True)
    eda.plot_monthly_boxplots(df, save=True)
    eda.plot_annual_totals(df, save=True)
    eda.plot_correlation(df, save=True)
    eda.plot_renewable_share(df, save=True)
    eda.plot_outliers(df, save=True)
    print("  ✓ Saved EDA figures to outputs/figures/")

    # ---------------------------------------------------------
    # 3. Existing Forecast Comparison
    # ---------------------------------------------------------
    print("\n[Phase 3] Comparing Pre-Existing Forecasts vs Actuals...")
    existing_metrics = run_forecast_comparison(df)
    print("  ✓ Saved comparison figures and existing_forecast_metrics.csv")

    # ---------------------------------------------------------
    # 4. Preprocessing & Stationarity Tests
    # ---------------------------------------------------------
    print("\n[Phase 4] Time Series Preprocessing & Stationarity Analysis...")
    adf_df = preprocessing.run_adf_all(df)
    print(adf_df[["variable", "adf_stat", "p_value", "stationary"]].to_string(index=False))
    for target in TARGETS:
        preprocessing.plot_rolling_stats(df[target], target, window=30, save=True)
        preprocessing.plot_decomposition(df[target], target, period=365, model="additive", save=True)
        preprocessing.plot_acf_pacf(df[target], target, lags=60, save=True)
    print("  ✓ Preprocessing plots saved.")

    # ---------------------------------------------------------
    # 5. Chronological Train / Val / Test Split
    # ---------------------------------------------------------
    print("\n[Phase 5] Chronological Data Splitting (No Data Leakage)...")
    train, val, test = get_splits(df)
    print(f"  ✓ Train set : {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days)")
    print(f"  ✓ Val set   : {val.index[0].date()} -> {val.index[-1].date()} ({len(val)} days)")
    print(f"  ✓ Test set  : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")

    # ---------------------------------------------------------
    # 6. Model Training and Testing
    # ---------------------------------------------------------
    print("\n[Phase 6] Training All Forecasting Models...")
    results_all = []
    all_preds = {"solar_MWh": {}, "wind_MWh": {}}

    for target in TARGETS:
        print(f"\n--- Modelling for Target: {target} ---")
        actual_test = test[target].values

        # 1. Baseline Naive
        preds_naive = naive_forecast(train, val, test, target)
        m_naive = evaluate(actual_test[:len(preds_naive)], preds_naive, "Naive", target)
        results_all.append(m_naive)
        all_preds[target]["Naive"] = preds_naive
        print(f"  [Naive]         RMSE: {m_naive['RMSE (MWh)']:,.1f} | MAE: {m_naive['MAE (MWh)']:,.1f} | MAPE: {m_naive['MAPE (%)']:.2f}% | R²: {m_naive['R²']:.3f}")

        # 2. ARIMA
        try:
            preds_arima = fit_arima(train, val, test, target, order=(2, 1, 2))
            m_arima = evaluate(actual_test[:len(preds_arima)], preds_arima, "ARIMA", target)
            results_all.append(m_arima)
            all_preds[target]["ARIMA"] = preds_arima
            print(f"  [ARIMA]         RMSE: {m_arima['RMSE (MWh)']:,.1f} | MAE: {m_arima['MAE (MWh)']:,.1f} | MAPE: {m_arima['MAPE (%)']:.2f}% | R²: {m_arima['R²']:.3f}")
        except Exception as e:
            print(f"  [ARIMA] Error: {e}")

        # 3. SARIMA
        try:
            preds_sarima = fit_sarima(train, val, test, target, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
            m_sarima = evaluate(actual_test[:len(preds_sarima)], preds_sarima, "SARIMA", target)
            results_all.append(m_sarima)
            all_preds[target]["SARIMA"] = preds_sarima
            print(f"  [SARIMA]        RMSE: {m_sarima['RMSE (MWh)']:,.1f} | MAE: {m_sarima['MAE (MWh)']:,.1f} | MAPE: {m_sarima['MAPE (%)']:.2f}% | R²: {m_sarima['R²']:.3f}")
        except Exception as e:
            print(f"  [SARIMA] Error: {e}")

        # 4. Holt-Winters
        try:
            preds_hw = fit_holtwinters(train, val, test, target, seasonal_periods=365)
            m_hw = evaluate(actual_test[:len(preds_hw)], preds_hw, "Holt-Winters", target)
            results_all.append(m_hw)
            all_preds[target]["Holt-Winters"] = preds_hw
            print(f"  [Holt-Winters]  RMSE: {m_hw['RMSE (MWh)']:,.1f} | MAE: {m_hw['MAE (MWh)']:,.1f} | MAPE: {m_hw['MAPE (%)']:.2f}% | R²: {m_hw['R²']:.3f}")
        except Exception as e:
            print(f"  [Holt-Winters] Error: {e}")

        # 5. Linear Regression
        preds_lr, idx_lr, _, _ = fit_linear_regression(train, val, test, target)
        act_lr = test.loc[idx_lr, target].values
        m_lr = evaluate(act_lr, preds_lr, "Linear Regression", target)
        results_all.append(m_lr)
        all_preds[target]["Linear Regression"] = preds_lr
        print(f"  [Linear Reg]    RMSE: {m_lr['RMSE (MWh)']:,.1f} | MAE: {m_lr['MAE (MWh)']:,.1f} | MAPE: {m_lr['MAPE (%)']:.2f}% | R²: {m_lr['R²']:.3f}")

        # 6. Random Forest
        preds_rf, idx_rf, _, imp_rf = fit_random_forest(train, val, test, target)
        act_rf = test.loc[idx_rf, target].values
        m_rf = evaluate(act_rf, preds_rf, "Random Forest", target)
        results_all.append(m_rf)
        all_preds[target]["Random Forest"] = preds_rf
        plot_feature_importance(imp_rf, "Random Forest", target, save=True)
        print(f"  [Random Forest] RMSE: {m_rf['RMSE (MWh)']:,.1f} | MAE: {m_rf['MAE (MWh)']:,.1f} | MAPE: {m_rf['MAPE (%)']:.2f}% | R²: {m_rf['R²']:.3f}")

        # 7. XGBoost
        preds_xgb, idx_xgb, _, imp_xgb = fit_xgboost(train, val, test, target)
        act_xgb = test.loc[idx_xgb, target].values
        m_xgb = evaluate(act_xgb, preds_xgb, "XGBoost", target)
        results_all.append(m_xgb)
        all_preds[target]["XGBoost"] = preds_xgb
        plot_feature_importance(imp_xgb, "XGBoost", target, save=True)
        print(f"  [XGBoost]       RMSE: {m_xgb['RMSE (MWh)']:,.1f} | MAE: {m_xgb['MAE (MWh)']:,.1f} | MAPE: {m_xgb['MAPE (%)']:.2f}% | R²: {m_xgb['R²']:.3f}")

        # 8. LSTM
        if TF_AVAILABLE:
            preds_lstm, idx_lstm = fit_lstm(train, val, test, target, epochs=40)
            if preds_lstm is not None:
                act_lstm = test.loc[idx_lstm, target].values if idx_lstm is not None else actual_test[:len(preds_lstm)]
                m_lstm = evaluate(act_lstm[:len(preds_lstm)], preds_lstm, "LSTM", target)
                results_all.append(m_lstm)
                all_preds[target]["LSTM"] = preds_lstm
                print(f"  [LSTM]          RMSE: {m_lstm['RMSE (MWh)']:,.1f} | MAE: {m_lstm['MAE (MWh)']:,.1f} | MAPE: {m_lstm['MAPE (%)']:.2f}% | R²: {m_lstm['R²']:.3f}")

        # 9. GRU
        if TF_AVAILABLE:
            preds_gru, idx_gru = fit_gru(train, val, test, target, epochs=40)
            if preds_gru is not None:
                act_gru = test.loc[idx_gru, target].values if idx_gru is not None else actual_test[:len(preds_gru)]
                m_gru = evaluate(act_gru[:len(preds_gru)], preds_gru, "GRU", target)
                results_all.append(m_gru)
                all_preds[target]["GRU"] = preds_gru
                print(f"  [GRU]           RMSE: {m_gru['RMSE (MWh)']:,.1f} | MAE: {m_gru['MAE (MWh)']:,.1f} | MAPE: {m_gru['MAPE (%)']:.2f}% | R²: {m_gru['R²']:.3f}")

    # ---------------------------------------------------------
    # 7. Model Evaluation & Comparison
    # ---------------------------------------------------------
    print("\n[Phase 7] Generating Evaluation Tables & Figures...")
    results_df = build_results_table(results_all)
    results_csv = os.path.join(RESULTS_DIR, "model_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n--- SOLAR RESULTS ---")
    print(results_df[results_df["Target"] == "solar_MWh"].to_string(index=False))
    print(f"\n--- WIND RESULTS ---")
    print(results_df[results_df["Target"] == "wind_MWh"].to_string(index=False))

    for target in TARGETS:
        plot_model_predictions(test[target], all_preds[target], target, save=True)
        plot_model_comparison(results_df, target, save=True)
        plot_residuals(test[target].values, all_preds[target], target, save=True)
    plot_r2_comparison(results_df, save=True)

    # ---------------------------------------------------------
    # 8. Future 90-Day Forecast
    # ---------------------------------------------------------
    print("\n[Phase 8] Computing Future 90-Day Predictions...")
    horizon = 90
    future_preds = {}

    for target in TARGETS:
        sub = results_df[(results_df["Target"] == target) & (results_df["Model"].isin(["XGBoost", "Random Forest", "Linear Regression"]))]
        best_name = sub.sort_values("RMSE (MWh)").iloc[0]["Model"] if not sub.empty else "Random Forest"
        print(f"  Selected best ML model for {target}: {best_name}")

        if best_name == "XGBoost":
            _, _, model, _ = fit_xgboost(train, val, test, target)
            scaler = None
        elif best_name == "Random Forest":
            _, _, model, _ = fit_random_forest(train, val, test, target)
            scaler = None
        else:
            _, _, model, scaler = fit_linear_regression(train, val, test, target)

        full_df = pd.concat([train, val, test])
        future_series = iterative_forecast(model, full_df, target, scaler, horizon=horizon)
        future_preds[target] = future_series

    recent_mean_demand = df["demand_MWh"].tail(90).mean()
    forecast_df = plot_future_forecast(
        df,
        solar_future=future_preds["solar_MWh"],
        wind_future=future_preds["wind_MWh"],
        horizon=horizon,
        demand_mean=recent_mean_demand,
        save=True,
    )
    future_csv = os.path.join(RESULTS_DIR, "future_forecast.csv")
    forecast_df.to_csv(future_csv, index=False)
    print(f"  ✓ Saved future predictions to {future_csv}")

    print("\n" + "=" * 75)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  • Figures: outputs/figures/ ({len(os.listdir(FIG_DIR))} files)")
    print(f"  • Metrics & Forecasts: outputs/results/")
    print(f"  • Final Report: report/final_report.md")
    print("=" * 75)


if __name__ == "__main__":
    main()

