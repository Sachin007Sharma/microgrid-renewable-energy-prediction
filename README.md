# ⚡ Microgrid Renewable Energy Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Time Series Analysis framework for forecasting daily **Solar Generation**, **Wind Generation**, and **Electricity Demand** in renewable-dominated microgrids, with interactive date-based prediction, clean energy share tracking, and a full research paper.

---

## 🌟 Key Features

- **Multi-Paradigm Modeling:** Implements and benchmarks **9 forecasting architectures**:
  - *Trivial Baseline:* Naive Persistence
  - *Classical Statistical:* ARIMA(2,1,2), SARIMA(1,1,1)×(1,0,1,7), Holt-Winters Exponential Smoothing
  - *Machine Learning:* L2-Regularized Ridge Regression, Random Forest, XGBoost
  - *Deep Learning:* 2-layer Stacked LSTM and Gated Recurrent Units (GRU)
- **Zero Data Leakage:** Strict chronological partitioning:
  - **Train:** Jan 2015 – Jun 2017 (912 days, 62.5%)
  - **Validation:** Jul 2017 – Dec 2017 (184 days, 12.6%)
  - **Test:** Jan 2018 – Dec 2018 (364 days, 24.9%)
- **Legacy Forecast Audit:** Evaluates pre-existing operational day-ahead forecasts to uncover systemic grid over-forecasting bias.
- **Dynamic Date Predictor (CLI & GUI):** Predicts solar/wind generation, combined clean energy, and renewable share percentage for *any* custom date or date range.
- **Interactive Web App:** Modern dark-themed Streamlit dashboard with interactive Plotly visualizations, radar charts, and CSV downloads.
- **Formal IEEE Research Report:** Fully documented technical report in both Markdown (`report/final_report.md`) and styled Microsoft Word (`report/final_report.docx`).

---

## 🏆 Model Benchmark Results (2018 Test Data)

Evaluated on the unseen 2018 holdout set (364 daily observations):

| Target | Model | RMSE (MWh) | MAE (MWh) | MAPE (%) | $R^2$ Score | Rank |
|---|---|---|---|---|---|---|
| **Solar Generation** | **GRU** | **11,466.1** | **8,934.1** | **45.52%** | **0.5141** | 🥇 **Best DL** |
| | Linear Regression (Ridge) | 11,475.3 | 9,151.7 | 46.39% | 0.5131 | 🥈 Runner-Up |
| | XGBoost | 11,859.9 | 9,335.0 | 48.49% | 0.4799 | 🥉 3rd |
| | Random Forest | 11,872.1 | 9,249.4 | 48.04% | 0.4788 | 4th |
| | LSTM | 12,542.2 | 9,746.4 | 48.90% | 0.4187 | 5th |
| | Naive Persistence | 13,643.3 | 9,948.4 | 46.93% | 0.3121 | Baseline |
| | Holt-Winters | 15,193.4 | 12,098.1 | 61.55% | 0.1469 | 7th |
| | ARIMA(2,1,2) | 19,225.1 | 15,526.0 | 53.85% | -0.3659 | Degraded |
| | SARIMA(1,1,1)×(7) | 19,446.5 | 15,671.0 | 53.50% | -0.3975 | Degraded |
| **Wind Generation** | **Linear Regression (Ridge)** | **56,392.6** | **43,933.8** | **44.93%** | **0.3867** | 🥇 **Best Overall** |
| | GRU | 58,922.9 | 46,406.2 | 49.14% | 0.3281 | 🥈 Runner-Up |
| | Random Forest | 59,258.9 | 46,686.1 | 48.69% | 0.3227 | 🥉 3rd |
| | XGBoost | 61,062.4 | 48,697.5 | 51.48% | 0.2809 | 4th |
| | Naive Persistence | 64,112.4 | 49,278.3 | 44.13% | 0.2045 | Baseline |
| | LSTM | 70,500.9 | 57,546.7 | 59.58% | 0.0381 | 6th |
| | ARIMA(2,1,2) | 72,461.4 | 61,074.6 | 71.73% | -0.0161 | Degraded |
| | SARIMA(1,1,1)×(7) | 72,514.0 | 61,150.8 | 71.98% | -0.0176 | Degraded |
| | Holt-Winters | 145,454.9 | 129,968.3 | 163.03% | -3.0944 | Degraded |

---

## 📁 Repository Structure

```text
├── daily_data.csv                 # Open Power System Data (1,460 rows, 2015-2018)
├── main.py                        # Full end-to-end training and benchmark pipeline
├── predict.py                     # Interactive CLI date-based predictor
├── app.py                         # Interactive Streamlit Web Application
├── run_app.sh                     # Helper startup shell script
├── requirements.txt               # Python package dependencies
├── microgrid_forecasting.ipynb   # Interactive analysis notebook
├── src/                           # Modular core codebase
│   ├── config.py                  # Global hyperparameters, targets, and paths
│   ├── data_loader.py             # Data ingestion and chronological splitting
│   ├── eda.py                     # Statistical and exploratory graphics
│   ├── forecast_comparison.py     # Legacy operational forecast audit
│   ├── preprocessing.py           # ADF stationarity, decomposition, ACF/PACF
│   ├── models_statistical.py      # Naive, ARIMA, SARIMA, Holt-Winters
│   ├── models_ml.py               # Linear Regression, Random Forest, XGBoost
│   ├── models_dl.py               # Stacked LSTM and GRU neural networks
│   ├── evaluation.py              # Metric calculation (RMSE, MAE, MAPE, R2)
│   └── future_forecast.py         # Multi-step 90-day recursive rollout
├── outputs/                       # Generated artifacts
│   ├── figures/                   # 20+ analytical figures and comparisons
│   └── results/                   # Metric CSVs and future predictions
└── report/                        # Academic Documentation
    ├── final_report.md            # IEEE Research Paper (Markdown)
    └── final_report.docx          # IEEE Research Paper (Microsoft Word)
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/microgrid-renewable-energy-prediction.git
cd microgrid-renewable-energy-prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Complete Training Pipeline
To run data validation, EDA, legacy audit, train all 9 models, and generate evaluation figures:
```bash
python3 main.py
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- **🔮 Date Prediction:** Query any single date or date range (e.g., 2022 to 2026).
- **📊 Visuals & EDA:** Review seasonal trends, monthly boxplots, and correlation matrices.
- **🏆 Model Comparison:** View interactive leaderboards and radar performance charts.

### 5. Run the Interactive CLI Predictor
```bash
# Interactive mode (prompts for dates)
python3 predict.py

# Direct command-line query
python3 predict.py --start 2022-06-01 --end 2022-08-31 --model ensemble
```

---

## 📄 Academic Research Paper
A comprehensive technical paper formatted to **IEEE standards** is included:
- **Word Document:** [`report/final_report.docx`](report/final_report.docx)
- **Markdown Version:** [`report/final_report.md`](report/final_report.md)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
