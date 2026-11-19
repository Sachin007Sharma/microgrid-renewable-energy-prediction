#!/opt/anaconda3/bin/python3
"""
app.py — Microgrid Renewable Energy Prediction Web Application
Run with: streamlit run app.py
"""

import os
import sys
import warnings
import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Microgrid Energy Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "daily_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGS_DIR    = os.path.join(BASE_DIR, "outputs", "figures")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a1a2e 100%);
}
/* Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .label  { color: #aaa; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
.metric-card .value  { color: #fff; font-size: 22px; font-weight: 700; }
.metric-card .unit   { color: #aaa; font-size: 12px; }
/* Section headers */
.section-header {
    font-size: 26px; font-weight: 700; color: #fff;
    padding: 6px 0 2px 0; margin-bottom: 4px;
    border-bottom: 2px solid rgba(255,255,255,0.15);
}
.section-sub {
    color: #aaa; font-size: 14px; margin-bottom: 20px;
}
/* Model table */
.best-row { background-color: rgba(46,213,115,0.15) !important; }
/* Plotly charts – remove white bg */
.js-plotly-plot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (same as predict.py so both use identical model)
# ══════════════════════════════════════════════════════════════════════════
def extract_features(dt_series: pd.Series) -> pd.DataFrame:
    doy   = dt_series.dt.dayofyear
    dow   = dt_series.dt.dayofweek
    month = dt_series.dt.month
    year_offset = (dt_series.dt.year - 2015) + (doy / 365.25)
    return pd.DataFrame({
        "doy_sin1":  np.sin(2 * np.pi * doy / 365.25),
        "doy_cos1":  np.cos(2 * np.pi * doy / 365.25),
        "doy_sin2":  np.sin(4 * np.pi * doy / 365.25),
        "doy_cos2":  np.cos(4 * np.pi * doy / 365.25),
        "doy_sin3":  np.sin(6 * np.pi * doy / 365.25),
        "doy_cos3":  np.cos(6 * np.pi * doy / 365.25),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin":   np.sin(2 * np.pi * dow / 7),
        "dow_cos":   np.cos(2 * np.pi * dow / 7),
        "is_weekend":(dow >= 5).astype(float),
        "year_trend": year_offset,
    })


# ══════════════════════════════════════════════════════════════════════════
#  CACHED: load & train  (runs once, stays in Streamlit session cache)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️  Training models on 2015-2018 data…")
def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    X = extract_features(df["time"])
    models = {}
    for target in ["solar_MWh", "wind_MWh", "demand_MWh"]:
        y = df[target].values
        m1 = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, random_state=42)
        m2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        m3 = Ridge(alpha=1.0)
        m1.fit(X, y); m2.fit(X, y); m3.fit(X, y)
        models[target] = (m1, m2, m3)
    return df, models


@st.cache_data(show_spinner=False)
def load_model_results():
    path = os.path.join(RESULTS_DIR, "model_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_existing_metrics():
    path = os.path.join(RESULTS_DIR, "existing_forecast_metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ══════════════════════════════════════════════════════════════════════════
#  PREDICT for a date range
# ══════════════════════════════════════════════════════════════════════════
def predict_range(models, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    dates  = pd.date_range(start=str(start), end=str(end), freq="D", tz="UTC")
    X_fut  = extract_features(pd.Series(dates))
    result = {"date": dates.date}
    for target in ["solar_MWh", "wind_MWh", "demand_MWh"]:
        m1, m2, m3 = models[target]
        pred = 0.50*m1.predict(X_fut) + 0.35*m2.predict(X_fut) + 0.15*m3.predict(X_fut)
        result[target] = np.clip(pred, 0, None).round(1)
    pred_df = pd.DataFrame(result)
    pred_df["day_name"]            = [d.strftime("%A") for d in dates]
    pred_df["total_renewable_MWh"] = pred_df["solar_MWh"] + pred_df["wind_MWh"]
    pred_df["renewable_share_pct"] = (pred_df["total_renewable_MWh"] / pred_df["demand_MWh"] * 100).round(2)
    return pred_df


# ══════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════
PLOT_BG  = "rgba(0,0,0,0)"
GRID_COL = "rgba(255,255,255,0.08)"
FONT_COL = "#cccccc"

def _base_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(color="#fff", size=15)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, size=12),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
    )


def chart_forecast_lines(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    dates = pred_df["date"].astype(str)

    # Solar
    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["solar_MWh"], name="Solar Generation",
        line=dict(color="#E76F51", width=2), fill="tozeroy",
        fillcolor="rgba(231,111,81,0.12)"
    ))
    # Wind
    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["wind_MWh"], name="Wind Generation",
        line=dict(color="#2A9D8F", width=2), fill="tozeroy",
        fillcolor="rgba(42,157,143,0.12)"
    ))
    # Demand
    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["demand_MWh"], name="Total Demand",
        line=dict(color="#E9C46A", width=1.5, dash="dot")
    ))

    fig.update_layout(**_base_layout("Daily Generation vs Grid Demand (MWh)", height=380))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MWh")
    return fig


def chart_stacked_area(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    dates = pred_df["date"].astype(str)

    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["solar_MWh"], name="Solar",
        stackgroup="one", line=dict(color="#E76F51", width=0.5),
        fillcolor="rgba(231,111,81,0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["wind_MWh"], name="Wind",
        stackgroup="one", line=dict(color="#2A9D8F", width=0.5),
        fillcolor="rgba(42,157,143,0.7)"
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["demand_MWh"], name="Total Demand",
        line=dict(color="#E9C46A", width=2, dash="dash")
    ))
    fig.update_layout(**_base_layout("Stacked Clean Energy vs Demand", height=350))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="MWh")
    return fig


def chart_renewable_share(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    dates  = pred_df["date"].astype(str)
    mean_s = pred_df["renewable_share_pct"].mean()

    fig.add_trace(go.Scatter(
        x=dates, y=pred_df["renewable_share_pct"],
        name="Renewable Share",
        line=dict(color="#8338EC", width=2.2),
        fill="tozeroy", fillcolor="rgba(131,56,236,0.15)"
    ))
    fig.add_hline(y=mean_s, line=dict(color="#E63946", width=1.5, dash="dot"),
                  annotation_text=f"Avg: {mean_s:.1f}%",
                  annotation_font_color="#E63946")
    fig.update_layout(**_base_layout("Renewable Share of Total Demand (%)", height=300))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Share (%)", range=[0, max(60, pred_df["renewable_share_pct"].max()*1.2)])
    return fig


def chart_donut_single(row: pd.Series) -> go.Figure:
    renew = row["renewable_share_pct"]
    non_r = max(0.0, 100.0 - renew)
    fig = go.Figure(go.Pie(
        labels=["Renewables", "Thermal / Grid"],
        values=[renew, non_r],
        hole=0.60,
        marker=dict(colors=["#2A9D8F", "#555"], line=dict(color="#111", width=2)),
        textfont=dict(color="#fff", size=12),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL),
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"<b>{renew:.1f}%</b><br>Clean",
                          x=0.5, y=0.5, font_size=15, font_color="#fff",
                          showarrow=False)]
    )
    return fig


def chart_bar_breakdown(row: pd.Series) -> go.Figure:
    cats   = ["Solar", "Wind", "Total Clean", "Demand"]
    vals   = [row["solar_MWh"], row["wind_MWh"], row["total_renewable_MWh"], row["demand_MWh"]]
    colors = ["#E76F51", "#2A9D8F", "#8338EC", "#E9C46A"]
    fig = go.Figure(go.Bar(
        x=cats, y=vals, marker_color=colors,
        text=[f"{v:,.0f}" for v in vals], textposition="outside",
        textfont=dict(color="#fff", size=11),
    ))
    fig.update_layout(**_base_layout("Energy Breakdown for Selected Date (MWh)", height=320))
    fig.update_yaxes(title_text="MWh")
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON CHARTS
# ══════════════════════════════════════════════════════════════════════════
MODEL_COLORS = {
    "GRU":               "#8338EC",
    "LSTM":              "#6A4FCF",
    "XGBoost":           "#F4A261",
    "Random Forest":     "#2A9D8F",
    "Linear Regression": "#E76F51",
    "Holt-Winters":      "#E9C46A",
    "ARIMA":             "#264653",
    "SARIMA":            "#457B9D",
    "Naive":             "#888",
}

def chart_model_rmse(results_df: pd.DataFrame, target: str) -> go.Figure:
    df_t = results_df[results_df["Target"] == target].sort_values("RMSE (MWh)")
    colors = [MODEL_COLORS.get(m, "#aaa") for m in df_t["Model"]]
    fig = go.Figure(go.Bar(
        y=df_t["Model"], x=df_t["RMSE (MWh)"], orientation="h",
        marker_color=colors, marker_line_color="rgba(0,0,0,0.3)",
        text=[f"{v:,.0f}" for v in df_t["RMSE (MWh)"]],
        textposition="outside", textfont=dict(color="#fff", size=11),
    ))
    tname = "Solar" if "solar" in target else "Wind"
    fig.update_layout(**_base_layout(f"{tname} Generation — RMSE Comparison (lower = better)", height=360))
    fig.update_xaxes(title_text="RMSE (MWh)")
    return fig


def chart_model_r2(results_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Solar Generation  R²", "Wind Generation  R²"])

    for col_idx, target in enumerate(["solar_MWh", "wind_MWh"], start=1):
        df_t = results_df[results_df["Target"] == target].sort_values("R²")
        colors = [MODEL_COLORS.get(m, "#aaa") for m in df_t["Model"]]
        fig.add_trace(go.Bar(
            y=df_t["Model"], x=df_t["R²"], orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in df_t["R²"]],
            textposition="outside", textfont=dict(color="#fff", size=10),
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, size=12),
        height=360, margin=dict(l=50, r=30, t=60, b=40),
        title=dict(text="R² Score by Model (higher = better)", font=dict(color="#fff", size=15)),
    )
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=GRID_COL, row=1, col=i)
        fig.update_yaxes(gridcolor=GRID_COL, row=1, col=i)
    return fig


def chart_mape_heatmap(results_df: pd.DataFrame) -> go.Figure:
    pivot = results_df.pivot(index="Model", columns="Target", values="MAPE (%)")
    pivot.columns = ["Solar MAPE %", "Wind MAPE %"]
    pivot = pivot.sort_values("Solar MAPE %")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorbar=dict(title="MAPE %", tickfont=dict(color="#ccc"), titlefont=dict(color="#ccc")),
    ))
    fig.update_layout(**_base_layout("MAPE Heatmap — Lower is Better", height=360))
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Safely convert a hex color string like '#E76F51' to 'rgba(231,111,81,0.12)'."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(150,150,150,{alpha})"


def chart_radar(results_df: pd.DataFrame, target: str) -> go.Figure:
    df_t = results_df[results_df["Target"] == target].copy()

    # Normalize metrics: 0=worst 1=best  (invert MAE/RMSE/MAPE)
    for col in ["MAE (MWh)", "RMSE (MWh)", "MAPE (%)"]:
        mx = df_t[col].max(); mn = df_t[col].min()
        df_t[f"{col}_n"] = 1 - (df_t[col] - mn) / (mx - mn + 1e-9)
    r2_min = df_t["R²"].min(); r2_max = df_t["R²"].max()
    df_t["R²_n"] = (df_t["R²"] - r2_min) / (r2_max - r2_min + 1e-9)

    cats = ["MAE", "RMSE", "MAPE", "R²"]
    fig  = go.Figure()
    # Only show top-5 models for readability
    top5 = df_t.sort_values("RMSE (MWh)").head(5)
    for _, row in top5.iterrows():
        vals = [row["MAE (MWh)_n"], row["RMSE (MWh)_n"], row["MAPE (%)_n"], row["R²_n"]]
        vals += [vals[0]]  # close polygon
        model_color = MODEL_COLORS.get(row["Model"], "#aaaaaa")
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=row["Model"],
            line=dict(color=model_color, width=1.5),
            fillcolor=_hex_to_rgba(model_color, 0.10),
        ))

    tname = "Solar" if "solar" in target else "Wind"
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, size=11),
        polar=dict(bgcolor=PLOT_BG,
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COL, tickfont=dict(color="#aaa")),
                   angularaxis=dict(gridcolor=GRID_COL, tickfont=dict(color="#ddd"))),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(text=f"{tname} — Model Performance Radar (Top 5)", font=dict(color="#fff", size=14)),
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def chart_historical_overview(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Solar Generation (MWh)", "Wind Generation (MWh)", "Total Demand (MWh)"],
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=df["time"], y=df["solar_MWh"],
                             line=dict(color="#E76F51", width=1), name="Solar",
                             fill="tozeroy", fillcolor="rgba(231,111,81,0.12)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["wind_MWh"],
                             line=dict(color="#2A9D8F", width=1), name="Wind",
                             fill="tozeroy", fillcolor="rgba(42,157,143,0.12)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["demand_MWh"],
                             line=dict(color="#E9C46A", width=1), name="Demand",
                             fill="tozeroy", fillcolor="rgba(233,196,106,0.10)"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL),
        height=520, showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40),
        title=dict(text="Historical Energy Data — 2015 to 2018", font=dict(color="#fff", size=15)),
    )
    for r in [1, 2, 3]:
        fig.update_xaxes(gridcolor=GRID_COL, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID_COL, row=r, col=1)
    return fig


def chart_monthly_avg(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["month"] = df["time"].dt.month
    monthly = df.groupby("month")[["solar_MWh", "wind_MWh", "demand_MWh"]].mean().reset_index()
    months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = [months[m-1] for m in monthly["month"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["solar_MWh"],
                         name="Solar Avg", marker_color="#E76F51"))
    fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["wind_MWh"],
                         name="Wind Avg", marker_color="#2A9D8F"))
    fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["demand_MWh"],
                             name="Demand Avg", line=dict(color="#E9C46A", width=2.5), yaxis="y2"))

    layout = _base_layout("Monthly Average Energy Production & Demand", height=360)
    layout["yaxis"]  = dict(title="Generation (MWh)", gridcolor=GRID_COL)
    layout["yaxis2"] = dict(title="Demand (MWh)", overlaying="y", side="right",
                            gridcolor=GRID_COL, showgrid=False)
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    return fig


def chart_renewable_share_hist(df: pd.DataFrame) -> go.Figure:
    rs = df["renewable_share"] * 100
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Renewable Share Over Time (%)", "Distribution of Renewable Share"])
    fig.add_trace(go.Scatter(x=df["time"], y=rs,
                             line=dict(color="#8338EC", width=0.9),
                             fill="tozeroy", fillcolor="rgba(131,56,236,0.12)",
                             name="Renewable Share"), row=1, col=1)
    fig.add_trace(go.Histogram(x=rs, nbinsx=40, marker_color="#8338EC",
                               opacity=0.8, name="Distribution"), row=1, col=2)
    fig.add_vline(x=rs.mean(), line=dict(color="#E63946", dash="dash", width=1.5), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL), height=340,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
        title=dict(text=f"Renewable Share Analysis — Mean: {rs.mean():.1f}%", font=dict(color="#fff", size=14)),
    )
    for r in [1]:
        for c in [1, 2]:
            fig.update_xaxes(gridcolor=GRID_COL, row=r, col=c)
            fig.update_yaxes(gridcolor=GRID_COL, row=r, col=c)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px 0'>
        <div style='font-size:40px'>⚡</div>
        <div style='color:#fff;font-size:18px;font-weight:700'>Microgrid Forecaster</div>
        <div style='color:#aaa;font-size:12px'>Trained on 2015–2018 Historical Data</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔮  Date Prediction", "📊  Visuals & EDA", "🏆  Model Comparison"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='color:#888;font-size:12px;line-height:1.6;padding:0 4px'>
        <b style='color:#ccc'>Dataset:</b> 1,460 daily observations<br>
        <b style='color:#ccc'>Range:</b> Jan 2015 – Dec 2018<br>
        <b style='color:#ccc'>Targets:</b> Solar · Wind · Demand<br>
        <b style='color:#ccc'>Model:</b> Ensemble (GBM + RF + Ridge)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════
df_hist, trained_models = load_and_train()
results_df  = load_model_results()
existing_df = load_existing_metrics()


# ── Shared helper: metric card (available on all pages) ──────────────────
def card(label, value, unit=""):
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="unit">{unit}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DATE PREDICTION
# ══════════════════════════════════════════════════════════════════════════
if "🔮" in page:

    st.markdown('<div class="section-header">🔮 Energy Generation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter any date or date range — the model predicts solar & wind generation based on 2015–2018 learned patterns</div>', unsafe_allow_html=True)

    # ── Input controls ──────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1.1, 1.1, 0.8])
    with col_a:
        start_date = st.date_input("📅 From Date", value=datetime.date(2022, 1, 1),
                                   min_value=datetime.date(2019, 1, 1),
                                   max_value=datetime.date(2030, 12, 31))
    with col_b:
        end_date = st.date_input("📅 To Date", value=datetime.date(2022, 1, 1),
                                 min_value=datetime.date(2019, 1, 1),
                                 max_value=datetime.date(2030, 12, 31))
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict", type="primary", use_container_width=True)

    if end_date < start_date:
        st.error("⚠️ End date must be on or after start date.")
    elif predict_btn:
        with st.spinner("Computing predictions…"):
            pred_df = predict_range(trained_models, start_date, end_date)
        n_days = len(pred_df)

        # ── Summary metric cards ───────────────────────────────────────
        st.markdown("#### 📊 Prediction Summary")
        m1, m2, m3, m4, m5 = st.columns(5)

        total_solar  = pred_df["solar_MWh"].sum()
        total_wind   = pred_df["wind_MWh"].sum()
        total_renew  = pred_df["total_renewable_MWh"].sum()
        total_demand = pred_df["demand_MWh"].sum()
        avg_share    = pred_df["renewable_share_pct"].mean()

        m1.markdown(card("☀️ Solar (Total)", f"{total_solar:,.0f}", "MWh"), unsafe_allow_html=True)
        m2.markdown(card("💨 Wind (Total)",  f"{total_wind:,.0f}",  "MWh"), unsafe_allow_html=True)
        m3.markdown(card("🌿 Total Renewable", f"{total_renew:,.0f}", "MWh"), unsafe_allow_html=True)
        m4.markdown(card("⚡ Grid Demand",   f"{total_demand:,.0f}", "MWh"), unsafe_allow_html=True)
        m5.markdown(card("🔋 Avg Clean Share", f"{avg_share:.1f}", "%"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if n_days == 1:
            # ── Single-day: bar + donut ────────────────────────────────
            row = pred_df.iloc[0]
            st.markdown(f"#### Results for **{row['date']}** ({row['day_name']})")
            ch1, ch2 = st.columns([1.6, 1])
            with ch1:
                st.plotly_chart(chart_bar_breakdown(row), use_container_width=True)
            with ch2:
                st.plotly_chart(chart_donut_single(row), use_container_width=True)

            # Detailed stats table
            st.markdown("#### 📋 Detailed Values")
            detail = pd.DataFrame({
                "Metric":        ["Solar Generation", "Wind Generation", "Total Clean Energy",
                                  "Grid Demand", "Renewable Share"],
                "Value (MWh)":   [f"{row['solar_MWh']:,.1f}", f"{row['wind_MWh']:,.1f}",
                                  f"{row['total_renewable_MWh']:,.1f}", f"{row['demand_MWh']:,.1f}", "—"],
                "Value (TWh)":   [f"{row['solar_MWh']/1e6:.4f}", f"{row['wind_MWh']/1e6:.4f}",
                                  f"{row['total_renewable_MWh']/1e6:.4f}", f"{row['demand_MWh']/1e6:.4f}", "—"],
                "Note":          ["Peak: Jun–Aug | Low: Dec–Jan",
                                  "High in winter, lower in summer",
                                  "Combined solar + wind",
                                  "Estimated from 2015–18 patterns",
                                  f"{row['renewable_share_pct']:.2f}% of demand met by clean energy"],
            })
            st.dataframe(detail, hide_index=True, use_container_width=True)

        else:
            # ── Multi-day charts ───────────────────────────────────────
            tab1, tab2, tab3 = st.tabs(["📈 Generation vs Demand", "🏗️ Stacked Area", "🔋 Renewable Share"])
            with tab1:
                st.plotly_chart(chart_forecast_lines(pred_df), use_container_width=True)
            with tab2:
                st.plotly_chart(chart_stacked_area(pred_df), use_container_width=True)
            with tab3:
                st.plotly_chart(chart_renewable_share(pred_df), use_container_width=True)

            # Peak days
            st.markdown("#### 🏆 Peak Days in Period")
            pk1, pk2, pk3 = st.columns(3)
            best_solar  = pred_df.loc[pred_df["solar_MWh"].idxmax()]
            best_wind   = pred_df.loc[pred_df["wind_MWh"].idxmax()]
            best_renew  = pred_df.loc[pred_df["total_renewable_MWh"].idxmax()]

            pk1.markdown(card("☀️ Peak Solar Day",  f"{str(best_solar['date'])} ({best_solar['day_name'][:3]})", f"{best_solar['solar_MWh']:,.0f} MWh"), unsafe_allow_html=True)
            pk2.markdown(card("💨 Peak Wind Day",   f"{str(best_wind['date'])} ({best_wind['day_name'][:3]})",   f"{best_wind['wind_MWh']:,.0f} MWh"),  unsafe_allow_html=True)
            pk3.markdown(card("🌿 Peak Renewable Day", f"{str(best_renew['date'])} ({best_renew['day_name'][:3]})", f"{best_renew['total_renewable_MWh']:,.0f} MWh"), unsafe_allow_html=True)

            # Raw data table (collapsible)
            with st.expander(f"📋 View all {n_days} daily predictions"):
                display_cols = {
                    "date": "Date", "day_name": "Day",
                    "solar_MWh": "Solar (MWh)", "wind_MWh": "Wind (MWh)",
                    "total_renewable_MWh": "Total Renewable (MWh)",
                    "demand_MWh": "Demand (MWh)", "renewable_share_pct": "Renewable Share %"
                }
                show_df = pred_df.rename(columns=display_cols)[list(display_cols.values())]
                st.dataframe(show_df, hide_index=True, use_container_width=True)

                # Download button
                csv_data = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download as CSV",
                    data=csv_data,
                    file_name=f"prediction_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )

    else:
        # ── Placeholder when nothing predicted yet ─────────────────────
        st.info("👆 Select a date range above and click **⚡ Predict** to see the energy production forecast.")
        st.markdown("""
        **How it works:**
        - The model was trained on **1,460 days** of real microgrid data (2015–2018)
        - It learned the **seasonal solar cycle** (summer peaks, winter troughs), **wind patterns** (winter storms), and **demand behaviour**
        - Enter any date — past, present or future — to get predicted generation values
        """)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — VISUALS & EDA
# ══════════════════════════════════════════════════════════════════════════
elif "📊" in page:

    st.markdown('<div class="section-header">📊 Historical Data Visualisations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Exploratory analysis of the 2015–2018 training dataset</div>', unsafe_allow_html=True)

    tab_hist, tab_monthly, tab_share, tab_saved = st.tabs([
        "📈 Historical Overview", "📅 Monthly Patterns", "🌿 Renewable Share", "🖼️ Saved Plots"
    ])

    with tab_hist:
        st.plotly_chart(chart_historical_overview(df_hist), use_container_width=True)

        # Quick stats
        st.markdown("#### Dataset Statistics")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Days", f"{len(df_hist):,}")
        sc2.metric("Avg Solar / day", f"{df_hist['solar_MWh'].mean():,.0f} MWh")
        sc3.metric("Avg Wind / day",  f"{df_hist['wind_MWh'].mean():,.0f} MWh")
        sc4.metric("Avg Demand / day", f"{df_hist['demand_MWh'].mean():,.0f} MWh")

        # Descriptive table
        with st.expander("📋 Full Descriptive Statistics"):
            cols = ["solar_MWh", "wind_MWh", "demand_MWh", "renewable_share"]
            desc = df_hist[cols].describe().T.round(2)
            desc.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
            st.dataframe(desc, use_container_width=True)

    with tab_monthly:
        st.plotly_chart(chart_monthly_avg(df_hist), use_container_width=True)

        # Insight callout
        solar_peak = df_hist.groupby(df_hist["time"].dt.month)["solar_MWh"].mean().idxmax()
        wind_peak  = df_hist.groupby(df_hist["time"].dt.month)["wind_MWh"].mean().idxmax()
        months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        st.info(f"☀️ Solar peaks in **{months_lbl[solar_peak-1]}** · 💨 Wind peaks in **{months_lbl[wind_peak-1]}** — complementary renewable sources.")

    with tab_share:
        st.plotly_chart(chart_renewable_share_hist(df_hist), use_container_width=True)
        rs_pct = df_hist["renewable_share"] * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Renewable Share", f"{rs_pct.mean():.1f}%")
        c2.metric("Maximum Share (single day)", f"{rs_pct.max():.1f}%")
        c3.metric("Days above 40% renewable", f"{(rs_pct > 40).sum()}")

    with tab_saved:
        st.markdown("#### Generated Analysis Plots (from model training run)")
        saved_figs = {
            "Overview": "01_overview.png",
            "Monthly Boxplots": "02_monthly_boxplots.png",
            "Annual Totals": "03_annual_totals.png",
            "Correlation": "04_correlation.png",
            "Actual vs Forecast": "07_actual_vs_forecast.png",
            "Error Distributions": "08_error_distributions.png",
            "Solar Decomposition": "decomp_solar_MWh.png",
            "Wind Decomposition": "decomp_wind_MWh.png",
            "Solar ACF/PACF": "acf_pacf_solar_MWh.png",
            "Wind ACF/PACF": "acf_pacf_wind_MWh.png",
            "Solar Predictions": "predictions_solar_MWh.png",
            "Wind Predictions": "predictions_wind_MWh.png",
            "Future Forecast": "future_forecast.png",
        }
        select = st.selectbox("Choose a plot:", list(saved_figs.keys()))
        fig_path = os.path.join(FIGS_DIR, saved_figs[select])
        if os.path.exists(fig_path):
            st.image(fig_path, use_column_width=True)
        else:
            st.warning(f"Plot not found: `{saved_figs[select]}`  — Run `main.py` first to generate it.")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
elif "🏆" in page:

    st.markdown('<div class="section-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Evaluation of all 9 forecasting models on the 2018 test set (unseen data)</div>', unsafe_allow_html=True)

    if results_df is None:
        st.warning("⚠️  `outputs/results/model_results.csv` not found. Please run `main.py` first to train and evaluate all models.")
    else:
        # ── Leader-boards ────────────────────────────────────────────────
        st.markdown("#### 🥇 Best Models (by RMSE on 2018 test data)")
        b1, b2 = st.columns(2)
        for col, target, label in [(b1, "solar_MWh", "☀️ Solar"), (b2, "wind_MWh", "💨 Wind")]:
            best = results_df[results_df["Target"] == target].sort_values("RMSE (MWh)").iloc[0]
            col.markdown(card(
                f"{label} — Best Model",
                best["Model"],
                f"RMSE {best['RMSE (MWh)']:,.0f} MWh · MAPE {best['MAPE (%)']:.1f}% · R² {best['R²']:.3f}"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Full scores tables ────────────────────────────────────────────
        tab_solar, tab_wind, tab_charts, tab_radar, tab_existing = st.tabs([
            "☀️ Solar Scores", "💨 Wind Scores", "📊 Charts", "🕸️ Radar", "🔎 Existing Forecast"
        ])

        def style_results(df_t):
            """Highlight best row in green, worst in red."""
            df_t = df_t.drop(columns=["Target"]).reset_index(drop=True)
            best_idx  = int(df_t["RMSE (MWh)"].idxmin())
            worst_idx = int(df_t["RMSE (MWh)"].idxmax())

            def highlight_row(row):
                if row.name == best_idx:
                    return ["background-color: rgba(46,213,115,0.22); font-weight:600"] * len(row)
                elif row.name == worst_idx:
                    return ["background-color: rgba(231,76,60,0.15)"] * len(row)
                return [""] * len(row)

            return df_t.style.apply(highlight_row, axis=1)

        with tab_solar:
            df_solar = results_df[results_df["Target"] == "solar_MWh"].sort_values("RMSE (MWh)")
            st.dataframe(style_results(df_solar), use_container_width=True, hide_index=True)

        with tab_wind:
            df_wind = results_df[results_df["Target"] == "wind_MWh"].sort_values("RMSE (MWh)")
            st.dataframe(style_results(df_wind), use_container_width=True, hide_index=True)

        with tab_charts:
            st.plotly_chart(chart_model_rmse(results_df, "solar_MWh"), use_container_width=True)
            st.plotly_chart(chart_model_rmse(results_df, "wind_MWh"),  use_container_width=True)
            st.plotly_chart(chart_model_r2(results_df),                use_container_width=True)
            st.plotly_chart(chart_mape_heatmap(results_df),            use_container_width=True)

        with tab_radar:
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(chart_radar(results_df, "solar_MWh"), use_container_width=True)
            with r2:
                st.plotly_chart(chart_radar(results_df, "wind_MWh"),  use_container_width=True)

        with tab_existing:
            st.markdown("#### Pre-Existing Operational Forecast Accuracy")
            st.markdown("These forecasts were already in the dataset — **not** generated by our models. They represent the original operational forecast system.")
            if existing_df is not None:
                st.dataframe(existing_df, use_container_width=True, hide_index=True)
                # Compare bar
                fig_ex = go.Figure()
                for metric, color in [("MAPE (%)", "#E76F51"), ("R²", "#2A9D8F")]:
                    if metric in existing_df.columns:
                        fig_ex.add_trace(go.Bar(
                            x=existing_df.iloc[:, 0] if "Variable" in existing_df.columns else existing_df.index,
                            y=existing_df[metric],
                            name=metric, marker_color=color
                        ))
                fig_ex.update_layout(**_base_layout("Existing Forecast: MAPE & R² by Variable", height=300))
                st.plotly_chart(fig_ex, use_container_width=True)
            else:
                st.info("Run `main.py` to generate `existing_forecast_metrics.csv`.")

