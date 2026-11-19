#!/opt/anaconda3/bin/python3
"""
predict.py — Interactive Microgrid Renewable Energy Predictor.
Allows users to enter ANY date range (e.g. from YYYY-MM-DD to YYYY-MM-DD)
and generates predicted Solar and Wind generation, combined renewable production,
and estimated renewable share based on patterns learned from 2015-2018 historical data.

Usage:
  Interactive Mode:
    python3 predict.py

  Command-line Mode:
    python3 predict.py --start 2019-01-01 --end 2019-12-31
    python3 predict.py --start 2020-06-01 --end 2020-08-31 --model ensemble
"""

import os
import sys
import argparse
import datetime
import warnings

# Use headless backend for saving figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "daily_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# Feature Extraction (Cyclical, Seasonal Harmonics, Calendar, Trend)
# -------------------------------------------------------------------------
def extract_features(dt_series: pd.Series) -> pd.DataFrame:
    """
    Transforms datetime series into rich harmonic and calendar features.
    Captures:
      - 3 Fourier harmonics of day-of-year (annual solar & wind cycles)
      - Monthly cyclical variations
      - Weekly cycles & weekend indicators
      - Long-term capacity & efficiency trend
    """
    doy = dt_series.dt.dayofyear
    dow = dt_series.dt.dayofweek
    month = dt_series.dt.month
    year_offset = (dt_series.dt.year - 2015) + (doy / 365.25)

    feats = pd.DataFrame({
        "doy_sin1": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos1": np.cos(2 * np.pi * doy / 365.25),
        "doy_sin2": np.sin(4 * np.pi * doy / 365.25),
        "doy_cos2": np.cos(4 * np.pi * doy / 365.25),
        "doy_sin3": np.sin(6 * np.pi * doy / 365.25),
        "doy_cos3": np.cos(6 * np.pi * doy / 365.25),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(float),
        "year_trend": year_offset,
    })
    return feats


# -------------------------------------------------------------------------
# Model Training Class
# -------------------------------------------------------------------------
class MicrogridPredictor:
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type.lower()
        self.models = {}
        self.targets = ["solar_MWh", "wind_MWh", "demand_MWh"]
        self.is_trained = False

    def train(self, data_path: str = DATA_PATH):
        """Train models on the 2015-2018 historical dataset."""
        df = pd.read_csv(data_path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        X_train = extract_features(df["time"])

        for target in self.targets:
            y_train = df[target].values

            if self.model_type == "gradient_boost":
                m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, random_state=42)
                m.fit(X_train, y_train)
                self.models[target] = m

            elif self.model_type == "random_forest":
                m = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
                m.fit(X_train, y_train)
                self.models[target] = m

            elif self.model_type == "linear":
                m = Ridge(alpha=1.0)
                m.fit(X_train, y_train)
                self.models[target] = m

            else:  # 'ensemble' (blend of Gradient Boost + Random Forest + Ridge)
                m1 = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, random_state=42)
                m2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                m3 = Ridge(alpha=1.0)

                m1.fit(X_train, y_train)
                m2.fit(X_train, y_train)
                m3.fit(X_train, y_train)
                self.models[target] = (m1, m2, m3)

        self.is_trained = True

    def predict_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Predict solar, wind, and demand for all days between start_date and end_date."""
        if not self.is_trained:
            self.train()

        date_range = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
        if len(date_range) == 0:
            raise ValueError(f"End date ({end_date}) must be greater than or equal to start date ({start_date}).")

        X_future = extract_features(pd.Series(date_range))
        results = {"date": date_range.date}

        for target in self.targets:
            if self.model_type == "ensemble":
                m1, m2, m3 = self.models[target]
                p1 = m1.predict(X_future)
                p2 = m2.predict(X_future)
                p3 = m3.predict(X_future)
                # Weighted blend: 50% Gradient Boost, 35% Random Forest, 15% Ridge
                pred = 0.50 * p1 + 0.35 * p2 + 0.15 * p3
            else:
                pred = self.models[target].predict(X_future)

            # Energy generation cannot be negative
            pred = np.clip(pred, a_min=0, a_max=None)
            results[target] = np.round(pred, 1)

        pred_df = pd.DataFrame(results)
        pred_df["day_name"] = [d.strftime("%A") for d in date_range]
        pred_df["total_renewable_MWh"] = pred_df["solar_MWh"] + pred_df["wind_MWh"]
        pred_df["renewable_share_pct"] = np.round(
            (pred_df["total_renewable_MWh"] / pred_df["demand_MWh"]) * 100, 2
        )

        # Clean column names for user display
        pred_df = pred_df.rename(columns={
            "solar_MWh": "solar_predicted_MWh",
            "wind_MWh": "wind_predicted_MWh",
            "demand_MWh": "demand_predicted_MWh",
        })

        return pred_df


# -------------------------------------------------------------------------
# Visualization Function
# -------------------------------------------------------------------------
def plot_prediction_results(df: pd.DataFrame, start_str: str, end_str: str, save_path: str):
    """Generates an aesthetic dashboard: bar chart for single day, time-series for date ranges."""
    dates = pd.to_datetime(df["date"])

    if len(df) == 1:
        # Single Date View: Bar Chart + Renewable Share Gauge
        row = df.iloc[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Microgrid Energy Prediction for {row['date']} ({row['day_name']})",
                     fontsize=14, fontweight="bold")

        categories = ["Solar", "Wind", "Total Renewable", "Total Demand"]
        values = [row["solar_predicted_MWh"], row["wind_predicted_MWh"],
                  row["total_renewable_MWh"], row["demand_predicted_MWh"]]
        colors = ["#E76F51", "#2A9D8F", "#8338EC", "#264653"]

        bars = ax1.bar(categories, values, color=colors, width=0.55, edgecolor="black", alpha=0.85)
        ax1.set_ylabel("Electricity (MWh)", fontsize=11, fontweight="bold")
        ax1.set_title("Energy Breakdown (MWh)", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:,.0f} MWh",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Donut Chart for Renewable vs Non-Renewable
        renew_share = row["renewable_share_pct"]
        non_renew_share = max(0.0, 100.0 - renew_share)
        ax2.pie([renew_share, non_renew_share],
                labels=[f"Renewable\n({renew_share:.1f}%)", f"Grid / Thermal\n({non_renew_share:.1f}%)"],
                colors=["#2A9D8F", "#E9C46A"],
                startangle=90, counterclock=False,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
        ax2.set_title("Renewable Share (% of Demand)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=130)
        plt.close()
        return

    # Multi-day Time-Series View
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Microgrid Renewable Energy Forecast: {start_str} to {end_str}",
                 fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Solar and Wind Daily Generation
    ax1 = axes[0]
    ax1.plot(dates, df["solar_predicted_MWh"], color="#E76F51", lw=1.8, label="Predicted Solar (MWh)")
    ax1.plot(dates, df["wind_predicted_MWh"], color="#2A9D8F", lw=1.8, label="Predicted Wind (MWh)")
    ax1.set_ylabel("Daily Generation (MWh)", fontsize=11, fontweight="bold")
    ax1.set_title("Solar & Wind Daily Generation Forecast", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Stacked Area Chart (Solar + Wind vs Total Demand)
    ax2 = axes[1]
    ax2.fill_between(dates, 0, df["solar_predicted_MWh"], color="#E76F51", alpha=0.6, label="Solar Contribution")
    ax2.fill_between(dates, df["solar_predicted_MWh"], df["total_renewable_MWh"],
                     color="#2A9D8F", alpha=0.6, label="Wind Contribution")
    ax2.plot(dates, df["demand_predicted_MWh"], color="#264653", lw=2, linestyle="--", label="Total Electricity Demand")
    ax2.set_ylabel("Energy (MWh)", fontsize=11, fontweight="bold")
    ax2.set_title("Clean Energy Production vs Total Grid Demand", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Renewable Share (%)
    ax3 = axes[2]
    mean_share = df["renewable_share_pct"].mean()
    ax3.plot(dates, df["renewable_share_pct"], color="#8338EC", lw=2, label="Renewable Share (%)")
    ax3.axhline(mean_share, color="#E63946", linestyle=":", lw=1.8, label=f"Average Share ({mean_share:.1f}%)")
    ax3.fill_between(dates, 0, df["renewable_share_pct"], color="#8338EC", alpha=0.15)
    ax3.set_ylabel("Renewable Share (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax3.set_title("Proportion of Electricity Demand Met by Renewables", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, max(60, df["renewable_share_pct"].max() * 1.15))
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    if len(dates) > 60:
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates)//180)))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


# -------------------------------------------------------------------------
# Display Report
# -------------------------------------------------------------------------
def display_report(df: pd.DataFrame, start_str: str, end_str: str, model_type: str, csv_path: str, img_path: str):
    total_days = len(df)
    tot_solar = df["solar_predicted_MWh"].sum()
    tot_wind = df["wind_predicted_MWh"].sum()
    tot_renew = df["total_renewable_MWh"].sum()
    tot_demand = df["demand_predicted_MWh"].sum()
    avg_share = df["renewable_share_pct"].mean()

    peak_solar_idx = df["solar_predicted_MWh"].idxmax()
    peak_wind_idx = df["wind_predicted_MWh"].idxmax()
    peak_renew_idx = df["total_renewable_MWh"].idxmax()

    print("\n" + "=" * 80)
    print("☀️  MICROGRID RENEWABLE ENERGY PREDICTION REPORT")
    print("=" * 80)
    print(f"📅 Requested Period : {start_str}  to  {end_str}  ({total_days:,} days)")
    print(f"🧠 Model Engine     : {model_type.upper()} (Trained on 2015-2018 Historical Patterns)")
    print("-" * 80)

    print("📊 TOTAL PRODUCTION OVER REQUESTED TIMEFRAME:")
    print(f"  • Solar Generation    : {tot_solar:14,.1f} MWh   ({tot_solar / 1e6:8.3f} TWh)")
    print(f"  • Wind Generation     : {tot_wind:14,.1f} MWh   ({tot_wind / 1e6:8.3f} TWh)")
    print(f"  • Total Clean Energy  : {tot_renew:14,.1f} MWh   ({tot_renew / 1e6:8.3f} TWh)")
    print(f"  • Total Grid Demand   : {tot_demand:14,.1f} MWh   ({tot_demand / 1e6:8.3f} TWh)")
    print(f"  • Avg Renewable Share : {avg_share:14.2f} %")
    print("-" * 80)

    print("📈 DAILY GENERATION AVERAGES:")
    print(f"  • Solar Average       : {df['solar_predicted_MWh'].mean():14,.1f} MWh/day")
    print(f"  • Wind Average        : {df['wind_predicted_MWh'].mean():14,.1f} MWh/day")
    print(f"  • Total Clean Average : {df['total_renewable_MWh'].mean():14,.1f} MWh/day")
    print("-" * 80)

    print("🏆 PEAK PRODUCTION DAYS:")
    print(f"  • Highest Solar Day   : {df.loc[peak_solar_idx, 'date']} ({df.loc[peak_solar_idx, 'day_name']}) — {df.loc[peak_solar_idx, 'solar_predicted_MWh']:,.1f} MWh")
    print(f"  • Highest Wind Day    : {df.loc[peak_wind_idx, 'date']} ({df.loc[peak_wind_idx, 'day_name']}) — {df.loc[peak_wind_idx, 'wind_predicted_MWh']:,.1f} MWh")
    print(f"  • Highest Clean Day   : {df.loc[peak_renew_idx, 'date']} ({df.loc[peak_renew_idx, 'day_name']}) — {df.loc[peak_renew_idx, 'total_renewable_MWh']:,.1f} MWh (Share: {df.loc[peak_renew_idx, 'renewable_share_pct']:.1f}%)")
    print("-" * 80)

    print("\n📋 PREVIEW OF DAILY PREDICTIONS (First 5 days):")
    preview_cols = ["date", "day_name", "solar_predicted_MWh", "wind_predicted_MWh", "total_renewable_MWh", "renewable_share_pct"]
    print(df[preview_cols].head(5).to_string(index=False))

    if len(df) > 10:
        print("\n... (Last 3 days):")
        print(df[preview_cols].tail(3).to_string(index=False))

    print("\n" + "=" * 80)
    print(f"💾 Results CSV saved to: {csv_path}")
    print(f"📊 Visual Plot saved to: {img_path}")
    print("=" * 80 + "\n")


# -------------------------------------------------------------------------
# Interactive or CLI Execution
# -------------------------------------------------------------------------
def run_prediction(start_str: str, end_str: str, model_type: str = "ensemble"):
    # Validate date formats
    try:
        s_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        e_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        print("\n❌ Error: Invalid date format. Please use YYYY-MM-DD (e.g. 2019-01-01).")
        return

    if e_date < s_date:
        print(f"\n❌ Error: End date ({end_str}) cannot be earlier than start date ({start_str}).")
        return

    print(f"\n⏳ Training model on 2015-2018 data and predicting for {start_str} to {end_str}...")
    predictor = MicrogridPredictor(model_type=model_type)
    predictor.train(DATA_PATH)
    pred_df = predictor.predict_range(start_str, end_str)

    # Save CSV
    csv_filename = f"prediction_{start_str}_to_{end_str}.csv"
    csv_path = os.path.join(OUT_DIR, csv_filename)
    pred_df.to_csv(csv_path, index=False)

    # Save Plot
    img_filename = f"forecast_{start_str}_to_{end_str}.png"
    img_path = os.path.join(OUT_DIR, img_filename)
    plot_prediction_results(pred_df, start_str, end_str, img_path)

    # Print summary
    display_report(pred_df, start_str, end_str, model_type, csv_path, img_path)


def main():
    parser = argparse.ArgumentParser(description="Predict solar and wind energy for custom date or date range.")
    parser.add_argument("--date", type=str, help="Single target date in YYYY-MM-DD format (e.g. 2022-02-02)")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD format (e.g. 2019-01-01)")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD format (e.g. 2019-06-30)")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "gradient_boost", "random_forest", "linear"],
                        help="Model algorithm to use for prediction (default: ensemble)")

    args = parser.parse_args()

    # If single date provided
    if args.date:
        run_prediction(args.date, args.date, args.model)
        return

    # If date range provided
    if args.start:
        end_val = args.end if args.end else args.start
        run_prediction(args.start, end_val, args.model)
        return

    # Otherwise, interactive terminal prompt
    print("\n" + "=" * 65)
    print("  ☀️ MICROGRID ENERGY FORECASTER — DATE-BASED PREDICTOR")
    print("=" * 65)
    print("Trained on historical data (2015-2018) to learn solar & wind patterns.\n")

    try:
        user_input = input("👉 Enter Date (YYYY-MM-DD) [e.g. 2022-02-02]: ").strip()
        if not user_input:
            user_input = "2022-02-02"

        # Check if user entered a range with 'to' or space or comma
        if " to " in user_input.lower():
            parts = user_input.lower().split(" to ")
            start_date, end_date = parts[0].strip(), parts[1].strip()
        elif " " in user_input and len(user_input.split()) == 2:
            parts = user_input.split()
            start_date, end_date = parts[0].strip(), parts[1].strip()
        else:
            # Single date by default!
            start_date = user_input
            range_prompt = input("👉 (Optional) Enter End Date if you want a range, or press [ENTER] for this single day: ").strip()
            end_date = range_prompt if range_prompt else start_date

        run_prediction(start_date, end_date, "ensemble")

    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

