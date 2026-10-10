"""
app.py — Streamlit Dashboard
Helicopter Engine Health Monitoring & Parameter Forecasting
Sikorsky S-76A/C+  |  Research Prototype

Run: streamlit run app.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S-76A/C+ Engine Health Monitor",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PROC = BASE / "data" / "processed"
REP  = BASE / "reports"
FIG  = BASE / "results" / "figures"
MET  = BASE / "results" / "metrics"
MOD  = BASE / "models"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1b2a; }
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
.metric-card {
    background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
    border: 1px solid #2a4a6a;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    color: #e8f4f8;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #4db8ff; }
.metric-label { font-size: 0.85rem; color: #8ab4c8; margin-top: 4px; }
.disclaimer {
    background: #2a1a0a;
    border-left: 4px solid #ff8c00;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #ffd0a0;
}
.rul-unsupported {
    background: #1a0a0a;
    border-left: 4px solid #ff4444;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #ffaaaa;
}
h1, h2, h3 { color: #e8f4f8; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading consolidated dataset …")
def load_data():
    p = PROC / "consolidated_dataset.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(show_spinner="Loading feature set …")
def load_features():
    p = PROC / "features.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_flight_report():
    p = REP / "flight_duration_report.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_metrics(name):
    p = MET / name
    if not p.exists():
        return None
    if name.endswith(".csv"):
        return pd.read_csv(p)
    return open(p).read()


PARAM_LABELS = {
    "E1NF_S76C":          "Engine 1 N1 Speed (%)",
    "E2NF_S76C":          "Engine 2 N1 Speed (%)",
    "E1TQ_S76C":          "Engine 1 Torque (%)",
    "E2TQ_S76C":          "Engine 2 Torque (%)",
    "RTRSPD":             "Rotor Speed (%)",
    "OAT":                "Outside Air Temp (°C)",
    "Indicated_Airspeed": "Indicated Airspeed (kn)",
    "PRALT":              "Pressure Altitude (ft)",
}

NORMAL_BANDS = {
    "E1NF_S76C":          (95.0, 110.0),
    "E2NF_S76C":          (60.0,  80.0),
    "E1TQ_S76C":          (0.0,  110.0),
    "E2TQ_S76C":          (0.0,  110.0),
    "RTRSPD":             (95.0, 110.0),
    "OAT":                (-40.0, 55.0),
    "Indicated_Airspeed": (0.0,  155.0),
    "PRALT":              (-300.0, 10000.0),
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚁 S-76A/C+ Monitor")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "🎯 Live Model Predictor",
        "🤖 Health Model",
        "📈 Torque Forecasting",
        "🛫 Flight Duration",
        "📊 Parameter Analysis",
        "⚠️ Exceedance Analysis",
        "🔬 RUL Assessment",
        "📋 Reports",
    ])
    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <b>Research Prototype</b><br>
    Not certified for aviation maintenance or flight-safety decisions.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("🚁 Helicopter Engine Health Monitoring System")
    st.markdown("**Aircraft:** Sikorsky S-76A/C+  |  **Engines:** Two Allison 250-C30S turboshaft")
    st.markdown("---")

    df = load_data()
    if df is None:
        st.warning("Consolidated dataset not found. Run `python src/run_pipeline.py` first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['source_file'].nunique()}</div>
            <div class='metric-label'>Recording Sessions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        hours = len(df) / 3600
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{hours:.0f}h</div>
            <div class='metric-label'>Total Recorded Time</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>1-Hz Data Points</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        n_cols = len([c for c in PARAM_LABELS if c in df.columns])
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{n_cols}</div>
            <div class='metric-label'>Engine Parameters</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Recording Sessions")
    inv = load_metrics("../reports/dataset_inventory.csv") if (REP / "dataset_inventory.csv").exists() else None
    if inv is not None:
        show_cols = ["filename", "recording_date", "n_rows", "n_cols", "file_size_mb"]
        show_cols = [c for c in show_cols if c in inv.columns]
        st.dataframe(inv[show_cols], use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Schema — Identified Engine Parameters")
    schema_data = [
        {"Parameter": k, "Unit": v.split("(")[-1].rstrip(")") if "(" in v else "—", "Description": v}
        for k, v in PARAM_LABELS.items() if k in df.columns
    ]
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE MODEL PREDICTOR & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Live Model Predictor" in page:
    st.title("🎯 Live Model Predictor & Inference Engine")
    st.markdown("""
    Test and run the pre-trained machine learning models on **custom flight simulator scenarios**
    or **score recorded flight segments** in real time.
    """)

    from predict import load_health_model, load_forecasting_model
    try:
        health_clf, health_meta = load_health_model()
        forecast_reg, forecast_meta = load_forecasting_model()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models_loaded = False

    if models_loaded:
        st.success(f"✅ Models Loaded: Health Classifier (**{health_meta.get('best_model', 'Random Forest')}**) & Forecasting Model (**{forecast_meta.get('best_model', 'XGBoost')}**)")

        mode = st.radio("Select Prediction Mode", ["🎮 Flight Condition Simulator", "🛫 Score Historical Flight Window"], horizontal=True)

        if mode == "🎮 Flight Condition Simulator":
            st.markdown("### 1. Set Operating Conditions")

            preset = st.selectbox("Choose an Operational Preset or Customize Below:", [
                "Custom",
                "Normal Steady Cruise (Recommended)",
                "High Power Takeoff / Climb",
                "Torque Exceedance Scenario (>110% TQ)",
                "Gas Generator Overspeed (>110% N1)",
                "Engine 1 Oil Pressure Warning",
            ])

            defaults = {
                "e1tq": 65.0, "e2tq": 65.0, "e1nf": 107.0, "e2nf": 68.0,
                "rtr": 107.0, "ias": 125.0, "pralt": 2500.0, "oat": 22.0,
                "e1oil": "Off", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"
            }
            if preset == "Normal Steady Cruise (Recommended)":
                defaults = {"e1tq": 68.0, "e2tq": 68.0, "e1nf": 107.0, "e2nf": 68.0, "rtr": 107.0, "ias": 120.0, "pralt": 3000.0, "oat": 20.0, "e1oil": "Off", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"}
            elif preset == "High Power Takeoff / Climb":
                defaults = {"e1tq": 98.0, "e2tq": 98.0, "e1nf": 108.0, "e2nf": 72.0, "rtr": 107.0, "ias": 85.0, "pralt": 800.0, "oat": 28.0, "e1oil": "Off", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"}
            elif preset == "Torque Exceedance Scenario (>110% TQ)":
                defaults = {"e1tq": 118.0, "e2tq": 115.0, "e1nf": 108.0, "e2nf": 75.0, "rtr": 106.0, "ias": 135.0, "pralt": 1500.0, "oat": 24.0, "e1oil": "Off", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"}
            elif preset == "Gas Generator Overspeed (>110% N1)":
                defaults = {"e1tq": 75.0, "e2tq": 72.0, "e1nf": 114.0, "e2nf": 68.0, "rtr": 107.0, "ias": 115.0, "pralt": 4000.0, "oat": 18.0, "e1oil": "Off", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"}
            elif preset == "Engine 1 Oil Pressure Warning":
                defaults = {"e1tq": 50.0, "e2tq": 70.0, "e1nf": 104.0, "e2nf": 68.0, "rtr": 105.0, "ias": 100.0, "pralt": 2000.0, "oat": 25.0, "e1oil": "On", "e2oil": "Off", "xmsnoil": "Off", "batt": "Off"}

            c1, c2, c3 = st.columns(3)
            with c1:
                e1tq = st.slider("Engine 1 Torque — E1TQ (%)", 0.0, 130.0, float(defaults["e1tq"]), step=0.5)
                e2tq = st.slider("Engine 2 Torque — E2TQ (%)", 0.0, 130.0, float(defaults["e2tq"]), step=0.5)
                rtr = st.slider("Rotor Speed — RTRSPD (%)", 80.0, 120.0, float(defaults["rtr"]), step=0.5)
            with c2:
                e1nf = st.slider("Engine 1 N1 Speed — E1NF (%)", 50.0, 120.0, float(defaults["e1nf"]), step=0.5)
                e2nf = st.slider("Engine 2 N1 Speed — E2NF (%)", 50.0, 120.0, float(defaults["e2nf"]), step=0.5)
                ias = st.slider("Indicated Airspeed (knots)", 0.0, 180.0, float(defaults["ias"]), step=1.0)
            with c3:
                pralt = st.slider("Pressure Altitude (ft)", -300.0, 10000.0, float(defaults["pralt"]), step=50.0)
                oat = st.slider("Outside Air Temp — OAT (°C)", -30.0, 50.0, float(defaults["oat"]), step=0.5)
                e1oil = st.checkbox("Engine 1 Oil Pressure Warning (On)", value=(defaults["e1oil"] == "On"))
                batt = st.checkbox("Battery Overtemp Warning (On)", value=(defaults["batt"] == "On"))

            st.markdown("---")
            run_btn = st.button("🚀 Run Models & Predict Engine State", type="primary", use_container_width=True)

            if run_btn:
                row_dict = {
                    "E1TQ_S76C": e1tq, "E2TQ_S76C": e2tq,
                    "E1NF_S76C": e1nf, "E2NF_S76C": e2nf,
                    "RTRSPD": rtr, "OAT": oat,
                    "Indicated_Airspeed": ias, "PRALT": pralt,
                    "E1TQ_S76C_roll30s_mean": e1tq, "E1TQ_S76C_roll30s_std": 0.5,
                    "E1NF_S76C_roll30s_mean": e1nf, "E2NF_S76C_roll30s_mean": e2nf,
                    "E2TQ_S76C_roll30s_mean": e2tq,
                    "E1NF_S76C_roll300s_mean": e1nf, "E1TQ_S76C_roll300s_mean": e1tq,
                    "E1NF_S76C_roll30s_std": 0.2,
                    "cum_op_seconds": 3600.0,
                    "E1_OIL_PRS_bin": 1 if e1oil else 0,
                    "E2_OIL_PRS_bin": 0,
                    "BATT_OVERTEMP_bin": 1 if batt else 0,
                    "XMSN_OIL_PRS_bin": 0,
                    "XMSN_OIL_TEMP_bin": 0,
                }
                input_df = pd.DataFrame([row_dict])

                # 1. Health Classification
                health_feats = [c for c in health_meta["features"] if c in input_df.columns]
                X_h = input_df[health_feats]
                health_prob = health_clf.predict_proba(X_h)[0, 1] if hasattr(health_clf, "predict_proba") else float(health_clf.predict(X_h)[0])
                health_pred = int(health_clf.predict(X_h)[0])

                # 2. Torque Forecasting
                fc_feats = [c for c in forecast_meta["features"] if c in input_df.columns]
                X_f = input_df[fc_feats]
                pred_tq = float(forecast_reg.predict(X_f)[0])
                tq_delta = pred_tq - e1tq

                st.subheader("📊 Real-Time Model Prediction Results")
                rc1, rc2 = st.columns(2)

                with rc1:
                    if health_pred == 1 or health_prob > 0.5:
                        st.error(f"### ⚠️ Health Status: EXCEEDANCE DETECTED\n**Exceedance Probability:** `{health_prob*100:.1f}%`\n\nThe model detected one or more parameters operating outside normal Allison 250-C30S turbine envelopes.")
                    else:
                        st.success(f"### ✅ Health Status: NORMAL OPERATION\n**Normal Operation Confidence:** `{(1 - health_prob)*100:.1f}%`\n\nAll engine parameters within safe continuous operating envelopes.")

                with rc2:
                    st.metric(
                        label="Engine 1 Torque Next-Second Forecast (E1TQ at t+1)",
                        value=f"{pred_tq:.2f} %",
                        delta=f"{tq_delta:+.2f} % from current {e1tq:.1f}%",
                    )
                    st.caption(f"Forecast produced by trained {forecast_meta.get('best_model', 'XGBoost')} dynamics regressor.")

        elif mode == "🛫 Score Historical Flight Window":
            st.markdown("### 2. Batch Score Recorded FDR Telemetry")
            feat_df = load_features()
            if feat_df is None:
                st.warning("Processed feature dataset not found.")
            else:
                files = feat_df["source_file"].unique()
                sel_file = st.selectbox("Select Flight Recording Session:", files)
                file_data = feat_df[feat_df["source_file"] == sel_file].sort_values("elapsed_seconds").reset_index(drop=True)

                max_s = int(file_data["elapsed_seconds"].max())
                st.write(f"Session contains **{len(file_data):,} seconds** of flight data.")

                start_s = st.slider("Select Start Time (seconds):", 0, max(0, max_s - 120), 1000, step=10)
                window_len = st.select_slider("Select Window Duration:", options=[30, 60, 120, 300, 600], value=120)

                sub_df = file_data[(file_data["elapsed_seconds"] >= start_s) & (file_data["elapsed_seconds"] < start_s + window_len)].copy()

                if st.button("⚡ Score Selected Flight Segment", type="primary"):
                    with st.spinner("Scoring telemetry window..."):
                        # Predict health
                        hf = [c for c in health_meta["features"] if c in sub_df.columns]
                        sub_df["exceedance_prob"] = health_clf.predict_proba(sub_df[hf])[:, 1]
                        sub_df["exceedance_pred"] = health_clf.predict(sub_df[hf])

                        # Predict next torque
                        ff = [c for c in forecast_meta["features"] if c in sub_df.columns]
                        sub_df["E1TQ_pred_next"] = forecast_reg.predict(sub_df[ff])

                        # Plot Actual vs Predicted Torque
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(sub_df["elapsed_seconds"], sub_df["E1TQ_S76C"], label="Actual E1 Torque (t)", color="#3498db", lw=1.5)
                        ax.plot(sub_df["elapsed_seconds"], sub_df["E1TQ_pred_next"], label="Model Predicted Torque (t+1)", color="#e74c3c", lw=1.5, linestyle="--")
                        ax.set_title(f"Actual vs Predicted Torque — {sel_file} (t={start_s}s to {start_s+window_len}s)", fontsize=11)
                        ax.set_xlabel("Elapsed Time (seconds)")
                        ax.set_ylabel("Torque (%)")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        # Exceedance Timeline
                        fig2, ax2 = plt.subplots(figsize=(12, 2.5))
                        ax2.fill_between(sub_df["elapsed_seconds"], sub_df["exceedance_prob"], color="#e74c3c", alpha=0.5, label="Exceedance Probability")
                        ax2.set_ylim(-0.05, 1.05)
                        ax2.set_title("Exceedance Risk Timeline", fontsize=11)
                        ax2.set_xlabel("Elapsed Time (seconds)")
                        ax2.set_ylabel("Probability")
                        ax2.legend(loc="upper right")
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close()

                        st.dataframe(sub_df[["elapsed_seconds", "E1TQ_S76C", "E1TQ_pred_next", "exceedance_prob", "exceedance_pred"]].head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FLIGHT DURATION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Flight Duration" in page:
    st.title("🛫 Flight Duration Analysis")
    fr = load_flight_report()
    if fr is None:
        st.warning("Flight duration report not found. Run `python src/run_pipeline.py` first.")
        st.stop()

    st.subheader("Three-Hour Constraint Verification")
    counts = fr["constraint_status"].value_counts()
    col1, col2, col3 = st.columns(3)
    for col, (status, label, color) in zip([col1, col2, col3], [
        ("within_3h", "✅ Within 3 h", "#2ecc71"),
        ("exceeds_3h", "⚠️ Exceeds 3 h", "#e74c3c"),
        ("unknown", "❓ Unknown", "#7f8c8d"),
    ]):
        val = counts.get(status, 0)
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{color}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Flight Segment Table")
    st.dataframe(fr, use_container_width=True)

    st.subheader("Duration Distribution")
    fig_p = FIG / "flight_duration_hist.png"
    if fig_p.exists():
        st.image(str(fig_p), use_container_width=True)

    with st.expander("ℹ️ Notes on flight segmentation"):
        st.markdown("""
        - Each source file is treated as one recording session.
        - Flights within a session are separated where elapsed time wraps back to 00:00:00
          or where a gap >60 s coincides with near-zero engine speed.
        - Segments shorter than ~5 minutes may be taxiing or engine-start/shutdown periods.
        - Segments shown as "exceeds_3h" should be investigated individually — they may
          represent multiple consecutive flights recorded without session breaks.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PARAMETER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Parameter Analysis" in page:
    st.title("📊 Engine Parameter Analysis")
    df = load_features()
    if df is None:
        df = load_data()
    if df is None:
        st.warning("Dataset not found.")
        st.stop()

    params = [c for c in PARAM_LABELS if c in df.columns]
    param = st.selectbox("Select Parameter", params,
                         format_func=lambda x: PARAM_LABELS.get(x, x))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Distribution: {PARAM_LABELS.get(param, param)}")
        fig_p = FIG / "parameter_distributions.png"
        if fig_p.exists():
            st.image(str(fig_p), use_container_width=True)

    with col2:
        st.subheader("Statistics (fault-cleaned)")
        from data_loader import FAULT_THRESHOLDS
        data = df[param].copy()
        if param in FAULT_THRESHOLDS:
            lo, hi = FAULT_THRESHOLDS[param]
            data = data[(data >= lo) & (data <= hi)]
        stats = data.describe()
        st.dataframe(stats.rename("Value").to_frame(), use_container_width=True)

        if param in NORMAL_BANDS:
            lo_n, hi_n = NORMAL_BANDS[param]
            pct_exceed = ((data < lo_n) | (data > hi_n)).mean() * 100
            st.metric("Exceedance %", f"{pct_exceed:.2f}%")

    st.subheader("Time Series (all sessions)")
    fig_p = FIG / "time_series_per_file.png"
    if fig_p.exists():
        st.image(str(fig_p), use_container_width=True)

    st.subheader("Correlation Matrix")
    fig_p = FIG / "correlation_matrix.png"
    if fig_p.exists():
        st.image(str(fig_p), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXCEEDANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Exceedance" in page:
    st.title("⚠️ Exceedance Analysis")
    st.markdown("""
    An **exceedance** is defined here as a one-second observation where a parameter
    falls outside the normal operating band for the Sikorsky S-76A/C+.
    
    > ⚠️ An exceedance is **not** an engine failure. It may reflect a legitimate
    > manoeuvre, a recording artefact, or an actual limit exceedance requiring
    > maintenance action — further investigation is always required.
    """)

    fig1 = FIG / "exceedance_counts.png"
    fig2 = FIG / "exceedance_by_file.png"
    if fig1.exists():
        st.subheader("Exceedance Counts by Parameter")
        st.image(str(fig1), use_container_width=True)
    if fig2.exists():
        st.subheader("Exceedance Seconds by Recording Session")
        st.image(str(fig2), use_container_width=True)

    feat_df = load_features()
    if feat_df is not None and "any_exceedance" in feat_df.columns:
        exceed_cols = [c for c in feat_df.columns if c.endswith("_exceed")]
        total_seconds = len(feat_df)
        data = {c.replace("_exceed", ""): int(feat_df[c].fillna(0).sum()) for c in exceed_cols}
        exc_df = pd.DataFrame(list(data.items()), columns=["Parameter", "Exceedance Seconds"])
        exc_df["% of Total"] = (exc_df["Exceedance Seconds"] / total_seconds * 100).round(3)
        exc_df = exc_df.sort_values("Exceedance Seconds", ascending=False)
        st.subheader("Exceedance Summary Table")
        st.dataframe(exc_df, use_container_width=True)
        csv = exc_df.to_csv(index=False)
        st.download_button("⬇️ Download Exceedance Summary", csv,
                           "exceedance_summary.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEALTH MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Health Model" in page:
    st.title("🤖 Engine Health / Exceedance Detection Model")
    st.markdown("""
    **Task:** Binary classification — detect whether a given second of operation
    constitutes a parameter exceedance.
    
    **Target:** `any_exceedance` (1 if ≥ 1 parameter outside its normal band)
    
    **Validation:** Chronological split (70% train / 30% test per session)
    """)

    metrics = load_metrics("health_model_metrics.csv")
    if metrics is not None:
        st.subheader("Model Comparison")
        st.dataframe(metrics.style.highlight_max(subset=["f1", "roc_auc"], color="#1e4d2b")
                                  .highlight_min(subset=["f1", "roc_auc"], color="#4d1e1e"),
                     use_container_width=True)
        csv = metrics.to_csv(index=False)
        st.download_button("⬇️ Download Metrics", csv, "health_metrics.csv", "text/csv")
    else:
        st.info("Run the pipeline first to see model metrics.")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    c1, c2, c3 = st.columns(3)
    for col, model in zip([c1, c2, c3],
                          ["logistic_regression", "random_forest", "xgboost"]):
        p = FIG / f"cm_{model}.png"
        if p.exists():
            col.image(str(p), caption=model.replace("_", " ").title())

    # Feature importance
    st.subheader("Feature Importance (best model)")
    for model in ["random_forest", "xgboost"]:
        p = FIG / f"feat_imp_{model}.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
            break

    feat_meta = MOD / "health_model_features.json"
    if feat_meta.exists():
        with open(feat_meta) as f:
            meta = json.load(f)
        st.info(f"Best model: **{meta.get('best_model', '—')}**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TORQUE FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif "Forecasting" in page:
    st.title("📈 Next-Step Torque Forecasting")
    st.markdown("""
    **Task:** Regression — predict Engine 1 Torque (E1TQ_S76C) at second *t+1*
    given all available measurements at second *t*.
    
    **Unit of error:** All MAE / RMSE values are in **percentage torque (%)**.
    An MAE of 2.0 means the average absolute error is 2 percentage points of torque.
    
    **Validation:** Chronological split (75% train / 25% test per session, no overlap)
    """)

    metrics = load_metrics("forecasting_model_metrics.csv")
    if metrics is not None:
        st.subheader("Model Comparison")
        st.dataframe(metrics.style.highlight_min(subset=["MAE_%", "RMSE_%"], color="#1e4d2b")
                                  .highlight_max(subset=["R2"], color="#1e4d2b"),
                     use_container_width=True)

    st.subheader("Actual vs Predicted")
    for model in ["xgboost", "random_forest", "linear_regression"]:
        p = FIG / f"forecast_actual_vs_pred_{model}.png"
        if p.exists():
            st.image(str(p), use_container_width=True)

    st.subheader("Residual Distributions")
    c1, c2, c3 = st.columns(3)
    for col, model in zip([c1, c2, c3],
                          ["linear_regression", "random_forest", "xgboost"]):
        p = FIG / f"forecast_residuals_{model}.png"
        if p.exists():
            col.image(str(p))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RUL ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif "RUL" in page:
    st.title("🔬 Remaining Useful Life (RUL) Assessment")
    st.markdown("""<div class='rul-unsupported'>
    ❌ <b>Supervised RUL Prediction is NOT supported by the available dataset.</b><br><br>
    The current data does not contain engine failure events, cumulative operating hours,
    Time Between Overhaul (TBO) records, or maintenance history needed to construct
    a valid RUL target label.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    report = load_metrics("rul_assessment.txt")
    if report:
        st.text(report)

    st.subheader("What is needed to enable RUL prediction?")
    st.markdown("""
    | Required Data | Status |
    |---|---|
    | Engine logbook (installation date, current CSOH) | ❌ Not available |
    | Time Between Overhaul (TBO) endpoint | ❌ Not available |
    | Engine failure or scheduled removal event | ❌ Not available |
    | Cumulative operating hours counter | ❌ Not available |
    | Maintenance / component replacement records | ❌ Not available |

    **For the Allison 250-C30S**, the standard TBO is ~3 500 hours.
    To estimate RUL, the engine's current on-wing hours would need to be
    provided from the aircraft logbook.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Reports" in page:
    st.title("📋 Reports & Downloads")

    for label, path in [
        ("Dataset Inventory", REP / "dataset_inventory.csv"),
        ("Column Mapping", REP / "column_mapping.csv"),
        ("Flight Duration Report", REP / "flight_duration_report.csv"),
        ("Health Model Metrics", MET / "health_model_metrics.csv"),
        ("Forecasting Metrics", MET / "forecasting_model_metrics.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            st.subheader(label)
            st.dataframe(df, use_container_width=True)
            st.download_button(f"⬇️ {label}", df.to_csv(index=False),
                               path.name, "text/csv")
            st.markdown("---")

    st.subheader("Data Quality Report")
    dqr_path = REP / "data_quality_report.txt"
    if dqr_path.exists():
        st.text(open(dqr_path).read())
