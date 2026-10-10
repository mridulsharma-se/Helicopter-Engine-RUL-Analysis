# Helicopter Engine Health Monitoring & Predictive Maintenance System
## Sikorsky S-76A/C+ — Allison 250-C30S Turboshaft Engines

---

## 1. Executive Summary & Core Findings

This project implements a complete, modular, and scientifically rigorous machine learning pipeline for analyzing flight and engine data from a **Sikorsky S-76A/C+** helicopter powered by twin **Allison 250-C30S** turboshaft engines.

### Ground Truth Regarding Dataset & Remaining Useful Life (RUL)
The input dataset comprises nine Excel workbooks named `KNH Exceedance <DDMMYY>.xlsx`, representing over **3.3 million seconds (~930 hours)** of continuous 1-Hz Flight Data Monitoring (FDM / FDR) recordings collected between November 2025 and June 2026.

Key engineering and data science findings:
1. **The files are continuous flight recordings, not isolated exceedance event logs.** Despite the word "Exceedance" in the filenames, each file contains continuous, second-by-second telemetry spanning complete operational sorties.
2. **Supervised Remaining Useful Life (RUL) estimation is NOT supported by this dataset.**
   - There are **no engine failure, removal, or component distress events** documented.
   - The dataset lacks **cumulative operating hours (CSOH/CSN)**; the timestamp column (`HH:MM:SS`) is an elapsed time counter that resets to 00:00:00 at the start of each recording session.
   - There are no **Time Between Overhaul (TBO)** baseline markers, airframe cycle counts, or maintenance logs.
   - Manufacturing artificial or synthetic RUL targets would be invalid and dangerous for aviation safety.
3. **Supported Machine Learning Tasks Implemented**:
   - **Engine Health & Exceedance Classification**: Identifies abnormal operation and exceedances across critical turbine operating envelopes (N1/Nf gas generator speed, torque, rotor speed, oil pressure, temperatures).
   - **One-Step-Ahead Engine Parameter Forecasting**: Time-series regression predicting next-second Engine 1 Torque (`E1TQ_S76C`) using lagged dynamic features, persistence baselines, linear regression, Random Forest, and XGBoost.
   - **Flight Segmentation & 3-Hour Endurance Verification**: Algorithmic segmentation of recording sessions into discrete sorties, evaluating flight durations against operational and regulatory constraints.

---

## 2. Aircraft & Engine Architecture

- **Aircraft Model**: Sikorsky S-76A/C+ twin-engine medium commercial helicopter.
- **Powerplant**: Two Allison (Rolls-Royce) 250-C30S turboshaft engines (`E1` and `E2`).
- **Telemetry Recording**: 1-Hz continuous sample rate.
- **Normal Operating Bands & Plausible Physical Ranges**:
  - `E1NF_S76C` / `E2NF_S76C` (N1 / Nf Gas Generator Speed): 95% – 110% (values >120% are FDR fault sentinels).
  - `E1TQ_S76C` / `E2TQ_S76C` (Engine Torque): 0% – 110% (values >130% are FDR fault sentinels).
  - `RTRSPD` (Main Rotor Speed): 95% – 110% (nominal 100%).
  - `Indicated_Airspeed`: 0 – 155 knots (Vne ~155 kn).
  - `PRALT` (Pressure Altitude): -300 ft to 10,000 ft (sentinel value -16,300 ft indicates sensor invalidity).
  - `OAT` (Outside Air Temp): -40°C to +55°C (sentinel values ±530°C indicate invalid FDR word).

---

## 3. Project Directory Structure

```
helicopter-rul-project/
├── app.py                      # Interactive Streamlit dashboard
├── data/
│   ├── raw/                    # Symlinks/copies of source Excel files
│   └── processed/              # Parquet files (consolidated & feature datasets)
├── models/
│   ├── health_model.pkl        # Best trained exceedance/health model
│   ├── health_model_features.json
│   ├── forecasting_model.pkl   # Best trained parameter forecasting model
│   └── forecasting_model_features.json
├── reports/
│   ├── dataset_inventory.csv   # Comprehensive inventory across all 9 workbooks
│   ├── column_mapping.csv      # Standardized telemetry schema dictionary
│   ├── data_quality_report.txt # Missing values, duplicates, and sentinel audit
│   └── flight_duration_report.csv # Flight segmentation and 3-hour constraint check
├── results/
│   ├── figures/                # EDA & model evaluation plots (PNG)
│   └── metrics/                # Metrics CSVs & RUL assessment manifesto
├── src/
│   ├── data_loader.py          # Schema mapping, fault sentinel masking, ingestion
│   ├── dataset_consolidation.py# Master consolidation & quality audits
│   ├── flight_segmentation.py  # Vectorized sortie segmentation & duration checks
│   ├── feature_engineering.py  # Rolling stats, lag features, flag encoding
│   ├── eda.py                  # Exploratory data analysis & figure generator
│   ├── train_health_model.py   # Baseline, LogReg, RF, XGBoost exceedance models
│   ├── train_forecasting_model.py # Persistence, Linear, RF, XGBoost forecasting
│   ├── train_rul_model.py      # Formal RUL gap analysis and manifesto
│   ├── predict.py              # Inference interface for scoring new telemetry
│   └── run_pipeline.py         # End-to-end master pipeline runner
├── tests/
│   └── test_pipeline.py        # Comprehensive test suite (unittest)
└── requirements.txt            # Project dependencies
```

---

## 4. Pipeline Execution & Usage

### Setup & Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Master Pipeline
Execute all phases from ingestion to evaluation:
```bash
python3 src/run_pipeline.py
```

### Run Test Suite
```bash
python3 -m unittest discover tests/ -v
```

### Launch Interactive Dashboard
```bash
streamlit run app.py
```

---

## 5. Model Evaluation & Benchmarks

1. **Exceedance / Health Classifier**:
   - Compared against a majority-class baseline.
   - Evaluated on chronological test split (30% per file, strictly time-separated).
   - XGBoost and Random Forest achieve superior F1 scores and ROC-AUC on detecting operational deviations without data leakage.

2. **Parameter Forecasting (E1 Torque at t+1)**:
   - Compared against Persistence baseline ($y_{t+1} = y_t$).
   - Evaluates MAE, RMSE, and $R^2$ across Linear Regression, Random Forest, and XGBoost.
