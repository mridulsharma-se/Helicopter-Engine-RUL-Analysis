# Complete Research & Technical Report
## Prediction of Remaining Useful Life (RUL) of a Conventional Helicopter Engine Using Machine Learning and Historical Flight Data
**Platform**: Sikorsky S-76A/C+ Commercial Twin-Turboshaft Helicopter  
**Powerplants**: Twin Allison 250-C30S (Rolls-Royce) Turboshaft Engines  
**Author**: Senior ML Engineer & Aviation Data Analyst  

---

## 1. Executive Summary

This project delivers a complete, modular, reproducible, and mathematically rigorous machine learning framework for analyzing continuous flight and engine data, auditing engine operational health, and evaluating the feasibility of Remaining Useful Life (RUL) estimation.

### Key Operational & Scientific Conclusions
1. **Nature of the Data**:
   - The dataset consists of nine Excel workbooks titled `KNH Exceedance <DDMMYY>.xlsx`, aggregating **3,358,499 rows** (3.36 million seconds / ~933 hours) of 1-Hz continuous Flight Data Monitoring (FDM / FDR) data collected between January 2025 and June 2026.
   - Despite the keyword "Exceedance" in the filenames, the workbooks are **not** event-triggered exceedance logs; they are full continuous telemetry traces spanning prolonged multi-day download sessions (65 to 114 hours per file).
2. **Definitive RUL Feasibility Assessment**:
   - **Supervised RUL prediction is NOT viable with this dataset.**
   - There are **zero engine failure, unscheduled removal, or distress events** in the dataset.
   - The timestamp column (`HH:MM:SS`) is an elapsed counter that resets per session; no cumulative engine life counter (CSOH/CSN), airframe cycle log, or engine logbook history exists.
   - No Time Between Overhaul (TBO, nominal 3,500 hours for Allison 250-C30S) baseline position is documented.
   - Generating synthetic or artificial RUL degradation labels would violate aviation maintenance standards and safety integrity.
3. **Machine Learning Implementations Delivered**:
   - **Engine Health & Exceedance Classifier**: Accurately classifies whether the engine is operating outside its manufacturer-defined operational envelope.
     - *Baseline*: Accuracy 81.54%, F1 0.00%
     - *Logistic Regression*: Accuracy 99.95%, F1 0.9986, ROC-AUC 1.0000
     - *Random Forest*: Accuracy 100.0%, F1 1.0000, ROC-AUC 1.0000
     - *XGBoost*: Accuracy 100.0%, F1 1.0000, ROC-AUC 1.0000
   - **One-Step-Ahead Engine Parameter Forecasting**: Regresses next-second Engine 1 Torque (`E1TQ_S76C`) using 12 dynamic features and rolling statistics across 3.35 million rows:
     - *Persistence Baseline ($y_{t+1} = y_t$)*: MAE = 1.402%, RMSE = 2.143%, $R^2 = 0.9935$
     - *Linear Regression*: MAE = 1.413%, RMSE = 2.105%, $R^2 = 0.9937$
     - *Random Forest Regressor*: MAE = 1.255%, RMSE = 1.782%, $R^2 = 0.9955$
     - *XGBoost Regressor (Best Model)*: **MAE = 1.247%, RMSE = 1.765%, $R^2 = 0.9956$**

---

## 2. Ingestion & Quality Audit Findings

### Physical Limits vs. FDR Fault Sentinels
During telemetry ingestion, extreme values outside plausible physical boundaries were identified as FDR invalid-word sentinels:
- `PRALT` (Pressure Altitude): -16,300 ft is a known digital FDR invalidity flag. Valid physical envelope: -300 ft to 10,000 ft. Flagged sentinels: 12,142 records (0.36%).
- `OAT` (Outside Air Temp): Sentinels ±530°C flagged in 12,230 records (0.36%). Valid range: -40°C to +55°C.
- `E1NF_S76C` / `E2NF_S76C` (N1 / Nf Gas Generator): Sentinels >120% flagged in 543 and 897 records.
- `E1TQ_S76C` / `E2TQ_S76C` (Torque): Sentinels >130% flagged in 387 and 1,937 records.
- `Indicated_Airspeed`: Sentinels >200 knots flagged in 11,586 records (0.34%).
- `PITCH`: Values outside ±90° flagged in 76,722 records (2.28%).

### Schema Reconciliation & Deduplication
- Across the 9 files, schema variations were harmonized into a standardized canonical schema.
- In `KNH Exceedance 120125.xlsx`, both `E1NF S76C+` and `E1NF S76C` existed concurrently with identical values; the duplicate channel was mapped to `E1NF_S76C_plus` to avoid column collisions while safeguarding data completeness.
- Clean consolidated telemetry was serialized to `data/processed/consolidated_dataset.parquet` (3,358,499 rows, 40 columns).

---

## 3. Flight Segmentation & Duration Analysis

Helicopter operational guidelines typically benchmark single sorties within a 2.5 to 3.0-hour fuel endurance window (10,800 seconds).
The segmentation algorithm analyzed continuity across all recordings:
- Total sessions: 9
- Mean session duration: ~104.3 hours (375,388 seconds)
- Range: 65.4 hours (`120125.xlsx`) to 114.3 hours (`120526.xlsx`)
- Finding: None of the files represent single flights. Rather, each file represents an extended multi-day continuous buffer recording downloaded from the Flight Data Recorder.

---

## 4. Feature Engineering Architecture

A feature space of **90 variables** was engineered in `data/processed/features.parquet`:
1. **Fault-Masked Parameter Values**: Raw sensors with FDR sentinel codes converted to nulls prior to numerical rolling calculations.
2. **Binary Operational Flags**: Discrete warning indicators (`E1_OIL_PRS`, `E2_OIL_PRS`, `XMSN_OIL_PRS`, `XMSN_OIL_TEMP`, `BATT_OVERTEMP`, `ROTBRK`, `HYPRS1`, `HYPRS2`) converted to binary 0/1 integers.
3. **Operational Limit Exceedances**: Flags indicating deviations outside normal turbine cruise bands:
   - $E1NF \notin [95\%, 110\%]$
   - $E2NF \notin [60\%, 80\%]$
   - $E1TQ, E2TQ > 110\%$
   - $RTRSPD \notin [95\%, 110\%]$
   - Overall exceedance prevalence across 3.35M rows: **18.5% – 21.5%**.
4. **Dynamic Rolling Windows**: 30-second and 300-second rolling means and standard deviations computed per recording session.
5. **Session Elapsed Cumulative Seconds**: Continuous counter within each file.

---

## 5. Model Training & Evaluation

### Health / Exceedance Classification
- **Dataset Split**: Chronological 70% train (2,350,946 rows) / 30% test (1,007,553 rows). Zero look-ahead bias.
- **Training Subset**: 100,000 stratified samples used to fit models rapidly while preserving exact class distribution.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Majority Baseline** | 81.54% | 0.00% | 0.00% | 0.00% | 0.5000 |
| **Logistic Regression** | 99.95% | 99.86% | 99.85% | 99.86% | 1.0000 |
| **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |
| **XGBoost** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** |

*Key Feature Importances*: The most influential predictors for exceedance classification were `E2NF_S76C` (Engine 2 N1 speed), `E1NF_S76C` (Engine 1 N1 speed), `E1NF_S76C_roll30s_mean`, and `E1TQ_S76C_roll30s_mean`.

### One-Step-Ahead Engine Torque Forecasting
- **Target**: $E1TQ_{t+1}$
- **Dataset Split**: Chronological train (2,514,669 rows) / test (838,375 rows).
- **Benchmark**: Evaluated against the strict physical persistence baseline $y_{t+1} = y_t$.

| Model | MAE (%) | RMSE (%) | $R^2$ Score | Performance vs. Baseline |
|---|---|---|---|---|
| **Persistence ($y_t$)** | 1.402% | 2.143% | 0.9935 | Reference |
| **Linear Regression** | 1.413% | 2.105% | 0.9937 | -0.7% MAE / +1.8% RMSE |
| **Random Forest** | 1.255% | 1.782% | 0.9955 | **+10.5% MAE Improvement** |
| **XGBoost (Best)** | **1.247%** | **1.765%** | **0.9956** | **+11.1% MAE / +17.6% RMSE Improvement** |

---

## 6. System Deliverables & Verification

- **Master Pipeline**: `src/run_pipeline.py` executes end-to-end in ~4 minutes.
- **Unit Test Suite**: `tests/test_pipeline.py` passes all 10 tests verifying schema transformation, sentinel detection, flight segmentation, lag construction, and RUL assertions.
- **Inference Module**: `src/predict.py` loads serialized model artifacts from `models/` and performs real-time scoring.
- **Streamlit Dashboard**: `app.py` delivers an interactive UI across 8 specialized views:
  1. *Overview*: Fleet statistics, hours recorded, recording inventory.
  2. *Flight Duration*: Sortie durations and 3-hour limit analysis.
  3. *Parameter Analysis*: Dynamic time-series plots and physical distributions.
  4. *Exceedance Analysis*: Event rates, limits comparison, cross-file trends.
  5. *Health Model*: Confusion matrices, ROC curves, feature importances.
  6. *Torque Forecasting*: Actual vs. predicted profiles, residual analysis.
  7. *RUL Assessment*: Formal engineering manifesto on lifecycle data gaps.
  8. *Reports*: In-app inspection of inventory, schema mapping, and quality audits.
