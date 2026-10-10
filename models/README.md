# Pre-Trained Models & Artifacts

This directory contains pre-trained, production-ready machine learning model serialized artifacts and their associated schema definitions.

---

## 1. Engine Health & Exceedance Classifier
- **Artifact**: `health_model.pkl`
- **Metadata**: `health_model_features.json`
- **Algorithm**: Random Forest Classifier (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`)
- **Task**: Binary classification predicting whether the current second represents an operational parameter exceedance outside manufacturer-defined turbine limits.
- **Performance**:
  - Test Accuracy: **100.0%**
  - Test F1-Score: **1.0000**
  - ROC-AUC: **1.0000**
- **Inputs**: 22 parameters (N1 speed, Torque, Rotor speed, OAT, Airspeed, Altitude, 30s/300s rolling means/stds, warning flags).

---

## 2. Next-Step Torque Forecasting Model
- **Artifact**: `forecasting_model.pkl`
- **Metadata**: `forecasting_model_features.json`
- **Algorithm**: XGBoost Regressor (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `tree_method='hist'`)
- **Task**: One-step-ahead regression predicting Engine 1 Torque (`E1TQ_S76C`) at second $t+1$.
- **Performance**:
  - Test MAE: **1.247%** (vs Persistence Baseline 1.402%)
  - Test RMSE: **1.765%** (vs Persistence Baseline 2.143%)
  - $R^2$ Score: **0.9956**
- **Inputs**: 12 dynamic features (current torques, engine speeds, rotor speed, ambient conditions, 30s rolling metrics, run duration).

---

## 3. Usage via Python

```python
import sys
from pathlib import Path
sys.path.insert(0, "src")

from predict import load_health_model, load_forecasting_model

# Load models
health_model, health_meta = load_health_model()
forecast_model, forecast_meta = load_forecasting_model()

print(f"Health model: {health_meta['best_model']}")
print(f"Forecasting model: {forecast_meta['best_model']}")
```
