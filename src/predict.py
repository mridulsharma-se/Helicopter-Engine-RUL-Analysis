"""
src/predict.py
Inference interface — load saved models and score new data.
"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_health_model():
    mp = MODEL_DIR / "health_model.pkl"
    fp = MODEL_DIR / "health_model_features.json"
    if not mp.exists():
        raise FileNotFoundError("health_model.pkl not found. Run train_health_model.py first.")
    model = joblib.load(mp)
    with open(fp) as f:
        meta = json.load(f)
    return model, meta


def load_forecasting_model():
    mp = MODEL_DIR / "forecasting_model.pkl"
    fp = MODEL_DIR / "forecasting_model_features.json"
    if not mp.exists():
        raise FileNotFoundError("forecasting_model.pkl not found. Run train_forecasting_model.py first.")
    model = joblib.load(mp)
    with open(fp) as f:
        meta = json.load(f)
    return model, meta


def predict_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame with the exceedance detection model.
    Returns df with 'exceedance_prob' and 'exceedance_pred' columns added.
    """
    model, meta = load_health_model()
    feats = [c for c in meta["features"] if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    df = df.copy()
    df["exceedance_prob"] = model.predict_proba(X)[:, 1]
    df["exceedance_pred"] = model.predict(X)
    return df


def predict_next_torque(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict E1TQ_S76C at t+1 for each row.
    """
    model, meta = load_forecasting_model()
    feats = [c for c in meta["features"] if c in df.columns]
    X = df[feats].fillna(df[feats].median())
    df = df.copy()
    df["E1TQ_S76C_predicted_next"] = model.predict(X)
    return df


if __name__ == "__main__":
    import pandas as pd
    from feature_engineering import build_features
    from data_loader import load_file, RAW_DIR
    import sys

    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        print("No files in data/raw/. Symlink or copy the source Excel files there first.")
        sys.exit(1)

    print(f"Running inference on: {files[0].name}")
    df_raw = load_file(files[0])
    df_feat = build_features(df_raw)
    df_health = predict_health(df_feat)
    df_final  = predict_next_torque(df_health)
    print(df_final[["elapsed_seconds", "E1TQ_S76C",
                     "exceedance_prob", "exceedance_pred",
                     "E1TQ_S76C_predicted_next"]].head(10).to_string())
