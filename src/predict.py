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
    import argparse
    import pandas as pd
    from feature_engineering import build_features
    from data_loader import load_file, RAW_DIR
    import sys

    parser = argparse.ArgumentParser(description="Run inference using trained health and forecasting models.")
    parser.add_argument("--file", type=str, default=None, help="Path to raw Excel or CSV file to score.")
    args = parser.parse_args()

    sample_csv = RAW_DIR.parent / "sample" / "sample_telemetry.csv"

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            print(f"Error: file not found: {fp}")
            sys.exit(1)
        print(f"Running inference on: {fp.name}")
        df_raw = load_file(fp) if fp.suffix in [".xlsx", ".xls"] else pd.read_csv(fp)
    elif sample_csv.exists():
        print(f"Running inference on sample telemetry: {sample_csv.name}")
        df_raw = pd.read_csv(sample_csv)
    else:
        files = sorted(RAW_DIR.glob("*.xlsx"))
        if not files:
            parent_candidate = RAW_DIR.resolve().parent.parent.parent
            files = sorted(parent_candidate.glob("*.xlsx"))
        if files:
            print(f"Running inference on: {files[0].name}")
            df_raw = load_file(files[0])
        else:
            print("No input telemetry found.")
            sys.exit(1)

    df_feat = build_features(df_raw)
    df_health = predict_health(df_feat)
    df_final  = predict_next_torque(df_health)
    cols_to_show = [c for c in ["elapsed_seconds", "E1TQ_S76C",
                                "exceedance_prob", "exceedance_pred",
                                "E1TQ_S76C_predicted_next"] if c in df_final.columns]
    print("\n--- Model Inference Output (First 10 records) ---")
    print(df_final[cols_to_show].head(10).to_string(index=False))
