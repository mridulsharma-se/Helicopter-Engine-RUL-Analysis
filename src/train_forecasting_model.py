"""
src/train_forecasting_model.py
Phase 7/8 — Next-Step Engine Parameter Forecasting

Task: Given sensor readings at second t, predict E1TQ_S76C at second t+1
      (one-step-ahead regression).  This is the primary time-series task
      the dataset can support.

Target: E1TQ_S76C at t+1  (Engine 1 Torque — the most operationally
        relevant continuous parameter for health monitoring)

Models compared:
  • Persistence baseline (predict t = t-1)
  • Linear Regression with lagged features
  • Random Forest with lagged features
  • XGBoost with lagged features

Validation: strict chronological split (no overlap between train/test windows).
"""
import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

PROC_DIR   = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR  = Path(__file__).resolve().parent.parent / "models"
FIG_DIR    = Path(__file__).resolve().parent.parent / "results" / "figures"
METRIC_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"
for d in (MODEL_DIR, FIG_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED         = 42
TARGET_COL   = "E1TQ_S76C"
FORECAST_COL = "E1TQ_S76C_next"   # t+1 target

LAG_FEATURES = [
    "E1TQ_S76C", "E2TQ_S76C", "E1NF_S76C", "E2NF_S76C",
    "RTRSPD", "OAT", "Indicated_Airspeed", "PRALT",
    "E1TQ_S76C_roll30s_mean", "E1TQ_S76C_roll30s_std",
    "E1NF_S76C_roll30s_mean",
    "cum_op_seconds",
]


def build_lagged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create t+1 target within each source file.
    Rows are sorted by elapsed_seconds within each file.
    The last row of each file (no future value) is dropped.
    """
    frames = []
    for _, grp in df.groupby("source_file"):
        grp = grp.sort_values("elapsed_seconds").reset_index(drop=True)
        grp[FORECAST_COL] = grp[TARGET_COL].shift(-1)
        # drop last row (NaN target)
        grp = grp.iloc[:-1]
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def chronological_split(df: pd.DataFrame, test_frac: float = 0.25):
    train_frames, test_frames = [], []
    for _, grp in df.groupby("source_file"):
        grp = grp.sort_values("elapsed_seconds").reset_index(drop=True)
        cut = int(len(grp) * (1 - test_frac))
        train_frames.append(grp.iloc[:cut])
        test_frames.append(grp.iloc[cut:])
    return pd.concat(train_frames), pd.concat(test_frames)


def prepare_Xy(df: pd.DataFrame):
    cols = [c for c in LAG_FEATURES if c in df.columns]
    mask = df[FORECAST_COL].notna() & df[TARGET_COL].notna()
    df_valid = df[mask].copy()
    X = df_valid[cols].fillna(df_valid[cols].median())
    y = df_valid[FORECAST_COL]
    return X, y, cols


def evaluate_reg(name, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name}: MAE={mae:.3f}%  RMSE={rmse:.3f}%  R²={r2:.4f}")
    return {"model": name, "MAE_%": round(mae,3), "RMSE_%": round(rmse,3), "R2": round(r2,4)}


def plot_actual_vs_predicted(name, y_true, y_pred, n=3000):
    idx = np.arange(min(n, len(y_true)))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(idx, np.array(y_true)[:n], label="Actual", lw=0.8, color="steelblue")
    ax.plot(idx, np.array(y_pred)[:n], label="Predicted", lw=0.8, color="tomato", alpha=0.8)
    ax.set_title(f"E1 Torque — Actual vs Predicted ({name})")
    ax.set_xlabel("Time step (s)")
    ax.set_ylabel("E1TQ (%)")
    ax.legend()
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(FIG_DIR / f"forecast_actual_vs_pred_{safe}.png", dpi=120)
    plt.close()
    print(f"  [fig] forecast_actual_vs_pred_{safe}.png")


def plot_residuals(name, y_true, y_pred):
    res = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res, bins=80, color="mediumseagreen", edgecolor="none")
    ax.axvline(0, color="red", lw=1)
    ax.set_title(f"Residuals Distribution — {name}")
    ax.set_xlabel("Residual (Actual − Predicted, %)")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(FIG_DIR / f"forecast_residuals_{safe}.png", dpi=120)
    plt.close()
    print(f"  [fig] forecast_residuals_{safe}.png")


def run(feat_df: pd.DataFrame | None = None) -> dict:
    if feat_df is None:
        feat_df = pd.read_parquet(PROC_DIR / "features.parquet")

    if TARGET_COL not in feat_df.columns:
        print(f"  [SKIP] {TARGET_COL} not found.")
        return {}

    print("Building lagged dataset ...")
    lag_df = build_lagged_dataset(feat_df)
    print(f"  Lagged dataset: {len(lag_df):,} rows")

    train_df, test_df = chronological_split(lag_df)
    X_train, y_train, feat_names = prepare_Xy(train_df)
    X_test,  y_test,  _          = prepare_Xy(test_df)
    print(f"  Train: {len(X_train):,}  | Test: {len(X_test):,}")

    MAX_TRAIN_SAMPLES = 100_000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(f"  Subsampling training set to {MAX_TRAIN_SAMPLES:,} samples for efficient training...")
        step = max(1, len(X_train) // MAX_TRAIN_SAMPLES)
        train_idx = X_train.index[::step][:MAX_TRAIN_SAMPLES]
        X_train_fit = X_train.loc[train_idx]
        y_train_fit = y_train.loc[train_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    all_metrics = []

    # ── Persistence baseline ───────────────────────────────────────────────────
    if TARGET_COL in X_test.columns:
        y_persistence = X_test[TARGET_COL].values
        m = evaluate_reg("Persistence", y_test.values, y_persistence)
        all_metrics.append(m)

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            tree_method="hist",
            random_state=SEED, verbosity=0,
        ),
    }

    best_mae, best_name, best_model = np.inf, None, None

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train_fit, y_train_fit)
        y_pred = model.predict(X_test)
        m = evaluate_reg(name, y_test.values, y_pred)
        all_metrics.append(m)
        plot_actual_vs_predicted(name, y_test.values, y_pred)
        plot_residuals(name, y_test.values, y_pred)
        if m["MAE_%"] < best_mae:
            best_mae, best_name, best_model = m["MAE_%"], name, model

    print(f"\n  Best model: {best_name}  (MAE={best_mae:.3f}%)")
    joblib.dump(best_model, MODEL_DIR / "forecasting_model.pkl")
    with open(MODEL_DIR / "forecasting_model_features.json", "w") as f:
        json.dump({"features": feat_names, "target": FORECAST_COL, "best_model": best_name}, f, indent=2)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRIC_DIR / "forecasting_model_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    return {"metrics": all_metrics, "best_model": best_name}


if __name__ == "__main__":
    run()
