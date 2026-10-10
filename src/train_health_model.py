"""
src/train_health_model.py
Phase 7/8 — Engine Health / Exceedance Detection Model

Task: Binary classification — predict whether the CURRENT second is an
      exceedance event (any parameter outside its normal band).

Target: any_exceedance (1 = at least one parameter exceeds its normal limit)

Model selection rationale:
  • Logistic Regression  — interpretable baseline
  • Random Forest        — handles non-linearity, gives feature importance
  • XGBoost              — typically best performance on tabular data

Validation: chronological split per source file (70% train / 30% test),
            no look-ahead from features.

NOTE: This model detects exceedances, NOT engine failures.
      No engine failure labels exist in the dataset.
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

PROC_DIR    = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR   = Path(__file__).resolve().parent.parent / "models"
FIG_DIR     = Path(__file__).resolve().parent.parent / "results" / "figures"
METRIC_DIR  = Path(__file__).resolve().parent.parent / "results" / "metrics"
for d in (MODEL_DIR, FIG_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

FEATURE_COLS = [
    "E1NF_S76C", "E2NF_S76C", "E1TQ_S76C", "E2TQ_S76C",
    "RTRSPD", "OAT", "Indicated_Airspeed", "PRALT",
    "E1NF_S76C_roll30s_mean", "E2NF_S76C_roll30s_mean",
    "E1TQ_S76C_roll30s_mean", "E2TQ_S76C_roll30s_mean",
    "E1NF_S76C_roll300s_mean", "E1TQ_S76C_roll300s_mean",
    "E1NF_S76C_roll30s_std",  "E1TQ_S76C_roll30s_std",
    "cum_op_seconds",
    "E1_OIL_PRS_bin", "E2_OIL_PRS_bin", "BATT_OVERTEMP_bin",
    "XMSN_OIL_PRS_bin", "XMSN_OIL_TEMP_bin",
]
TARGET = "any_exceedance"


def chronological_split(df: pd.DataFrame, test_frac: float = 0.30):
    """
    Split chronologically within each source file, never mixing future into training.
    """
    train_frames, test_frames = [], []
    for _, grp in df.groupby("source_file"):
        grp = grp.sort_values("elapsed_seconds").reset_index(drop=True)
        cut = int(len(grp) * (1 - test_frac))
        train_frames.append(grp.iloc[:cut])
        test_frames.append(grp.iloc[cut:])
    return pd.concat(train_frames), pd.concat(test_frames)


def prepare_Xy(df: pd.DataFrame):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET].fillna(0).astype(int) if TARGET in df.columns else None
    X = X.fillna(X.median())
    return X, y, cols


def evaluate(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n  {name}:")
    for k, v in metrics.items():
        if k != "model":
            print(f"    {k:12s}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Normal","Exceedance"],
                                 zero_division=0))
    return metrics


def plot_confusion_matrix(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Exceedance"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(FIG_DIR / f"cm_{safe}.png", dpi=120)
    plt.close()


def plot_feature_importance(name, model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf", model.named_steps.get("model", None))
        if inner is None or not hasattr(inner, "feature_importances_"):
            return
        imp = pd.Series(inner.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    else:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    imp.plot.bar(ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance — {name}")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    plt.savefig(FIG_DIR / f"feat_imp_{safe}.png", dpi=120)
    plt.close()
    print(f"  [fig] feat_imp_{safe}.png")


def run(feat_df: pd.DataFrame | None = None) -> dict:
    if feat_df is None:
        feat_df = pd.read_parquet(PROC_DIR / "features.parquet")

    if TARGET not in feat_df.columns:
        print("  [SKIP] Target 'any_exceedance' not found in features. Run feature_engineering first.")
        return {}

    print("Splitting data (chronological 70/30) ...")
    train_df, test_df = chronological_split(feat_df)
    X_train, y_train, feature_names = prepare_Xy(train_df)
    X_test,  y_test,  _             = prepare_Xy(test_df)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Exceedance prevalence — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # Subsample training data if very large to prevent memory exhaustion and excessive training times
    MAX_TRAIN_SAMPLES = 100_000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(f"  Subsampling training set to {MAX_TRAIN_SAMPLES:,} stratified samples for efficient training...")
        train_idx = X_train.groupby(y_train, group_keys=False).apply(
            lambda g: g.sample(min(len(g), int(MAX_TRAIN_SAMPLES * len(g) / len(X_train))), random_state=SEED)
        ).index
        X_train_fit = X_train.loc[train_idx]
        y_train_fit = y_train.loc[train_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    # ── Baseline: majority class ───────────────────────────────────────────────
    majority = int(y_train_fit.mode()[0])
    y_baseline = np.full(len(y_test), majority)
    base_metrics = {
        "model":     "Baseline (majority)",
        "accuracy":  round(accuracy_score(y_test, y_baseline), 4),
        "precision": round(precision_score(y_test, y_baseline, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_baseline, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_baseline, zero_division=0), 4),
        "roc_auc":   0.5,
    }
    print(f"\n  Baseline (majority class = {majority}): accuracy={base_metrics['accuracy']}")

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=SEED, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            tree_method="hist",
            scale_pos_weight=(y_train_fit == 0).sum() / max((y_train_fit == 1).sum(), 1),
            random_state=SEED, eval_metric="logloss", verbosity=0,
        ),
    }

    all_metrics = [base_metrics]
    best_f1, best_name, best_model = -1, None, None

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train_fit, y_train_fit)
        m = evaluate(name, model, X_test, y_test)
        all_metrics.append(m)
        plot_confusion_matrix(name, model, X_test, y_test)
        plot_feature_importance(name, model, feature_names)
        if m["f1"] > best_f1:
            best_f1, best_name, best_model = m["f1"], name, model

    # ── Save best model ────────────────────────────────────────────────────────
    print(f"\n  Best model: {best_name}  (F1={best_f1:.4f})")
    joblib.dump(best_model, MODEL_DIR / "health_model.pkl")
    with open(MODEL_DIR / "health_model_features.json", "w") as f:
        json.dump({"features": feature_names, "target": TARGET, "best_model": best_name}, f, indent=2)

    # ── Metrics table ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRIC_DIR / "health_model_metrics.csv", index=False)
    print(f"\n  Metrics saved → results/metrics/health_model_metrics.csv")
    print(metrics_df.to_string(index=False))

    return {"metrics": all_metrics, "best_model": best_name, "feature_names": feature_names}


if __name__ == "__main__":
    run()
