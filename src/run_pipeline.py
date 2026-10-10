"""
src/run_pipeline.py
Master pipeline — runs all phases in order.
Usage: python src/run_pipeline.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import RAW_DIR

def symlink_raw_data():
    """Symlink source Excel files into data/raw/ if not already there."""
    src = Path(__file__).parent.parent.parent   # /Users/.../model/
    for f in src.glob("*.xlsx"):
        dest = RAW_DIR / f.name
        if not dest.exists():
            dest.symlink_to(f)
    count = len(list(RAW_DIR.glob("*.xlsx")))
    print(f"  {count} Excel file(s) available in data/raw/")


def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("  HELICOPTER ENGINE HEALTH PIPELINE — S-76A/C+")
    print("="*60)

    # ── Phase 2: Consolidate ───────────────────────────────────────────────
    print("\n[Phase 2] Dataset Consolidation & Reports")
    symlink_raw_data()
    from dataset_consolidation import run as consolidate
    df = consolidate()

    # ── Phase 3: Flight segmentation ──────────────────────────────────────
    print("\n[Phase 3] Flight Segmentation & 3-Hour Verification")
    from flight_segmentation import run as segment
    flight_report = segment(df)

    # ── Phase 6: Feature engineering ──────────────────────────────────────
    print("\n[Phase 6] Feature Engineering")
    from feature_engineering import run as engineer
    feat_df = engineer(df)

    # ── Phase 5: EDA ──────────────────────────────────────────────────────
    print("\n[Phase 5] Exploratory Data Analysis")
    from eda import run as eda
    eda(feat_df, flight_report)

    # ── Phase 7/8: Health model ───────────────────────────────────────────
    print("\n[Phase 7/8] Engine Health / Exceedance Detection Model")
    from train_health_model import run as train_health
    train_health(feat_df)

    # ── Phase 7/8: Forecasting model ─────────────────────────────────────
    print("\n[Phase 7/8] Next-Step Torque Forecasting Model")
    from train_forecasting_model import run as train_forecast
    train_forecast(feat_df)

    # ── Phase 9: RUL assessment ───────────────────────────────────────────
    print("\n[Phase 9] RUL Prediction Assessment")
    from train_rul_model import run as rul_assess
    rul_assess()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes.")
    print(f"  Results in: results/  |  Models in: models/")
    print(f"  Dashboard: streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
