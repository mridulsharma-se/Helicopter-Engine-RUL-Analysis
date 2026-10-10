"""
src/feature_engineering.py
Phase 6 — Feature engineering pipeline.

Features built on the consolidated 1-Hz dataset:
  • Raw engine parameters (fault-cleaned)
  • Binary warning flags encoded as 0/1
  • Rolling statistics (30 s, 300 s windows)
  • Cumulative engine operating time per source file
  • Exceedance detection (parameters outside normal operating limits)
  • Flight-level aggregates joined back to second-level data
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Normal operating bands for the S-76A/C+ (used for exceedance detection)
NORMAL_BANDS = {
    "E1NF_S76C":          (95.0, 110.0),
    "E2NF_S76C":          (60.0, 80.0),
    "E1TQ_S76C":          (0.0,  110.0),
    "E2TQ_S76C":          (0.0,  110.0),
    "RTRSPD":             (95.0, 110.0),
    "OAT":                (-40.0, 55.0),
    "Indicated_Airspeed": (0.0,  155.0),
    "PRALT":              (-300.0, 10000.0),
}

FLAG_COLS = [
    "E1_OIL_PRS", "E2_OIL_PRS", "XMSN_OIL_PRS", "XMSN_OIL_TEMP",
    "ROTBRK", "BATT_OVERTEMP", "HYPRS1", "HYPRS2",
]


def clean_faults(df: pd.DataFrame) -> pd.DataFrame:
    """Replace fault-sentinel values with NaN for numeric processing."""
    from data_loader import FAULT_THRESHOLDS
    df = df.copy()
    for col, (lo, hi) in FAULT_THRESHOLDS.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, col] = np.nan
    return df


def encode_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Encode On/Off flag columns as 1/0 integers."""
    df = df.copy()
    for col in FLAG_COLS:
        if col in df.columns:
            df[col + "_bin"] = (df[col].str.strip().str.lower() == "on").astype("Int8")
    return df


def exceedance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary exceedance columns for each parameter that exceeds its normal band."""
    df = df.copy()
    for col, (lo, hi) in NORMAL_BANDS.items():
        if col in df.columns:
            df[col + "_exceed"] = (
                (df[col].notna()) & ((df[col] < lo) | (df[col] > hi))
            ).astype("Int8")
    df["any_exceedance"] = (
        df[[c for c in df.columns if c.endswith("_exceed")]]
        .fillna(0).max(axis=1)
    ).astype("Int8")
    return df


def rolling_features(df: pd.DataFrame, numeric_cols: list, windows_s=(30, 300)) -> pd.DataFrame:
    """
    Compute rolling mean and std within each source file.
    Uses elapsed_seconds as the ordering key. Computations are per-file
    to avoid mixing data across recordings.
    """
    df = df.copy()
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if not valid_cols:
        return df

    rolling_results = {f"{c}_roll{w}s_{stat}": [] for c in valid_cols for w in windows_s for stat in ("mean", "std")}

    for _, grp in df.groupby("source_file", sort=False):
        grp_sorted = grp.sort_values("elapsed_seconds")
        for win in windows_s:
            rolled = grp_sorted[valid_cols].rolling(win, min_periods=1)
            means = rolled.mean()
            stds = rolled.std()
            for col in valid_cols:
                rolling_results[f"{col}_roll{win}s_mean"].append(means[col])
                rolling_results[f"{col}_roll{win}s_std"].append(stds[col])

    for col_name, chunks in rolling_results.items():
        df[col_name] = pd.concat(chunks).reindex(df.index)

    return df


def cumulative_operating_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative operating time (seconds) per source file."""
    df = df.copy()
    cum = []
    for _, grp in df.groupby("source_file", sort=False):
        grp_sorted = grp.sort_values("elapsed_seconds")
        es = grp_sorted["elapsed_seconds"].ffill()
        cum.append(es - es.min())
    df["cum_op_seconds"] = pd.concat(cum).reindex(df.index)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Cleaning fault sentinels ...")
    df = clean_faults(df)
    print("  Encoding binary flags ...")
    df = encode_flags(df)
    print("  Adding exceedance flags ...")
    df = exceedance_flags(df)

    num_cols = ["E1NF_S76C", "E2NF_S76C", "E1TQ_S76C", "E2TQ_S76C",
                "RTRSPD", "OAT", "Indicated_Airspeed", "PRALT"]
    present_num = [c for c in num_cols if c in df.columns]
    print(f"  Computing rolling features for: {present_num}")
    df = rolling_features(df, present_num, windows_s=(30, 300))
    print("  Adding cumulative operating time ...")
    df = cumulative_operating_time(df)
    return df


def run(df: pd.DataFrame | None = None, force_recompute: bool = False) -> pd.DataFrame:
    out = PROC_DIR / "features.parquet"
    if out.exists() and not force_recompute:
        print(f"  Loading existing features from {out} ...")
        return pd.read_parquet(out)

    if df is None:
        df = pd.read_parquet(PROC_DIR / "consolidated_dataset.parquet")

    print("Building feature set ...")
    feat_df = build_features(df)
    for col in feat_df.select_dtypes(include="object").columns:
        feat_df[col] = feat_df[col].astype(str)
    feat_df.to_parquet(out, index=False)
    print(f"  Saved: {out}  ({len(feat_df):,} rows, {len(feat_df.columns)} columns)")
    return feat_df


if __name__ == "__main__":
    df = pd.read_parquet(PROC_DIR / "consolidated_dataset.parquet")
    run(df)
