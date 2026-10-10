"""
src/eda.py
Phase 5 — Exploratory Data Analysis
Produces figures saved to results/figures/
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_PARAMS = ["E1NF_S76C", "E2NF_S76C", "E1TQ_S76C", "E2TQ_S76C",
                  "RTRSPD", "OAT", "Indicated_Airspeed", "PRALT"]

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

sns.set_theme(style="darkgrid", palette="muted")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fault-sentinel values before plotting."""
    from data_loader import FAULT_THRESHOLDS
    df = df.copy()
    for col, (lo, hi) in FAULT_THRESHOLDS.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def plot_missing_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    cols = [c for c in NUMERIC_PARAMS if c in df.columns]
    miss_pct = df[cols].isna().mean() * 100
    miss_pct.plot.bar(ax=ax, color="steelblue")
    ax.set_title("Missing Value Percentage per Parameter (after fault removal)", fontsize=13)
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_values.png", dpi=120)
    plt.close()
    print("  [fig] missing_values.png")


def plot_distributions(df: pd.DataFrame):
    df = _clean(df)
    cols = [c for c in NUMERIC_PARAMS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        if col in NORMAL_BANDS:
            lo, hi = NORMAL_BANDS[col]
            ax.axvspan(lo, hi, alpha=0.15, color="green", label="Normal band")
        ax.hist(data, bins=80, color="royalblue", edgecolor="none", alpha=0.8)
        ax.set_title(PARAM_LABELS.get(col, col), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Parameter Distributions (fault sentinels removed)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "parameter_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  [fig] parameter_distributions.png")


def plot_correlation(df: pd.DataFrame):
    df = _clean(df)
    cols = [c for c in NUMERIC_PARAMS if c in df.columns]
    sample = df[cols].dropna().sample(min(50000, len(df)), random_state=42)
    corr = sample.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Parameter Correlation Matrix (50 k random sample)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=120)
    plt.close()
    print("  [fig] correlation_matrix.png")


def plot_time_series_per_file(df: pd.DataFrame):
    """Plot key parameters over elapsed time, one file = one colour."""
    df = _clean(df)
    plot_cols = ["E1TQ_S76C", "E2TQ_S76C", "PRALT", "Indicated_Airspeed"]
    plot_cols = [c for c in plot_cols if c in df.columns]
    n = len(plot_cols)
    files = df["source_file"].unique()
    palette = sns.color_palette("tab10", n_colors=len(files))
    file_color = {f: palette[i] for i, f in enumerate(files)}

    fig, axes = plt.subplots(n, 1, figsize=(18, n * 3), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_cols):
        for fname, grp in df.groupby("source_file"):
            sub = grp[["elapsed_seconds", col]].dropna()
            if len(sub) == 0:
                continue
            # downsample to at most 5 000 points per file for speed
            step = max(1, len(sub) // 5000)
            sub = sub.iloc[::step]
            ax.plot(sub["elapsed_seconds"] / 3600, sub[col],
                    lw=0.6, alpha=0.7, color=file_color[fname])
        if col in NORMAL_BANDS:
            lo, hi = NORMAL_BANDS[col]
            ax.axhline(lo, color="red", lw=0.8, ls="--", alpha=0.5)
            ax.axhline(hi, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(PARAM_LABELS.get(col, col), fontsize=9)
        ax.set_xlabel("Elapsed Time (hours within file)")
    fig.suptitle("Key Parameters over Time (each colour = one recording session)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_series_per_file.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  [fig] time_series_per_file.png")


def plot_exceedance_counts(df: pd.DataFrame):
    exceed_cols = [c for c in df.columns if c.endswith("_exceed")]
    if not exceed_cols:
        return
    counts = {c.replace("_exceed", ""): int(df[c].fillna(0).sum()) for c in exceed_cols}
    fig, ax = plt.subplots(figsize=(12, 5))
    pd.Series(counts).sort_values(ascending=False).plot.bar(ax=ax, color="tomato")
    ax.set_title("Exceedance Event Counts by Parameter", fontsize=13)
    ax.set_ylabel("Count of 1-second exceedance observations")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exceedance_counts.png", dpi=120)
    plt.close()
    print("  [fig] exceedance_counts.png")


def plot_exceedance_by_file(df: pd.DataFrame):
    if "any_exceedance" not in df.columns:
        return
    grp = df.groupby("source_file")["any_exceedance"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    grp.plot.bar(ax=ax, color="darkorange")
    ax.set_title("Total Exceedance Seconds by Recording Session", fontsize=13)
    ax.set_ylabel("Seconds with ≥1 parameter exceeding normal band")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exceedance_by_file.png", dpi=120)
    plt.close()
    print("  [fig] exceedance_by_file.png")


def plot_flight_duration_hist(flight_report: pd.DataFrame):
    if flight_report is None or "duration_hours" not in flight_report.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    valid = flight_report["duration_hours"].dropna()
    valid.hist(bins=40, ax=ax, color="mediumseagreen", edgecolor="white")
    ax.axvline(3.0, color="red", lw=1.5, ls="--", label="3-hour limit")
    ax.set_title("Flight Segment Duration Distribution", fontsize=13)
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Number of segments")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "flight_duration_hist.png", dpi=120)
    plt.close()
    print("  [fig] flight_duration_hist.png")


def run(df: pd.DataFrame, flight_report: pd.DataFrame | None = None):
    print("Running EDA ...")
    df_clean = df.copy()

    plot_missing_heatmap(df_clean)
    plot_distributions(df_clean)
    plot_correlation(df_clean)
    plot_time_series_per_file(df_clean)

    if "any_exceedance" in df_clean.columns:
        plot_exceedance_counts(df_clean)
        plot_exceedance_by_file(df_clean)

    if flight_report is not None:
        plot_flight_duration_hist(flight_report)

    print(f"  All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    df = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "processed" / "features.parquet"
    )
    run(df)
