"""
src/flight_segmentation.py
Phase 3 — Segment each recording session into individual flights.

Method:
  Within each source file (recording session) the HH:MM:SS column counts
  seconds from 00:00:00.  A new flight is identified when elapsed_seconds
  DECREASES (the counter wrapped back to 00:00:00) or when a gap of
  >60 s occurs with E1NF_S76C ≈ 0 (rotor stopped).

The three-hour constraint is verified by comparing the calculated duration
of each flight segment against 10 800 seconds (3 h).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

REP_DIR = Path(__file__).resolve().parent.parent / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)

THREE_HOURS_S = 3 * 3600   # 10 800 seconds
IDLE_NF       = 5.0        # % — threshold below which engine is considered stopped


def segment_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns:
      flight_id          — unique identifier (file_date + sequential index)
      flight_segment_idx — row index within the flight
    """
    df = df.copy()
    df["flight_id"] = ""
    df["flight_segment_idx"] = 0

    flight_counter = 0
    records = []

    for source_file, grp in df.groupby("source_file", sort=False):
        grp = grp.reset_index(drop=True)
        rec_date = grp["recording_date"].iloc[0]

        es_arr = grp["elapsed_seconds"].to_numpy()
        e1nf_arr = grp["E1NF_S76C"].to_numpy() if "E1NF_S76C" in grp.columns else np.full(len(grp), np.nan)
        diffs = np.diff(es_arr)
        
        # New flight when time wraps (diff < 0) or gap > 60 s with near-zero engine speed
        is_wrap = diffs < 0
        is_gap = (diffs > 60) & (np.isnan(e1nf_arr[1:]) | (e1nf_arr[1:] < IDLE_NF))
        cut_mask = is_wrap | is_gap
        cut_indices = np.where(cut_mask)[0] + 1

        starts = np.insert(cut_indices, 0, 0)
        ends = np.append(cut_indices, len(grp))

        for seg_id, (st_idx, end_idx) in enumerate(zip(starts, ends)):
            seg_rows = grp.iloc[st_idx:end_idx]
            flight_id = f"{rec_date}_F{seg_id:03d}"
            records.append(_flight_summary(seg_rows, flight_id, source_file))

    report_df = pd.DataFrame(records)
    return report_df


def _flight_summary(seg: pd.DataFrame, flight_id: str, source_file: str) -> dict:
    es = seg["elapsed_seconds"].dropna()
    if len(es) < 2:
        duration_s = np.nan
    else:
        duration_s = float(es.max() - es.min())

    duration_h = duration_s / 3600 if not np.isnan(duration_s) else np.nan

    if np.isnan(duration_s):
        constraint = "unknown"
    elif duration_s <= THREE_HOURS_S:
        constraint = "within_3h"
    else:
        constraint = "exceeds_3h"

    # Count On-flag warnings (safety events)
    warning_cols = [c for c in seg.columns if seg[c].dtype == object
                    and c not in ("source_file","source_sheet","recording_date",
                                  "aircraft_type","elapsed_time","flight_id")]
    n_warnings = 0
    for wc in warning_cols:
        if wc in seg.columns:
            n_warnings += int((seg[wc] == "On").sum())

    return {
        "flight_id":        flight_id,
        "source_file":      source_file,
        "recording_date":   seg["recording_date"].iloc[0] if "recording_date" in seg.columns else "",
        "n_rows":           len(seg),
        "elapsed_start_s":  float(es.min()) if len(es) else np.nan,
        "elapsed_end_s":    float(es.max()) if len(es) else np.nan,
        "duration_seconds": duration_s,
        "duration_hours":   round(duration_h, 3) if not np.isnan(duration_h) else np.nan,
        "constraint_status":constraint,
        "n_warning_events": n_warnings,
        "missing_data_pct": round(seg.isna().mean().mean() * 100, 2),
    }


def run(df: pd.DataFrame) -> pd.DataFrame:
    print("Segmenting flights ...")
    report = segment_flights(df)
    out = REP_DIR / "flight_duration_report.csv"
    report.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(f"  Total flight segments: {len(report)}")
    print(report["constraint_status"].value_counts().to_string())
    return report


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "processed" / "consolidated_dataset.parquet"
    )
    run(df)
