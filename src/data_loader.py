"""
src/data_loader.py
Phase 2 — Load and consolidate the nine KNH Exceedance Excel files.

Key design decisions (based on Phase 1 findings):
  - Each file contains 1-Hz continuous flight data for a Sikorsky S-76A/C+
  - HH:MM:SS is an elapsed-time column (no absolute date embedded)
  - Recording date is inferred from the filename (DDMMYY pattern)
  - Schema varies slightly across files; only unambiguously equivalent columns
    are mapped to a common name; all others are preserved with a prefix
  - Impossible sensor values (fault-code sentinels) are flagged but NOT deleted
"""

import re
import warnings
from pathlib import Path
from datetime import date, timedelta, datetime, time as dtime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"
REP_DIR   = Path(__file__).resolve().parent.parent / "reports"

for d in (RAW_DIR, PROC_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── known sensor fault-code sentinels (from Phase 1 min/max analysis) ─────────
# Values at these thresholds are likely FDR invalid-data markers, not real readings.
FAULT_THRESHOLDS = {
    "PRALT":            (-300.0,   11500.0),   # ft  — real range ~−300 to 10 000; -16300 is sentinel
    "OAT":              (-100.0,   60.0),       # °C  — real range ~−40  to  55
    "E1NF_S76C":        (0.0,      120.0),      # %   — real range   0   to ~110
    "E2NF_S76C":        (0.0,      120.0),
    "E1TQ_S76C":        (0.0,      130.0),      # %   — real range   0   to ~120
    "E2TQ_S76C":        (0.0,      130.0),
    "RTRSPD":           (0.0,      115.0),      # %
    "Indicated_Airspeed": (0.0,   200.0),       # kn  — Vne ~155 kn for S-76
    "PITCH":            (-90.0,    90.0),        # deg — ±90° is hard physical limit
    "ROLL":             (-180.0,   180.0),
}

# ── canonical column mapping ───────────────────────────────────────────────────
# Maps every raw column name (lower-stripped) to a standardised name.
# Columns NOT listed here are kept under their original name.
COLUMN_MAP = {
    "hh:mm:ss":                           "elapsed_time",
    "aircraft type":                      "aircraft_type",
    "e1nf s76c":                          "E1NF_S76C",
    "e1nf s76c+":                         "E1NF_S76C_plus", # variant channel kept distinct from E1NF_S76C
    "e2nf s76c":                          "E2NF_S76C",
    "e2nf":                               "E2NF_raw",    # standalone — may duplicate E2NF_S76C; kept separate
    "e1tq s76c":                          "E1TQ_S76C",
    "e2tq s76c":                          "E2TQ_S76C",
    "e1oilprs":                           "E1_OIL_PRS",
    "e2oilprs":                           "E2_OIL_PRS",
    "main transmission oil pressure":     "XMSN_OIL_PRS",
    "main transmission oil temp":         "XMSN_OIL_TEMP",
    "rtrspd":                             "RTRSPD",
    "rotbrk":                             "ROTBRK",
    "pralt":                              "PRALT",
    "maghdg":                             "MAGHDG",
    "indicated airspeed":                 "Indicated_Airspeed",
    "ias":                                "IAS_flag",    # On/Off flag; distinct from numeric IAS
    "oat":                                "OAT",
    "battery overtemp":                   "BATT_OVERTEMP",
    "hyprs1":                             "HYPRS1",
    "hyprs2":                             "HYPRS2",
    "hysel1":                             "HYSEL1",
    "hysel2":                             "HYSEL2",
    "pitch":                              "PITCH",
    "roll":                               "ROLL",
}


def _parse_date_from_filename(fname: str) -> date | None:
    """
    Extract the recording date from filename patterns like:
      'KNH Exceedance 030226.xlsx'  → DDMMYY → 2026-02-03
      'KNH Exceedance 091125.xlsx'  → DDMMYY → 2025-11-09
    Returns None if no 6-digit sequence is found.
    """
    m = re.search(r"(\d{6})", fname)
    if not m:
        return None
    token = m.group(1)
    dd, mm, yy = int(token[:2]), int(token[2:4]), int(token[4:])
    year = 2000 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply COLUMN_MAP to rename columns; strip/clean names first."""
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = df.rename(columns=rename)
    # Clean remaining column names (replace spaces with underscores)
    df.columns = [c.replace(" ", "_") if c not in COLUMN_MAP.values() else c
                  for c in df.columns]
    # Safeguard against any duplicated column names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _parse_elapsed_seconds(series: pd.Series) -> pd.Series:
    """
    Convert the HH:MM:SS column (stored as datetime.time objects or strings)
    to total elapsed seconds (integer).
    """
    def to_sec(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, dtime):
            return v.hour * 3600 + v.minute * 60 + v.second
        if isinstance(v, timedelta):
            return int(v.total_seconds())
        try:
            parts = str(v).split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        except Exception:
            return np.nan
    return series.map(to_sec)


def _flag_fault_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each known numeric column, add a boolean _fault flag where the value
    is outside the plausible physical range. Original values are preserved.
    """
    for col, (lo, hi) in FAULT_THRESHOLDS.items():
        if col in df.columns:
            mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            df[col + "_fault"] = mask
    return df


def load_file(filepath: Path) -> pd.DataFrame:
    """
    Load one Excel workbook (one sheet) and return a clean DataFrame with
    source metadata columns added.
    """
    filepath = Path(filepath)
    recording_date = _parse_date_from_filename(filepath.name)

    # Identify the sheet and read data
    with pd.ExcelFile(filepath, engine="openpyxl") as xl:
        sheet_name = xl.sheet_names[0]
        df = pd.read_excel(xl, sheet_name=sheet_name)

    # ── metadata columns ──────────────────────────────────────────────────────
    df.insert(0, "source_file",  filepath.name)
    df.insert(1, "source_sheet", sheet_name)
    df.insert(2, "recording_date", str(recording_date) if recording_date else "unknown")

    # ── standardise column names ──────────────────────────────────────────────
    df = _standardise_columns(df)

    # ── drop fully-empty leading rows (Excel header-gap rows) ─────────────────
    # Rows where every column except elapsed_time is NaN
    non_time_cols = [c for c in df.columns if c not in ("elapsed_time", "source_file",
                                                          "source_sheet", "recording_date")]
    mask_empty = df[non_time_cols].isna().all(axis=1)
    df = df[~mask_empty].copy()

    # ── convert elapsed_time to seconds ──────────────────────────────────────
    if "elapsed_time" in df.columns:
        df["elapsed_seconds"] = _parse_elapsed_seconds(df["elapsed_time"])
        df["elapsed_time"] = df["elapsed_time"].astype(str)

    # ── flag fault values ─────────────────────────────────────────────────────
    df = _flag_fault_values(df)

    df = df.reset_index(drop=True)
    return df


def load_all(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load and vertically concatenate all Excel files in data_dir.
    Files with incompatible extra columns retain them with NaN in other files.
    """
    target_dir = Path(data_dir) if data_dir else RAW_DIR
    files = sorted(target_dir.glob("*.xlsx")) + sorted(target_dir.glob("*.xls"))
    files = [f for f in files if not f.name.startswith("~")]

    # If no files found in RAW_DIR, check parent workspace directory
    if not files and not data_dir:
        parent_candidate = RAW_DIR.resolve().parent.parent.parent
        files = sorted(parent_candidate.glob("*.xlsx")) + sorted(parent_candidate.glob("*.xls"))
        files = [f for f in files if not f.name.startswith("~")]
        if files:
            print(f"  Found {len(files)} Excel files in parent workspace: {parent_candidate}")

    if not files:
        # Check sample fallback
        sample_path = RAW_DIR.parent / "sample" / "sample_telemetry.csv"
        if sample_path.exists():
            print(f"  No Excel files found; loading sample data from {sample_path.name} ...")
            return pd.read_csv(sample_path)
        raise FileNotFoundError(f"No Excel files found in {target_dir} or parent directories.")

    frames = []
    for fp in files:
        print(f"  Loading: {fp.name} ...", end=" ", flush=True)
        df = load_file(fp)
        print(f"{len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  Total combined rows: {len(combined):,}")
    return combined


if __name__ == "__main__":
    df = load_all()
    out = PROC_DIR / "consolidated_dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved to {out}")
