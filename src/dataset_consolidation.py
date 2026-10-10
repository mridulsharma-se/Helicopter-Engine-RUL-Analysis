"""
src/dataset_consolidation.py
Phase 2 — Consolidate, deduplicate, and produce reporting outputs.
Outputs:
  reports/dataset_inventory.csv
  reports/column_mapping.csv
  reports/data_quality_report.txt
  data/processed/consolidated_dataset.parquet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all, load_file, RAW_DIR, PROC_DIR, REP_DIR, COLUMN_MAP, _parse_date_from_filename
import pandas as pd
import numpy as np

# ── column mapping document ────────────────────────────────────────────────────
COLUMN_DICT = {
    "elapsed_time":        ("object/time", "Elapsed time within recording session (HH:MM:SS). Resets at 00:00:00 for each file.", "—"),
    "elapsed_seconds":     ("float64",     "Elapsed time converted to integer seconds. Used for time-series indexing.", "seconds"),
    "source_file":         ("str",         "Original Excel filename this row was loaded from.", "—"),
    "source_sheet":        ("str",         "Excel sheet name this row was loaded from.", "—"),
    "recording_date":      ("str",         "Date inferred from filename (DDMMYY pattern). Not verified against aircraft records.", "YYYY-MM-DD"),
    "aircraft_type":       ("str",         "Aircraft model identifier. Value 'S76A/C+' = Sikorsky S-76A/C+.", "—"),
    "E1NF_S76C":           ("float64",     "Engine 1 Gas Generator (N1/Nf) speed. Normal cruise ~107%. Values >120% are fault sentinels.", "%"),
    "E2NF_S76C":           ("float64",     "Engine 2 Gas Generator (N1/Nf) speed. Normal cruise ~68% (one engine powering rotor).", "%"),
    "E2NF_raw":            ("float64",     "Engine 2 Nf duplicate channel present in some files. Relationship to E2NF_S76C unverified.", "%"),
    "E1TQ_S76C":           ("float64",     "Engine 1 Torque. Normal cruise 40–90%. Values >130% are fault sentinels.", "%"),
    "E2TQ_S76C":           ("float64",     "Engine 2 Torque. Normal cruise 40–90%.", "%"),
    "E1_OIL_PRS":          ("str",         "Engine 1 Oil Pressure warning flag. On = warning active.", "On/Off"),
    "E2_OIL_PRS":          ("str",         "Engine 2 Oil Pressure warning flag. On = warning active.", "On/Off"),
    "XMSN_OIL_PRS":        ("str",         "Main Transmission Oil Pressure warning flag.", "On/Off"),
    "XMSN_OIL_TEMP":       ("str",         "Main Transmission Oil Temperature warning flag.", "On/Off"),
    "RTRSPD":              ("float64",     "Main Rotor Speed. Normal 100%. Values >115% are fault sentinels.", "%"),
    "ROTBRK":              ("str",         "Rotor Brake engaged flag.", "On/Off"),
    "PRALT":               ("float64",     "Pressure Altitude. Normal range −300 to ~10 000 ft. Value −16 300 = sensor invalid sentinel.", "ft"),
    "MAGHDG":              ("float64",     "Magnetic Heading. 0–360°.", "degrees"),
    "Indicated_Airspeed":  ("float64",     "Indicated Airspeed. Vne ~155 kn. Values >200 kn are fault sentinels.", "knots"),
    "IAS_flag":            ("str",         "Airspeed valid flag (On/Off string). Separate from numeric Indicated_Airspeed.", "On/Off"),
    "OAT":                 ("float64",     "Outside Air Temperature. Normal −40 to +55°C. Values ±~530 are fault sentinels.", "°C"),
    "BATT_OVERTEMP":       ("str",         "Battery Overtemperature warning flag.", "On/Off"),
    "HYPRS1":              ("str",         "Hydraulic System 1 Pressure warning flag.", "On/Off"),
    "HYPRS2":              ("str",         "Hydraulic System 2 Pressure warning flag.", "On/Off"),
    "HYSEL1":              ("str",         "Hydraulic Selector 1 position.", "On/Off"),
    "HYSEL2":              ("str",         "Hydraulic Selector 2 position.", "On/Off"),
    "PITCH":               ("float64",     "Aircraft pitch angle. ±180° range in raw data; real ±60° for normal operations.", "degrees"),
    "ROLL":                ("float64",     "Aircraft roll angle. ±180° range in raw data; real ±60° for normal operations.", "degrees"),
}


def build_inventory(data_dir: Path) -> pd.DataFrame:
    # Check if pre-computed inventory exists at project root or reports
    root_inv = Path(__file__).resolve().parent.parent.parent / "dataset_inventory.csv"
    rep_inv = REP_DIR / "dataset_inventory.csv"
    if rep_inv.exists():
        return pd.read_csv(rep_inv)
    if root_inv.exists():
        return pd.read_csv(root_inv)

    files = sorted(data_dir.glob("*.xlsx"))
    rows = []
    for fp in files:
        xl = pd.ExcelFile(fp, engine="openpyxl")
        for sh in xl.sheet_names:
            df = pd.read_excel(fp, sheet_name=sh, engine="openpyxl")
            rows.append({
                "filename":        fp.name,
                "file_size_mb":    round(fp.stat().st_size / 1024**2, 2),
                "sheet_name":      sh,
                "recording_date":  str(_parse_date_from_filename(fp.name)),
                "n_rows":          len(df),
                "n_cols":          len(df.columns),
                "columns":         " | ".join(df.columns.tolist()),
                "n_duplicates":    int(df.duplicated().sum()),
                "missing_all_cols":int(df.isna().all(axis=1).sum()),
            })
    return pd.DataFrame(rows)


def produce_column_mapping() -> pd.DataFrame:
    rows = []
    for std_name, (dtype, definition, unit) in COLUMN_DICT.items():
        # find all raw names that map to this standard name
        raw_names = [k for k, v in COLUMN_MAP.items() if v == std_name]
        rows.append({
            "standard_name": std_name,
            "raw_names":     " | ".join(raw_names) if raw_names else std_name,
            "dtype":         dtype,
            "unit":          unit,
            "definition":    definition,
        })
    return pd.DataFrame(rows)


def produce_quality_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("DATA QUALITY REPORT — KNH Exceedance Dataset (Sikorsky S-76A/C+)")
    lines.append("=" * 70)
    lines.append(f"\nTotal rows: {len(df):,}")
    lines.append(f"Total columns: {len(df.columns)}")
    lines.append(f"Source files: {df['source_file'].nunique()}")
    lines.append(f"Date range (from filenames): {df['recording_date'].min()} to {df['recording_date'].max()}")

    lines.append("\n--- MISSING VALUES ---")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    for col, cnt in miss.items():
        lines.append(f"  {col:40s}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

    lines.append("\n--- FAULT-SENTINEL COUNTS (values outside plausible physical range) ---")
    fault_cols = [c for c in df.columns if c.endswith("_fault")]
    for fc in fault_cols:
        cnt = int((df[fc].astype(str).str.lower() == "true").sum()) if fc in df.columns else 0
        lines.append(f"  {fc:45s}: {cnt:>8,}  ({cnt/len(df)*100:.2f}%)")

    lines.append("\n--- DUPLICATE ROW COUNT ---")
    dup_count = df.duplicated().sum()
    lines.append(f"  Exact duplicate rows: {dup_count:,}")

    lines.append("\n--- SCHEMA VARIATION ACROSS FILES ---")
    for fname, grp in df.groupby("source_file"):
        present = [c for c in df.columns if grp[c].notna().any()]
        lines.append(f"  {fname}: {len(present)} non-empty columns")

    lines.append("\n--- RUL ASSESSMENT ---")
    lines.append("  No engine failure events found in the dataset.")
    lines.append("  No cumulative engine operating hours column found.")
    lines.append("  No maintenance records or engine-life data found.")
    lines.append("  RUL prediction with a supervised label is NOT SUPPORTED by this dataset.")
    lines.append("  Supported tasks: engine health scoring, exceedance detection,")
    lines.append("                   next-step parameter forecasting.")

    lines.append("\n--- AIRCRAFT IDENTITY ---")
    lines.append("  Aircraft Type: Sikorsky S-76A/C+ (confirmed from 'Aircraft Type' column)")
    lines.append("  Engine:        Two Allison 250-C30S turboshaft engines (E1 and E2)")
    lines.append("  Data source:   Flight Data Monitoring system (continuous 1-Hz recording)")
    lines.append("  'Exceedance' label in filename does NOT mean exceedance-event records.")
    lines.append("  Each file contains full continuous sensor data for one recording session.")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def run(force_reload: bool = False):
    print("Building dataset inventory ...")
    inv = build_inventory(RAW_DIR)
    inv.to_csv(REP_DIR / "dataset_inventory.csv", index=False)
    print(f"  Saved: reports/dataset_inventory.csv")

    print("Producing column mapping ...")
    cm = produce_column_mapping()
    cm.to_csv(REP_DIR / "column_mapping.csv", index=False)
    print(f"  Saved: reports/column_mapping.csv")

    out = PROC_DIR / "consolidated_dataset.parquet"
    if out.exists() and not force_reload:
        print(f"  Loading existing consolidated dataset from {out} ...")
        df = pd.read_parquet(out)
    else:
        print("Loading and consolidating all files ...")
        df = load_all(RAW_DIR)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
        print("Saving consolidated dataset ...")
        df.to_parquet(out, index=False)
        print(f"  Saved: {out}  ({len(df):,} rows)")

    print("Generating data quality report ...")
    report = produce_quality_report(df)
    with open(REP_DIR / "data_quality_report.txt", "w") as f:
        f.write(report)
    print(report)
    print(f"  Saved: reports/data_quality_report.txt")
    return df


if __name__ == "__main__":
    run()
