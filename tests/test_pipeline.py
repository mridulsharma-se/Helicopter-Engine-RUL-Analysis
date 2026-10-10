import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    COLUMN_MAP,
    FAULT_THRESHOLDS,
    _standardise_columns,
    _parse_date_from_filename,
    _parse_elapsed_seconds,
    _flag_fault_values,
)
from flight_segmentation import segment_flights, _flight_summary
from feature_engineering import (
    NORMAL_BANDS,
    clean_faults,
    encode_flags,
    exceedance_flags,
    rolling_features,
    cumulative_operating_time,
)
from train_forecasting_model import build_lagged_dataset
from train_rul_model import RUL_ASSESSMENT


class TestPipeline(unittest.TestCase):
    def test_filename_date_parsing(self):
        self.assertEqual(str(_parse_date_from_filename("KNH Exceedance 030226.xlsx")), "2026-02-03")
        self.assertEqual(str(_parse_date_from_filename("KNH Exceedance 091125.xlsx")), "2025-11-09")
        self.assertEqual(str(_parse_date_from_filename("KNh Exceedance 240626.xlsx")), "2026-06-24")
        self.assertIsNone(_parse_date_from_filename("invalid_filename.xlsx"))

    def test_column_standardisation(self):
        sample_df = pd.DataFrame({
            "HH:MM:SS": ["00:00:01", "00:00:02"],
            "Aircraft Type": ["S76A/C+", "S76A/C+"],
            "E1NF S76C": [107.0, 107.5],
            "E1TQ S76C": [55.0, 56.0],
        })
        standardised = _standardise_columns(sample_df)
        self.assertIn("elapsed_time", standardised.columns)
        self.assertIn("aircraft_type", standardised.columns)
        self.assertIn("E1NF_S76C", standardised.columns)
        self.assertIn("E1TQ_S76C", standardised.columns)

    def test_elapsed_seconds_parsing(self):
        series = pd.Series(["00:00:00", "00:01:30", "01:00:00", None])
        secs = _parse_elapsed_seconds(series)
        self.assertEqual(secs.iloc[0], 0)
        self.assertEqual(secs.iloc[1], 90)
        self.assertEqual(secs.iloc[2], 3600)
        self.assertTrue(np.isnan(secs.iloc[3]))

    def test_fault_threshold_detection(self):
        sample_df = pd.DataFrame({
            "PRALT": [-16300.0, 5000.0, 20000.0],
            "E1NF_S76C": [-10.0, 105.0, 428.0],
            "OAT": [-530.0, 25.0, 350.0],
        })
        flagged = _flag_fault_values(sample_df)
        self.assertEqual(flagged["PRALT_fault"].tolist(), [True, False, True])
        self.assertEqual(flagged["E1NF_S76C_fault"].tolist(), [True, False, True])
        self.assertEqual(flagged["OAT_fault"].tolist(), [True, False, True])

    def test_clean_faults(self):
        sample_df = pd.DataFrame({
            "PRALT": [-16300.0, 5000.0],
            "E1NF_S76C": [428.0, 105.0],
        })
        cleaned = clean_faults(sample_df)
        self.assertTrue(np.isnan(cleaned.loc[0, "PRALT"]))
        self.assertEqual(cleaned.loc[1, "PRALT"], 5000.0)
        self.assertTrue(np.isnan(cleaned.loc[0, "E1NF_S76C"]))
        self.assertEqual(cleaned.loc[1, "E1NF_S76C"], 105.0)

    def test_binary_flags_encoding(self):
        sample_df = pd.DataFrame({
            "E1_OIL_PRS": ["On", "Off", "off", "ON", np.nan],
            "BATT_OVERTEMP": ["Off", "Off", "On", "Off", "Off"],
        })
        encoded = encode_flags(sample_df)
        self.assertEqual(encoded["E1_OIL_PRS_bin"].fillna(-1).tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(encoded["BATT_OVERTEMP_bin"].fillna(-1).tolist(), [0, 0, 1, 0, 0])

    def test_exceedance_detection(self):
        sample_df = pd.DataFrame({
            "E1NF_S76C": [105.0, 115.0, 90.0],   # normal band: 95-110
            "E1TQ_S76C": [50.0, 115.0, 60.0],    # normal band: 0-110
        })
        exceed = exceedance_flags(sample_df)
        self.assertEqual(exceed["E1NF_S76C_exceed"].tolist(), [0, 1, 1])
        self.assertEqual(exceed["E1TQ_S76C_exceed"].tolist(), [0, 1, 0])
        self.assertEqual(exceed["any_exceedance"].tolist(), [0, 1, 1])

    def test_flight_segmentation_logic(self):
        df1 = pd.DataFrame({
            "source_file": ["file1.xlsx"] * 6,
            "recording_date": ["2026-02-03"] * 6,
            "elapsed_seconds": [0, 10, 20, 0, 10, 20],
            "E1NF_S76C": [107.0, 107.0, 107.0, 107.0, 107.0, 107.0],
        })
        report = segment_flights(df1)
        self.assertEqual(len(report), 2)
        self.assertEqual(report.iloc[0]["flight_id"], "2026-02-03_F000")
        self.assertEqual(report.iloc[1]["flight_id"], "2026-02-03_F001")
        self.assertEqual(report.iloc[0]["duration_seconds"], 20.0)
        self.assertEqual(report.iloc[0]["constraint_status"], "within_3h")

    def test_lagged_dataset_construction(self):
        df = pd.DataFrame({
            "source_file": ["test.xlsx"] * 4,
            "elapsed_seconds": [1, 2, 3, 4],
            "E1TQ_S76C": [50.0, 52.0, 54.0, 56.0],
        })
        lagged = build_lagged_dataset(df)
        self.assertEqual(len(lagged), 3)
        self.assertEqual(lagged["E1TQ_S76C_next"].tolist(), [52.0, 54.0, 56.0])

    def test_rul_assessment_statement(self):
        self.assertIn("SUPERVISED RUL PREDICTION IS NOT SUPPORTED", RUL_ASSESSMENT)
        self.assertIn("Allison 250-C30S", RUL_ASSESSMENT)
        self.assertIn("Sikorsky S-76A/C+", RUL_ASSESSMENT)


if __name__ == "__main__":
    unittest.main()
