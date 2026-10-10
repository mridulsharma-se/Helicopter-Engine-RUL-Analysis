# Data Directory

## Structure
- `raw/`       — Symlinks or copies of the original KNH Exceedance Excel files (never modified)
- `processed/` — Pipeline outputs (consolidated_dataset.parquet, features.parquet)

## Source Files (9 recordings)
| Filename | Date | Approx Rows |
|---|---|---|
| KNH Exceedance 091125.xlsx | 2025-11-09 | 407,369 |
| KNH Exceedance 120125.xlsx | 2026-01-12 | 235,521 |
| KNH Exceedance 260625.xlsx | 2025-06-26 | 267,471 |
| KNH Exceedance 030226.xlsx | 2026-02-03 | 404,643 |
| KNH Exceedance 210326.xlsx | 2026-03-21 | 406,056 |
| KNH Exceedance 200426.xlsx | 2026-04-20 | 410,562 |
| KNH Exceedance 120526.xlsx | 2026-05-12 | 411,622 |
| KNH Exceedance 300526.xlsx | 2026-05-30 | 404,498 |
| KNh Exceedance 240626.xlsx | 2026-06-24 | 410,835 |

## Important Notes
- These are **continuous 1-Hz flight data recordings**, NOT exceedance-event logs
- Aircraft: Sikorsky S-76A/C+
- Engines: Two Allison 250-C30S turboshaft (E1 and E2)
- The filename "Exceedance" refers to the monitoring programme name, not the data type
