# BDP-Airquality-Analysis

## Overview
- End-to-end pipeline for collecting, cleaning, normalizing, and analyzing air-quality data alongside thermal power generation, with a Streamlit dashboard for exploration.
- Data lives under `data/` (raw + preprocessed parquet) and analysis outputs under `results/`.

## Data Layout (`data/`)
- `data/raw/`: source files (e.g., KEPCO monthly CSV `kepco_thermal_power_monthly.csv`).
- `data/preprocessed_data/year_*/`: cleaned + z-scored Parquet partitions produced by Spark; each year folder contains `part-r-*.parquet` files with pollutant `_z` columns and `year`.

## Analysis Outputs (`results/`)
- `results/pandas_analysis/unified_national_merged_data.csv`: monthly merged PM10 + power (Date index) used by analysis scripts and the dashboard.
- `results/pandas_analysis/lagged_correlation_results_pm10.csv`: Pearson correlations for 1-6 month power lags.
- `results/pandas_analysis/regression_summary_PM10_final.txt`: OLS regression summary (PM10 vs lagged power + trend + month dummies).
- `results/pandas_analysis/seasonal_decomposition_pm10.png` and `seasonal_decomposition_power.png`: seasonal decomposition plots (period=12) for PM10 and power.

## Hadoop Spark Pipeline (`hadoop_code`)
- `airquality_spark_clean_v4.py`: Spark job to parse raw CSV (region, station, date_time, pollutants, address), drop invalid/negative values, and write Parquet.
- `spark_zscore_normalize_v2.py`: Reads cleaned Parquet, computes per-column mean/std for SO2/CO/O3/NO2/PM10/PM25, emits `_z` columns plus `year`, and saves Parquet.
- `run_full_pipeline_auto.sh`: Auto-detects `year=` partitions in HDFS, then runs cleaning and z-score jobs via `spark-submit` (YARN, python2.6) into `/user/airquality/clean_parquet` and `/user/airquality/zscore_parquet`.
- `clean_and_upload.sh` + `clean_only.py`: Local pre-clean of CSVs (remove negatives/invalids) into year/month temp folders, then upload to HDFS under `/user/airquality/data/year=YYYY/month=MM`.
- `export_parquet_to_shared.sh`: Copies z-score Parquet partitions from HDFS to a VM shared folder (`/mnt/hgfs/csv/zscore_results`) with metadata files.

## Data Collection Scripts (`scripts/01_ingest`)
- `windows_collector.py`: Pulls hourly AirKorea measurements by province (ver 1.5 API), enriches with station addresses, normalizes columns, and saves monthly CSVs to `C:\AirKorea_Data` (`data_YYYY_MM.csv`).
- `kepco_collector.py`: Fetches KEPCO thermal generation datasets (multiple URLs), normalizes `date_time`/fuel/power columns, filters target fuels, pivots to monthly totals, and saves `kepco_thermal_power_final.xlsx` to `C:\AirKorea_Data`.

## Pandas Analysis Scripts (`scripts/02_analysis`)
- `unified_analysis_parquet.py`: Reads all preprocessed Parquet files under `data/preprocessed_data/year_*`, extracts year/month, merges with monthly power CSV (`data/raw/kepco_thermal_power_monthly.csv`), aggregates to national monthly PM10 + power, and writes `results/pandas_analysis/unified_national_merged_data.csv` (Date index).
- `analyze_lag_correlation.py`: Loads the unified CSV, computes Pearson correlations between PM10 (`national_avg_PM10`) and lagged power (`Power_GWh`) for 1-6 month shifts, saves `lagged_correlation_results_pm10.csv`.
- `analyze_regression.py`: OLS regression of PM10 on 4-month lagged power + trend + month dummies; prints and saves summary to `regression_summary_PM10_final.txt`.
- `analyze_seasonal_decomposition.py`: Seasonal decomposition (additive, period=12) for PM10 and power, saving plots to `seasonal_decomposition_pm10.png` and `seasonal_decomposition_power.png`.

## Streamlit Dashboard (`dashboard_app.py`)
- Reads annual air-quality parquet partitions from `data/preprocessed_data/year_*/part-r-*.parquet` and power data from `annual_power.csv`; merges to annual table. Uses `results/pandas_analysis/unified_national_merged_data.csv` for seasonal decomposition if present.
- Sidebar: year-range slider, toggle to show raw annual tables.
- Visuals: regression scatter grid (power vs each pollutant z-score), correlation heatmap, pollutant trend lines over time, combined AirQualityIndex vs power regression, and seasonal decomposition plots for PM10 and power.
- Run locally with `streamlit run dashboard_app.py`; ensure data files above are available.
