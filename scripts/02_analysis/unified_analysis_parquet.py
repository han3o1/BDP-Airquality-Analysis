# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob 
import sys 
import statsmodels.api as sm 

# --- 데이터 경로 설정 (Mac 로컬 파일 시스템 기준) ---
# NOTE: 이 스크립트는 scripts/02_analysis/ 폴더에 위치하며, 두 단계 상위로 이동하여 PROJECT_ROOT를 찾습니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..') # BDP-Airquality-Analysis 루트 폴더

# 1. Air Quality 전국 월평균 데이터 (다운로드된 Parquet 폴더 경로)
#    경로: [프로젝트 루트]/data/preprocessed_data/year_XXXX 폴더들을 포함하는 BASE 경로
AIR_QUALITY_BASE_PATH = os.path.join(
    PROJECT_ROOT_DIR, 
    'data', 
    'preprocessed_data' # <- 이 폴더 안에 year_2003, year_2004 등의 폴더가 있어야 합니다.
) 

# 2. 화력발전 원본 데이터 (CSV)
#    경로: [프로젝트 루트]/data/raw/kepco_thermal_power_monthly.csv
POWER_DATA_CSV = os.path.join(
    PROJECT_ROOT_DIR, 
    'data', 
    'raw', 
    'kepco_thermal_power_monthly.csv'
)

# --- 분석 대상 컬럼 설정 ---
TARGET_POLLUTANT = 'national_avg_PM10' # PM10을 종속 변수로 사용
OPTIMAL_LAG = 1 

# --- 출력 경로 설정 (Mac 로컬 환경에 결과 저장) ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'results', 'pandas_analysis')
OUTPUT_LOCAL_MERGED_CSV = os.path.join(OUTPUT_DIR, "unified_national_merged_data.csv")


def prepare_data():
    """AirQuality Parquet과 화력발전 데이터를 Pandas로 로드 및 통합합니다."""
    
    # 1. Air Quality 데이터 로드 (Parquet 사용)
    print("=== 1. Air Quality 데이터 로드 (전체 연도 Parquet 통합) ===")
    
    # **전체 연도 폴더 검색** (year_2003, year_2004 등)
    all_year_folders = glob.glob(os.path.join(AIR_QUALITY_BASE_PATH, 'year_*'))
    
    if not all_year_folders:
        raise FileNotFoundError(f"'{AIR_QUALITY_BASE_PATH}' 폴더 내에서 연도별 데이터 폴더(year_XXXX)를 찾을 수 없습니다.")

    # 각 연도 폴더 내의 모든 .parquet 파일을 찾습니다.
    all_parquet_files = []
    for folder in all_year_folders:
        all_parquet_files.extend(glob.glob(os.path.join(folder, '*.parquet')))
    
    if not all_parquet_files:
        raise FileNotFoundError("연도별 폴더에서 Parquet 파일(*.parquet)이 발견되지 않았습니다.")
    
    # Parquet 파일 목록을 읽어 DataFrame 생성 (시간별/지역별 원시 데이터)
    df_air = pd.read_parquet(all_parquet_files)
    print(f"-> Parquet 파일 {len(all_parquet_files)}개 로드 완료. 총 {len(df_air)} 행의 시간별 데이터.")


    # 2. Air Quality 데이터 전처리 및 연월 추출
    # Parquet 파일에 year, month 컬럼이 없거나 파티션 메타데이터로 로드되지 않았을 경우 강제 추출
    if 'year' not in df_air.columns:
        df_air['year'] = df_air['date_time'].astype(str).str.slice(0, 4).astype(int)
    if 'month' not in df_air.columns:
        df_air['month'] = df_air['date_time'].astype(str).str.slice(4, 6).astype(int)
    
    # 원시 PM10 컬럼을 float으로 변환 (이전에 확인된 'pm10_z' 컬럼을 사용해야 함)
    PM10_RAW_COL = 'pm10_z'
    df_air[PM10_RAW_COL] = pd.to_numeric(df_air[PM10_RAW_COL], errors='coerce') 
    
    
    # 3. 화력발전 데이터 로드 및 Unpivot (Wide -> Long)
    print("=== 2. 화력발전 데이터 로드 및 Unpivot (Wide -> Long) ===")
    df_power_wide = pd.read_csv(POWER_DATA_CSV)
    
    # Wide Format을 Long Format으로 변환
    df_power_wide.rename(columns={df_power_wide.columns[0]: 'month'}, inplace=True)
    for col_name in df_power_wide.columns[1:]:
        df_power_wide[col_name] = df_power_wide[col_name].astype(str).str.replace(',', '').astype(float)
    
    df_power_long = df_power_wide.melt(id_vars=['month'], var_name='year', value_name='Power_GWh')
    df_power_long['year'] = df_power_long['year'].astype(int)
    df_power_long['month'] = df_power_long['month'].astype(int)
    
    
    # 4. 데이터 통합 (Merge) 및 최종 집계 (핵심!)
    print("=== 3. 최종 목표: 전국 월평균 집계 (264행 생성) ===")
    
    # Air Quality 원시 데이터와 월별 Power 데이터를 병합 (이 시점에서 Power_GWh가 중복 복사됨)
    df_merged_raw = pd.merge(df_air, df_power_long, on=['year', 'month'], how='inner')
    
    # 최종 집계: PM10 원시 데이터의 전국 평균을 계산하고 Power_GWh 중복을 제거
    df_national_monthly = df_merged_raw.groupby(['year', 'month']).agg(
        # PM10의 전국 평균 계산
        national_avg_PM10=(PM10_RAW_COL, 'mean'), 
        # Power_GWh는 이미 월별 값이므로 첫 번째 값만 사용 (중복 제거)
        Power_GWh=('Power_GWh', 'first') 
    ).reset_index()

    # Date 인덱스 생성
    df_national_monthly['Date'] = pd.to_datetime(df_national_monthly['year'].astype(str) + '-' + df_national_monthly['month'].astype(str) + '-01')
    df_national_monthly.set_index('Date', inplace=True)
    df_national_monthly.sort_index(inplace=True)
    
    print(f"-> 최종 집계 완료. 총 {len(df_national_monthly)}개월의 데이터 (원하는 형식).")

    return df_national_monthly

def save_merged_data(df):
    """통합된 DataFrame을 로컬 CSV 파일로 저장합니다."""
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # CSV 파일로 저장 (다음 분석 스크립트의 입력 파일이 됨)
    df.to_csv(OUTPUT_LOCAL_MERGED_CSV, index=True, encoding='utf-8-sig')
    print(f"\n✅ 데이터 통합 및 최종 집계 완료! 통합 파일이 로컬에 저장되었습니다: {OUTPUT_LOCAL_MERGED_CSV}")


if __name__ == "__main__":
    try:
        final_merged_df = prepare_data()
        save_merged_data(final_merged_df)
    except Exception as e:
        print(f"\n❌ 최종 실행 오류: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)