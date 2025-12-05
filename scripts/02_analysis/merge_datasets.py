# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob

# --- 데이터 경로 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트 파일의 경로

# 1. Air Quality 전국 월평균 데이터
AIR_QUALITY_PARQUET_PATH = "/Users/haneol/Desktop/2025_2/BDP/팀프로젝트/BDP-Airquality-Analysis/data/preprocessed_data/year2003_z_parquet" 

# 2. 화력발전 원본 데이터
POWER_DATA_CSV = "/Users/haneol/Desktop/2025_2/BDP/팀프로젝트/BDP-Airquality-Analysis/data/raw/kepco_thermal_power_monthly.csv" 

# --- 출력 경로 설정 ---
OUTPUT_DIR = "/Users/haneol/Desktop/2025_2/BDP/팀프로젝트/BDP-Airquality-Analysis/results/pandas_analysis"
OUTPUT_LOCAL_MERGED_CSV = os.path.join(OUTPUT_DIR, "unified_national_merged_data.csv")

def prepare_data():
    """AirQuality Parquet과 화력발전 데이터를 Pandas로 로드 및 통합합니다."""
    
    # 1. Air Quality 데이터 로드 (Parquet 사용)
    print("=== 1. Air Quality 데이터 로드 (Parquet 사용) ===")
    
    parquet_files = glob.glob(os.path.join(AIR_QUALITY_PARQUET_PATH, '*.parquet'))

    if not parquet_files:
        raise FileNotFoundError(f"Parquet 파일이 '{AIR_QUALITY_PARQUET_PATH}' 폴더 내에 없습니다.")

    df_air = pd.read_parquet(parquet_files)
    
    # year와 month 컬럼 타입 변환 (통합 키)
    if 'year' not in df_air.columns or 'month' not in df_air.columns:
         raise KeyError("Air Quality Parquet 파일에 'year' 또는 'month' 컬럼이 누락되었습니다.")

    df_air['year'] = df_air['year'].astype(int)
    df_air['month'] = df_air['month'].astype(int)
    
    
    # 2. 화력발전 데이터 로드 및 Unpivot (Wide -> Long)
    print("=== 2. 화력발전 데이터 로드 및 Unpivot (Wide -> Long) ===")
    df_power_wide = pd.read_csv(POWER_DATA_CSV)
    
    # 컬럼 정리 및 숫자 변환
    df_power_wide.rename(columns={df_power_wide.columns[0]: 'month'}, inplace=True)
    for col_name in df_power_wide.columns[1:]:
        # 콤마 제거 및 float 타입으로 변환
        df_power_wide[col_name] = df_power_wide[col_name].astype(str).str.replace(',', '').astype(float)
    
    # Wide Format을 Long Format으로 변환 (Wide -> Long)
    df_power_long = df_power_wide.melt(
        id_vars=['month'], 
        var_name='year', 
        value_name='Power_GWh'
    )
    
    # Year 컬럼을 Integer로 변환
    df_power_long['year'] = df_power_long['year'].astype(int)
    
    
    # 3. 최종 통합 (Merge)
    print("=== 3. 최종 통합 (Merge) ===")
    df_merged = pd.merge(df_air, df_power_long, on=['year', 'month'], how='inner')
    
    # Date 인덱스 생성 (분석 용이성을 위해)
    df_merged['Date'] = pd.to_datetime(df_merged['year'].astype(str) + '-' + df_merged['month'].astype(str) + '-01')
    df_merged.set_index('Date', inplace=True)
    df_merged.sort_index(inplace=True)
    
    return df_merged

def save_merged_data(df):
    """통합된 DataFrame을 로컬 CSV 파일로 저장합니다."""
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # CSV 파일로 저장 (다음 분석 스크립트의 입력 파일이 됨)
    df.to_csv(OUTPUT_LOCAL_MERGED_CSV, index=True, encoding='utf-8-sig')
    print(f"\n✅ 데이터 통합 완료! 통합 파일이 로컬에 저장되었습니다: {OUTPUT_LOCAL_MERGED_CSV}")


if __name__ == "__main__":
    try:
        final_merged_df = prepare_data()
        save_merged_data(final_merged_df)
    except Exception as e:
        print(f"\n❌ 최종 실행 오류: {e}")