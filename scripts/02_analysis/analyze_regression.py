# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm # 회귀 분석 라이브러리
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob 
import sys 

# --- 데이터 경로 설정 (Mac 로컬 파일 시스템 기준) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..') # BDP-Airquality-Analysis 루트 폴더

OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'results', 'pandas_analysis')

# 입력: 최종 통합 및 월별 집계가 완료된 CSV 파일
UNIFIED_MERGED_CSV = os.path.join(OUTPUT_DIR, "unified_national_merged_data.csv")

# --- 분석 대상 설정 (최적화된 값 사용) ---
TARGET_POLLUTANT = 'national_avg_PM10' # 종속 변수: PM10 사용 (PM25 결측치 문제 우회)
OPTIMAL_LAG = 4 # Lagged Correlation 분석에서 도출된 최적 시간차 (4개월)

# --- 출력 경로 설정 ---
OUTPUT_LOCAL_SUMMARY = os.path.join(OUTPUT_DIR, "regression_summary_PM10_final.txt")
OUTPUT_LOCAL_HEATMAP = os.path.join(OUTPUT_DIR, "correlation_heatmap_pm10.png")


def create_spark_session():
    # Spark Session 생성 (Pandas만 사용하므로 이 함수는 사용되지 않음)
    pass

def analyze_regression():
    
    # 1. 통합 데이터 로드
    print("=== 1. 통합 CSV 데이터 로드 시작 ===")
    try:
        # 통합된 CSV 파일은 Date 인덱스를 가지고 저장되었으므로 index_col=0 사용
        df_pandas = pd.read_csv(UNIFIED_MERGED_CSV, index_col=0, parse_dates=True)
        df_pandas.sort_index(inplace=True)
        print(f"-> 데이터 로드 완료. 총 {len(df_pandas)}개월의 시계열 데이터.")
        
    except FileNotFoundError:
        print(f"❌ 오류: 통합 데이터 파일({UNIFIED_MERGED_CSV})을 찾을 수 없습니다.")
        print("      'unified_analysis_parquet.py'를 먼저 실행하여 이 파일을 생성하세요.")
        return

    # 2. 분석용 시계열 및 변수 준비 (여기서 df_pandas는 이미 월평균 집계 데이터임)
    
    # 3. 다중 회귀 모델링 (Multiple Regression)
    print("\n=== 2. 다중 회귀 모델 설정 및 적합 ===")
    
    # --- 핵심 Lag 변수 생성 ---
    df = df_pandas.copy() # 원본 데이터 보존
    lag_col_name = f'Power_GWh_Lag{OPTIMAL_LAG}'
    df[lag_col_name] = df['Power_GWh'].shift(OPTIMAL_LAG)
    
    # --- 통제 변수 생성 ---
    df['month'] = df.index.month
    
    # 계절성 통제를 위한 월(Month) 더미 변수 생성
    month_dummies = pd.get_dummies(df['month'], prefix='Month', drop_first=True) # 1월을 기준(Reference)
    df = pd.concat([df, month_dummies], axis=1)

    # 장기 추세(Trend) 변수 생성
    df['Trend'] = np.arange(len(df))
    
    # Lagging으로 인해 발생한 NaN 행 및 분석에 필요 없는 행 제거
    df_regress = df.dropna()

    # 종속 변수 (Y): PM10
    Y = df_regress[TARGET_POLLUTANT]
    
    # 독립 변수 (X): Lagged Power, Trend, Month Dummies
    X_vars = [lag_col_name, 'Trend'] + [c for c in df_regress.columns if c.startswith('Month_')]
    
    X = df_regress[X_vars]
    
    # ********************* 오류 해결: 타입 강제 변환 (Final Fix) *********************
    # Statsmodels ValueError 방지를 위해 모든 변수를 Float 타입으로 명시적으로 변환합니다.
    Y = Y.astype(float)
    X = X.astype(float)
    # ******************************************************************************

    X = sm.add_constant(X) # 절편(Intercept) 추가

    # OLS (Ordinary Least Squares) 모델 적합
    model = sm.OLS(Y, X).fit()

    # 4. 결과 출력 및 저장
    print("\n=== 3. 회귀 분석 결과 요약 ===")
    summary_text = model.summary().as_text()
    
    print(summary_text)

    # 결과를 로컬 파일로 저장
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(OUTPUT_LOCAL_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    print(f"\n✅ 최종 분석 완료. 요약 결과가 로컬에 저장되었습니다: {OUTPUT_LOCAL_SUMMARY}")
    
def main():
    analyze_regression()

if __name__ == "__main__":
    main()