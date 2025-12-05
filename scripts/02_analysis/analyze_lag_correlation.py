# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import pearsonr # 피어슨 상관계수 계산용
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys

# --- 데이터 경로 설정 (로컬 파일 시스템 기준) ---
# NOTE: 이 스크립트는 scripts/02_analysis/ 폴더에 위치하며, 두 단계 상위로 이동하여 PROJECT_ROOT를 찾습니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..') # BDP-Airquality-Analysis 루트 폴더

# 출력 폴더 경로 (통합된 CSV 파일이 저장된 위치)
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'results', 'pandas_analysis')
# 입력: 최종 통합 데이터셋 (Merge Script의 출력 CSV 파일)
UNIFIED_MERGED_CSV = os.path.join(OUTPUT_DIR, "unified_national_merged_data.csv")

# 출력: 시간차 분석 결과 저장 경로
OUTPUT_LOCAL_LAG_CORR = os.path.join(OUTPUT_DIR, "lagged_correlation_results_pm10.csv")

# --- 분석 대상 설정 ---
TARGET_POLLUTANT = 'national_avg_PM10'   # 종속 변수
PREDICTOR = 'Power_GWh'                  # 예측 변수


def analyze_lag_correlation():
    
    # 1. 통합 데이터 로드
    print("=== 1. 통합 CSV 데이터 로드 시작 ===")
    try:
        # 통합된 CSV 파일은 Date 인덱스를 가지고 저장되었으므로 index_col=0 사용
        df_pandas = pd.read_csv(UNIFIED_MERGED_CSV, index_col=0, parse_dates=True)
        df_pandas.sort_index(inplace=True)
        print(f"-> 데이터 로드 완료. 총 {len(df_pandas)}개월의 데이터.")
        
    except FileNotFoundError:
        print(f"❌ 오류: 통합 데이터 파일({UNIFIED_MERGED_CSV})을 찾을 수 없습니다.")
        print("      'unified_analysis_parquet.py'를 먼저 실행하여 이 파일을 생성하세요.")
        return

    # 2. 분석 대상 컬럼 설정 및 유효성 검사
    df_analysis = df_pandas[[TARGET_POLLUTANT, PREDICTOR]].astype(float) 
    
    lag_results = []
    
    # 3. 선행 지연 상관관계 계산 (Lagged Correlation)
    # Lag 1개월부터 6개월까지 분석 (환경 영향의 시간차를 찾기 위함)
    print("\n=== 2. 발전량(Power)의 시간차(Lag) 상관관계 분석 시작 ===")
    
    for lag in range(1, 7):
        # Lag 컬럼 생성: 발전량 데이터를 lag 기간만큼 아래로 밀기 (shift)
        lag_col_name = f'{PREDICTOR}_Lag_{lag}'
        df_analysis[lag_col_name] = df_analysis[PREDICTOR].shift(lag)
        
        # Nan이 없는 유효한 값만 선택하여 배열 준비
        valid_data = df_analysis[[TARGET_POLLUTANT, lag_col_name]].dropna()
        
        if len(valid_data) > 5: # 최소 5개 이상의 데이터 포인트가 있어야 분석 가능
            correlation, p_value = pearsonr(
                valid_data[TARGET_POLLUTANT],
                valid_data[lag_col_name]
            )
            
            lag_results.append({
                'Lag': lag,
                'Correlation_Coefficient': correlation,
                'P_Value': p_value
            })
            
            print(f"  - Lag {lag}개월: 상관계수 = {correlation:.4f}, P-값 = {p_value:.4f}")
        else:
            print(f"  - Lag {lag}개월: 데이터 포인트 부족으로 분석 불가.")


    # 4. 결과 정리 및 저장
    df_lag_results = pd.DataFrame(lag_results)
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 결과를 로컬 파일로 저장
    df_lag_results.to_csv(OUTPUT_LOCAL_LAG_CORR, index=False)
    
    print(f"\n✅ 분석 완료. 결과 CSV 파일이 로컬에 저장되었습니다: {OUTPUT_LOCAL_LAG_CORR}")
    
    # 최적 Lag 시각적 출력
    if not df_lag_results.empty:
        best_lag = df_lag_results.loc[df_lag_results['Correlation_Coefficient'].abs().idxmax()]
        print(f"\n💡 [최적 Lag 분석 결과]: 상관계수가 가장 높은 시간차는 {int(best_lag['Lag'])}개월입니다 (상관계수: {best_lag['Correlation_Coefficient']:.4f}).")


if __name__ == "__main__":
    analyze_lag_correlation()