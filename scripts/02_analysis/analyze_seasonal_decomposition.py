# -*- coding: utf-8 -*-

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose # 계절성 분해 라이브러리
import matplotlib.pyplot as plt
import os
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

# 출력: 계절성 분해 분석 결과 저장 경로
OUTPUT_LOCAL_DECOMPOSE_PM10 = os.path.join(OUTPUT_DIR, "seasonal_decomposition_pm10.png")
OUTPUT_LOCAL_DECOMPOSE_POWER = os.path.join(OUTPUT_DIR, "seasonal_decomposition_power.png")

# --- 분석 대상 설정 ---
TARGET_POLLUTANT = 'national_avg_PM10'   # 종속 변수
TARGET_POWER = 'Power_GWh'


def analyze_decomposition():
    
    # 1. 통합 데이터 로드
    print("=== 1. 통합 CSV 데이터 로드 및 시계열 준비 시작 ===")
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
    df_analysis = df_pandas[[TARGET_POLLUTANT, TARGET_POWER]].astype(float)
    
    # 3. PM10 농도 시계열 분해 (Additive Model 사용)
    print("\n=== 2. PM10 농도 시계열 분해 시작 ===")
    # Additive 모델: 관측치 = 추세 + 계절 + 잔차 (계절 변동 폭이 일정하다고 가정)
    # period=12: 월별 데이터이므로 주기는 12개월
    # NaN 값이 있으면 분해에 실패하므로, dropna()를 사용하여 유효한 데이터만 사용합니다.
    result_pm10 = seasonal_decompose(df_analysis[TARGET_POLLUTANT].dropna(), model='additive', period=12)
    
    # 4. 화력 발전량 시계열 분해
    print("=== 3. 화력 발전량 시계열 분해 시작 ===")
    result_power = seasonal_decompose(df_analysis[TARGET_POWER].dropna(), model='additive', period=12)
    
    # 5. 시각화 1: PM10 분해 결과
    print(f"\n=== 4. 시각화: PM10 분해 결과 저장 ({OUTPUT_LOCAL_DECOMPOSE_PM10}) ===")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(result_pm10.observed, label='관측 데이터 (Observed)')
    axes[0].set_title('전국 평균 PM10 농도 분해 결과')
    axes[0].legend()
    
    axes[1].plot(result_pm10.trend, label='장기 추세 (Trend)', color='red')
    axes[1].legend()
    
    axes[2].plot(result_pm10.seasonal, label='계절 요인 (Seasonal)', color='green')
    axes[2].legend()
    
    axes[3].plot(result_pm10.resid, label='잔차 (Residual)', color='gray')
    axes[3].legend()
    
    plt.xlabel("날짜")
    plt.tight_layout()
    plt.savefig(OUTPUT_LOCAL_DECOMPOSE_PM10)
    
    # 6. 시각화 2: 발전량 분해 결과
    print(f"=== 5. 시각화: 발전량 분해 결과 저장 ({OUTPUT_LOCAL_DECOMPOSE_POWER}) ===")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(result_power.observed, label='관측 데이터 (Observed)')
    axes[0].set_title('전국 화력발전량 분해 결과')
    axes[0].legend()
    
    axes[1].plot(result_power.trend, label='장기 추세 (Trend)', color='red')
    axes[1].legend()
    
    axes[2].plot(result_power.seasonal, label='계절 요인 (Seasonal)', color='green')
    axes[2].legend()
    
    axes[3].plot(result_power.resid, label='잔차 (Residual)', color='gray')
    axes[3].legend()
    
    plt.xlabel("날짜")
    plt.tight_layout()
    plt.savefig(OUTPUT_LOCAL_DECOMPOSE_POWER)
    
    print("\n✅ 계절성 분해 분석 완료. 결과 이미지 파일이 로컬에 저장되었습니다.")
    
def main():
    # Matplotlib이 한글을 지원하지 않을 수 있으므로, 폰트 설정을 추가하여 깨짐 방지
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 환경 대체
        
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
    
    analyze_decomposition()

if __name__ == "__main__":
    main()