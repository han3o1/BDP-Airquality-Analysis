# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 데이터 경로 설정 (로컬 파일 기준) ---
# NOTE: 이 스크립트는 모든 CSV 파일이 로컬 디스크에 있다고 가정합니다.
AIR_QUALITY_CSV = "/data/training/national_monthly_avg.csv"  # 이전에 Spark로 생성된 전국 월평균 데이터
POWER_DATA_CSV = "/data/training/data/raw/kepco_thermal_power_monthly.csv" # 화력발전 원본 데이터 (깃허브 경로)

# --- 출력 경로 설정 ---
OUTPUT_DIR = "/data/training/analysis_results_pandas"
OUTPUT_LOCAL_SUMMARY = os.path.join(OUTPUT_DIR, "regression_summary_PM10.txt")
OUTPUT_LOCAL_HEATMAP = os.path.join(OUTPUT_DIR, "correlation_heatmap_pandas.png")

# --- 분석 대상 컬럼 설정 ---
TARGET_POLLUTANT = 'national_avg_PM10'
OPTIMAL_LAG = 1 # Lagged Correlation에서 도출된 최적 시간차 (예시)

def prepare_data():
    """AirQuality와 화력발전 데이터를 Pandas로 로드 및 통합합니다."""
    
    print("=== 1. Air Quality 데이터 로드 (전국 월평균) ===")
    # Air Quality 데이터 로드 (이미 전국 월평균이 계산되었다고 가정)
    # Spark에서 생성 시 'year', 'month', 'national_avg_PM10' 등을 포함
    df_air = pd.read_csv(AIR_QUALITY_CSV)
    
    # Air Quality 데이터의 year와 month를 Integer로 변환 (필수)
    df_air['year'] = df_air['year'].astype(int)
    df_air['month'] = df_air['month'].astype(int)
    
    
    print("=== 2. 화력발전 데이터 로드 및 Unpivot (Wide -> Long) ===")
    df_power_wide = pd.read_csv(POWER_DATA_CSV)
    
    # 컬럼 정리: 첫 번째 컬럼을 'month'로 설정
    df_power_wide.rename(columns={df_power_wide.columns[0]: 'month'}, inplace=True)
    
    # 콤마 제거 및 숫자 변환
    for col_name in df_power_wide.columns[1:]:
        df_power_wide[col_name] = df_power_wide[col_name].astype(str).str.replace(',', '').astype(float)
    
    # Wide Format을 Long Format으로 변환 (Melt/Unpivot)
    df_power_long = df_power_wide.melt(
        id_vars=['month'], 
        var_name='year', 
        value_name='Power_GWh'
    )
    
    # Year 컬럼을 Integer로 변환
    df_power_long['year'] = df_power_long['year'].astype(int)
    
    
    print("=== 3. 최종 통합 (Merge) ===")
    # Air Quality와 Power 데이터를 year와 month 키로 병합
    df_merged = pd.merge(df_air, df_power_long, on=['year', 'month'], how='inner')
    
    # Date 인덱스 생성
    df_merged['Date'] = pd.to_datetime(df_merged['year'].astype(str) + '-' + df_merged['month'].astype(str) + '-01')
    df_merged.set_index('Date', inplace=True)
    df_merged.sort_index(inplace=True)
    
    # 분석에 필요한 최종 데이터프레임 반환
    return df_merged

def analyze_and_regress(df):
    """상관관계 분석, 변수 생성 및 다중 회귀 모델링을 수행합니다."""
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 4. 분석 대상 컬럼 설정
    analysis_cols = [c for c in df.columns if c.startswith('national_avg_') or c == 'Power_GWh']
    df_analysis = df[analysis_cols].astype(float)

    # 5. 상관관계 분석 (Pearson Correlation)
    print("\n=== 4. 상관관계 분석 (Pearson) ===")
    corr_matrix = df_analysis.corr(method='pearson')
    print("발전량 vs 오염물질 상관계수:\n", corr_matrix['Power_GWh'].sort_values(ascending=False))
    
    # 히트맵 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar_kws={'label': '피어슨 상관계수'})
    plt.title('전국 발전량과 대기질 간의 상관관계')
    plt.tight_layout()
    plt.savefig(OUTPUT_LOCAL_HEATMAP)
    print(f"-> 히트맵 저장 완료: {OUTPUT_LOCAL_HEATMAP}")
    
    
    # 6. 다중 회귀 모델링 (Multiple Regression)
    print("\n=== 5. 다중 회귀 모델 설정 및 적합 ===")
    
    # --- 핵심 Lag 변수 생성 ---
    df['Power_GWh_Lag1'] = df['Power_GWh'].shift(OPTIMAL_LAG)
    
    # --- 통제 변수 생성 ---
    # 계절성 통제를 위한 월(Month) 더미 변수 생성
    month_dummies = pd.get_dummies(df['month'], prefix='Month', drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    # 장기 추세(Trend) 변수 생성
    df['Trend'] = np.arange(len(df))
    
    # Lagging으로 인해 발생한 NaN 행 및 모든 NaN 행 제거
    df_regress = df.dropna()

    # 종속 변수 (Y): PM10
    Y = df_regress[TARGET_POLLUTANT]
    
    # 독립 변수 (X): Lagged Power, Trend, Month Dummies
    X_vars = ['Power_GWh_Lag1', 'Trend'] + [c for c in df_regress.columns if c.startswith('Month_')]
    
    X = df_regress[X_vars]
    X = sm.add_constant(X) # 절편(Intercept) 추가

    # OLS (Ordinary Least Squares) 모델 적합
    model = sm.OLS(Y, X).fit()

    # 7. 결과 출력 및 저장
    print("\n=== 6. 회귀 분석 결과 요약 ===")
    summary_text = model.summary().as_text()
    
    print(summary_text)

    with open(OUTPUT_LOCAL_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        
    print(f"\n✅ 최종 분석 완료. 요약 결과가 로컬에 저장되었습니다: {OUTPUT_LOCAL_SUMMARY}")

if __name__ == "__main__":
    # 데이터 준비 전제: national_monthly_avg.csv 파일 생성 필요
    # 이 스크립트 실행 전에 해당 파일이 /data/training/ 경로에 있어야 합니다.
    try:
        final_df = prepare_data()
        analyze_and_regress(final_df)
    except Exception as e:
        print(f"\n❌ 최종 실행 오류: {e}")
        print("💡 [AirQuality CSV 파일명 확인 필요] 로컬 경로와 파일명을 다시 확인해주세요.")