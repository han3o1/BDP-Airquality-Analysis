# app.py
# ----------------------------------------
# Power Generation vs Air Quality Dashboard
# - Air quality: Parquet (2003~2024, hourly → monthly → annual)
# - Power: annual_power.csv (year, Power_MWh)
# ----------------------------------------

import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from pathlib import Path

sns.set(style="whitegrid")

# =========================
# 0. 경로 / 컬럼 설정
# =========================

# ▶ app.py 가 위치한 폴더 기준 상대경로
# C:\...\Team project\app.py 에서 실행한다고 가정

BASE_DIR = Path(__file__).resolve().parent

PARQUET_BASE_DIR = BASE_DIR / "preprocessed_data"
POWER_CSV_PATH = "annual_power.csv"   # 이미 이 폴더에 있음

# parquet에 실제로 존재하는 오염물질 컬럼들 후보
POLLUTANT_CANDIDATES = ["so2_z", "co_z", "o3_z", "no2_z", "pm10_z", "pm25_z"]


# =========================
# 1. 데이터 로딩 함수
# =========================

def load_air_quality(parquet_base_dir: str):
    """
    - year_* 폴더 안의 part-r-*.parquet 파일들을 모두 읽어서
    - date_time은 아예 무시하고, year + 오염도 컬럼들만 사용
    - 파일 단위로 연도별 합계/갯수를 누적해서, 마지막에 연도별 평균 오염도를 계산

    반환:
        aq_yearly: year, pollutant... (연도별 평균)
        pollutants: 사용된 오염도 컬럼 리스트
    """

    # year_2003, year_2004, ... 아래의 part-r-*.parquet 전부 탐색
    pattern = os.path.join(parquet_base_dir, "year_*", "part-r-*.parquet")
    files = glob.glob(pattern)

    if not files:
        st.error(f"Parquet 파일을 찾을 수 없습니다:\n{os.path.abspath(pattern)}")
        st.stop()

    # 먼저 첫 파일로부터 실제 컬럼 목록을 확인
    sample = pd.read_parquet(files[0])
    available_cols = sample.columns.tolist()

    # 실제 존재하는 오염도 컬럼만 선택
    pollutants = [c for c in POLLUTANT_CANDIDATES if c in available_cols]
    if "year" not in available_cols:
        st.error(f"parquet 데이터에 'year' 컬럼이 없습니다. 현재 컬럼들: {available_cols}")
        st.stop()
    if not pollutants:
        st.error("오염도(z-score) 컬럼을 찾을 수 없습니다.\n"
                 f"현재 컬럼들: {available_cols}")
        st.stop()

    use_cols = ["year"] + pollutants

    sum_df = None   # 연도별 오염도 합계
    cnt_df = None   # 연도별 오염도 개수(유효값 count)

    for f in files:
        # 필요한 컬럼만 읽어서 메모리 절약
        df = pd.read_parquet(f, columns=use_cols)

        df["year"] = df["year"].astype(int)

        grp = df.groupby("year")[pollutants]

        sums = grp.sum()    # 연도별 합계
        counts = grp.count()  # 연도별 유효값 개수

        if sum_df is None:
            sum_df = sums
            cnt_df = counts
        else:
            # 연도별 합계/카운트 누적
            sum_df = sum_df.add(sums, fill_value=0)
            cnt_df = cnt_df.add(counts, fill_value=0)

    # 최종 연도별 평균 = 총합 / 총개수
    mean_df = sum_df / cnt_df
    mean_df = mean_df.reset_index()   # index에 있던 year를 컬럼으로

    aq_yearly = mean_df  # year + pollutants

    return aq_yearly, pollutants





def load_power(csv_path: str):
    """
    annual_power.csv를 읽어서 year, Power_MWh 형태로 맞춰줌.

    현재 CSV 구조:
        MWh,2003,2004,...,2024
        sum,140269476,...

    → 이를
        year,Power_MWh
        2003,140269476
        ...
    로 변환해 준다.
    """

    if not os.path.exists(csv_path):
        st.error(f"연간 발전량 CSV 파일을 찾을 수 없습니다:\n{os.path.abspath(csv_path)}")
        st.stop()

    df = pd.read_csv(csv_path)

    # 이미 year / Power_MWh 형식인 경우 (혹시 나중에 바꾸면)
    if "year" in df.columns and "Power_MWh" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df["Power_MWh"] = pd.to_numeric(df["Power_MWh"], errors="coerce")
        return df[["year", "Power_MWh"]]

    # 지금처럼 MWh / 2003 / 2004 / ... 형태인 경우
    if "MWh" in df.columns:
        row = df.iloc[0]  # sum 행 하나
        data = []
        for col in df.columns:
            if col == "MWh":
                continue
            try:
                year = int(col)
            except ValueError:
                continue
            power = float(row[col])
            data.append({"year": year, "Power_MWh": power})

        out = pd.DataFrame(data)
        out["year"] = out["year"].astype(int)
        return out.sort_values("year").reset_index(drop=True)

    # 그 외 예상 못한 형태
    st.error(
        "annual_power.csv 형식을 해석할 수 없습니다.\n"
        f"현재 컬럼들: {list(df.columns)}"
    )
    st.stop()




def build_annual(aq_yearly: pd.DataFrame, pollutants: list[str], power_annual: pd.DataFrame):
    # aq_yearly는 이미 year별 평균 오염도
    annual = aq_yearly.merge(power_annual, on="year", how="inner")
    return annual




# =========================
# 2. Streamlit UI
# =========================

st.title("Power Generation vs Air Quality (2003–2024)")
st.write(
    "Spark로 집계한 대기질 Parquet 데이터와 연간 전력 통계(annual_power.csv)를 "
    "연결한 시각화 대시보드입니다."
)

# 실제 데이터 로딩
aq_yearly, pollutants = load_air_quality(PARQUET_BASE_DIR)
power_annual = load_power(POWER_CSV_PATH)
annual = build_annual(aq_yearly, pollutants, power_annual)

min_year = int(annual["year"].min())
max_year = int(annual["year"].max())

with st.sidebar:
    st.header("Settings")
    year_range = st.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
    show_raw = st.checkbox("Show raw tables", value=False)

# 연도 필터 적용
# 연도 필터 적용
y1, y2 = year_range
aq_yearly_f = aq_yearly[(aq_yearly["year"] >= y1) & (aq_yearly["year"] <= y2)].copy()
annual_f = annual[(annual["year"] >= y1) & (annual["year"] <= y2)].copy()

# 보기 좋은 이름
def pretty(col: str) -> str:
    return col.replace("_z", "").upper()

# Raw table 보기
if show_raw:
    st.subheader("Annual Air Quality (연도별 평균, from Parquet)")
    st.dataframe(aq_yearly_f.sort_values("year"))

    st.subheader("Annual Power Generation (연간 발전량)")
    st.dataframe(power_annual)

    st.subheader("Annual Merged Data (연간 오염도 + 발전량)")
    st.dataframe(annual_f)



# =========================
# 4. 그래프 1: 회귀선 산점도
# =========================

st.header("Power vs Individual Pollutants (Regression)")

if annual_f["Power_MWh"].isna().all():
    st.warning("Power_MWh 값이 모두 NaN입니다. annual_power.csv를 확인해주세요.")
else:
    n = len(pollutants)
    cols = 3
    rows = math.ceil(n / cols)

    fig1, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, p in enumerate(pollutants):
        ax = axes[i]
        sns.regplot(data=annual_f, x="Power_MWh", y=p, ax=ax)
        ax.set_title(f"Power vs {pretty(p)}")
        ax.set_xlabel("Power Generation (MWh)")
        ax.set_ylabel("Concentration (z-score)")

    # 남는 subplot은 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig1)


# =========================
# 5. 그래프 2: 상관관계 히트맵
# =========================

st.header("Correlation Heatmap")

corr_cols = ["Power_MWh"] + pollutants
corr_cols = [c for c in corr_cols if c in annual_f.columns]

if len(corr_cols) >= 2:
    corr = annual_f[corr_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", vmin=-1, vmax=1, ax=ax2)
    ax2.set_title("Correlation between Power and Air Quality Indicators")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("상관관계를 계산할 수 있는 컬럼 수가 부족합니다.")


# =========================
# 6. 그래프 3: 연도별 오염도 트렌드
# =========================

st.header("Trend of Air Pollutants over Time")

fig3, ax3 = plt.subplots(figsize=(10, 5))
for p in pollutants:
    ax3.plot(annual_f["year"], annual_f[p], marker="o", label=pretty(p))

ax3.set_xlabel("Year")
ax3.set_ylabel("Average z-score")
ax3.set_title("Trend of Air Pollutants over Time")
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)


# =========================
# 7. 그래프 4: 종합 대기질 지수 vs 발전량
# =========================

st.header("Combined Relationship: Power vs Overall Air Quality")

# AirQualityIndex는 PM10, NO2, SO2 z-score 평균으로 예시
aqi_cols = [c for c in ["pm10_z", "no2_z", "so2_z"] if c in pollutants]
if aqi_cols:
    annual_f["AirQualityIndex"] = annual_f[aqi_cols].mean(axis=1)

    if annual_f["Power_MWh"].isna().all():
        st.warning("발전량 데이터가 없어서 종합 회귀선을 그릴 수 없습니다.")
    else:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.regplot(data=annual_f, x="Power_MWh", y="AirQualityIndex", ax=ax4)
        ax4.set_xlabel("Power Generation (MWh)")
        ax4.set_ylabel("Air Quality Index (mean z-score)")
        ax4.set_title("Power vs Overall Air Quality")
        plt.tight_layout()
        st.pyplot(fig4)
else:
    st.info("pm10_z, no2_z, so2_z 중 최소 하나 이상이 있어야 AirQualityIndex를 계산할 수 있습니다.")

st.markdown("---")
st.caption(
    "※ app.py를 Team project 폴더에 두고 실행하는 것을 기준으로 경로를 설정했습니다.\n"
    "   경로가 다르다면 PARQUET_BASE_DIR, POWER_CSV_PATH만 수정하면 됩니다."
)

