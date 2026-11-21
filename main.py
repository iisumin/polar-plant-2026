import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="극지식물 생육 대시보드")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# 데이터 로딩 (캐싱 + 인코딩 이중 시도)
# ─────────────────────────────────────────────
@st.cache_data
def load_csv_bytes(file):
    """업로드된 파일에서 바이트를 꺼내 utf-8-sig → cp949 순서로 읽기."""
    if file is None:
        return None
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지)
# ─────────────────────────────────────────────
def std_cols(df):
    """열 이름을 표준 이름으로 통일하고 숫자형으로 변환."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ─────────────────────────────────────────────
# 학교명 추출 (파일명 앞부분)
# ─────────────────────────────────────────────
def file_school_name(filename):
    if filename is None:
        return "우리 학교"
    name = os.path.basename(filename)
    name = name.rsplit(".", 1)[0]
    parts = name.replace("-", "_").split("_")
    for p in parts:
        if p and (any(ord(ch) > 127 for ch in p) or p.isalpha()):
            return p
    return "우리 학교"


# ─────────────────────────────────────────────
# CSS 줄기선 helper
# ─────────────────────────────────────────────
def add_trendline(fig, x_vals, y_vals, color="#2980b9"):
    """np.polyfit으로 기울기와 절편을 구해 추세선 추가."""
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"추세선",
            line=dict(color=color, width=2, dash="dash"),
        ),
    )


# ─────────────────────────────────────────────
# 사이드바 — CSV 업로드
# ─────────────────────────────────────────────
st.sidebar.title("📁 데이터 업로드")

with st.sidebar:
    with st.spinner("CSV를 로딩 중…"):
        env_file = st.file_uploader("환경 CSV 업로드", type="csv")
        growth_file = st.file_uploader("생육 결과 CSV 업로드", type="csv")

        env_df_raw = load_csv_bytes(env_file)
        growth_df_raw = load_csv_bytes(growth_file)

    env_df = std_cols(env_df_raw) if (env_df_raw is not None and not env_df_raw.empty) else pd.DataFrame()
    growth_df = std_cols(growth_df_raw) if (growth_df_raw is not None and not growth_df_raw.empty) else pd.DataFrame()

school_name = file_school_name(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

env_ready = not env_df.empty
growth_ready = not growth_df.empty

st.sidebar.divider()
st.sidebar.info("📌 업로드한 CSV를 바탕으로 분석이 자동 표시됩니다.")

if env_ready or growth_ready:
    st.sidebar.success(f"🏫 학교명: **{school_name}**")

# ─────────────────────────────────────────────
# 생육 측정 항목 선택 (사이드바)
# ─────────────────────────────────────────────
all_growth_cols = [
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "생중량",
    "지상부 질량",
    "지하부 질량",
]
available_growth_cols = [c for c in all_growth_cols if c in growth_df.columns] if growth_ready else []
selected_growth_cols = st.sidebar.multiselect(
    "📊 생육 측정 항목 선택 (Tab 3 사용)",
    options=available_growth_cols,
    default=available_growth_cols,
    help="Tab 3 상관관계·예측에서 사용할 항목입니다.",
)

st.sidebar.divider()
st.sidebar.caption(
    "💡 통계 용어 안내\n"
    "• 평균: 여러 값을 더해 개수로 나눈 대표값\n"
    "• 표준편차: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수치\n"
    "• 변동계수(CV): 표준편차를 평균으로 나눈 값(백분율) — 서로 다른 단위의 산포를 비교\n"
    "• 상관계수(r): 두 변수가 함께 움직이는 정도(-1 ~ 1)\n"
    "• 결정계수(R²): 회귀식이 데이터를 얼마나 잘 설명하는지(0 ~ 1)"
)


# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬 연구 개요 및 스마트팜 환경",
    "📊 개체별 생육 차이 분석",
    "🔗 생육 지표와 생중량의 상관관계 및 예측",
])


# ═════════════════════════════════════════════
# TAB 1 — 연구 개요 및 스마트팜 환경
# ═════════════════════════════════════════════
with tab1:
    try:
        st.title(f"인천여고와 극지연구소의 콜라보레이션! 🌱")
        st.markdown(f"**{school_name}** 극지연구실 | 극지식물 개체별 생육 차이 분석")

        st.subheader("📖 연구 목적 및 가설")

        col_purpose, col_hypothesis = st.columns([1, 1])
        with col_purpose:
            st.markdown(
                """
- **배경**: 극지식물은 건조·고온·저온 등 척박한 환경에서도 살아남는 식물입니다.
  인천여고와 극지연구소가 함께 극지식물을 스마트팜에서 재배하며,
  개체마다 생장 속도에 차이가 나는 원인을 찾고 있습니다.
- **목적**: 같은 환경에서 자란 극지식물 개체들의
  잎 수, 지상부 길이, 지하부 길이, 생중량을 비교해
  **개체 간 생육 차이**가 얼마나 큰지 확인하고,
  생육 지표와 생중량 사이의 관계를 정량적으로 분석합니다.
- **방법**: 스마트팜 환경 데이터(온도·습도)와
  개체별 생육 측정 데이터를 나란히 분석해
  평균·표준편차·상관계수 등의 기초 통계로 차이를 설명합니다.
"""
            )
        with col_hypothesis:
            st.markdown(
                """
**🔬 연구 가설**
1. 잎 수가 많은 개체일수록 생중량이 더 클 것이다.
2. 지상부 길이가 긴 개체일수록 생중량이 더 클 것이다.
3. 지하부 길이도 생중량과 관련이 있을 것이다.
4. 같은 스마트팜 안에서도 개체별 생장 차이는 뚜렷하게 나타날 것이다.

> 💡 **상관**(두 값이 함께 움직이는 경향)이 있다고 해서
> 반드시 **인과**(한 값이 다른 값을 직접 일으킴)라고 할 수는 없습니다.
> 상관계수는 관계 방향과 강도를 알려줄 뿐, 원인을 증명하지는 않습니다.
"""
            )

        st.divider()

        # ── 데이터 요약 ──
        if growth_ready:
            n_individuals = growth_df["개체번호"].nunique()

            if env_ready and "time" in env_df.columns:
                dmin = env_df["time"].min()
                dmax = env_df["time"].max()
                if pd.notna(dmin) and pd.notna(dmax):
                    date_range = f"{dmin.strftime('%Y-%m-%d')} ~ {dmax.strftime('%Y-%m-%d')}"
                else:
                    date_range = "측정 기간 확인 불가"
            else:
                date_range = "환경 데이터 없음"

            st.subheader("📋 데이터 요약")
            summary_rows = []
            summary_rows.append(["개체 수", f"{n_individuals}개"])
            summary_rows.append(["측정 기간 (환경 CSV)", date_range])
            growth_measures = [c for c in all_growth_cols if c in growth_df.columns]
            summary_rows.append(["생육 측정 항목", ", ".join(growth_measures) if growth_measures else "측정 항목 없음"])
            st.dataframe(
                pd.DataFrame(summary_rows, columns=["구분", "값"]),
                width="stretch",
                hide_index=True,
            )

            st.divider()

            # ── 주요 지표 카드 ──
            st.subheader("📊 주요 지표 카드")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("총 개체수", f"{n_individuals}개")
            with c2:
                if env_ready and "temperature" in env_df.columns:
                    temps = env_df["temperature"].dropna()
                    if len(temps) >= 1:
                        st.metric("평균 온도", f"{round(float(temps.mean()), 1)} °C")
                    else:
                        st.info("🌡️ 온도 데이터가 부족합니다.")
                else:
                    st.info("🌡️ 온도 열이 없습니다.")
            with c3:
                if env_ready and "humidity" in env_df.columns:
                    hums = env_df["humidity"].dropna()
                    if len(hums) >= 1:
                        st.metric("평균 습도", f"{round(float(hums.mean()), 1)} %")
                    else:
                        st.info("💧 습도 데이터가 부족합니다.")
                else:
                    st.info("💧 습도 열이 없습니다.")
            with c4:
                if "생중량" in growth_df.columns:
                    gw = growth_df["생중량"].dropna()
                    if len(gw) >= 1:
                        st.metric("평균 생중량", f"{round(float(gw.mean()), 2)} g")
                    else:
                        st.info("⚖️ 생중량 데이터가 부족합니다.")
                else:
                    st.info("⚖️ 생중량 열이 없습니다.")

        if not env_ready and not growth_ready:
            st.info("👈 사이드바에서 환경 CSV와 생육 결과 CSV를 각각 업로드해 주세요.")

        st.divider()

        # ── 날짜별 온도·습도 꺾은선 ──
        if env_ready and "time" in env_df.columns:
            ts_cols = [c for c in ["temperature", "humidity"] if c in env_df.columns]
            if ts_cols:
                st.subheader("🌡️ 날짜별 환경 변화 (일별 평균)")
                env_daily = (
                    env_df.set_index("time")
                    .resample("D")
                    .mean(numeric_only=True)
                    .reset_index()
                )
                n_ts = len(ts_cols)
                fig_ts = make_subplots(rows=n_ts, cols=1, shared_xaxes=True, vertical_spacing=0.08)
                row_idx = 1
                color_map = {"temperature": ("온도 (°C)", "#e74c3c"), "humidity": ("습도 (%)", "#3498db")}
                for col in ts_cols:
                    if col in color_map and col in env_daily.columns:
                        vals = env_daily[col].dropna()
                        dates = env_daily.loc[vals.index, "time"]
                        if len(vals) >= 2:
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=dates,
                                    y=vals,
                                    mode="lines",
                                    name=color_map[col][0],
                                    line=dict(color=color_map[col][1], width=2.5),
                                ),
                                row=row_idx,
                                col=1,
                            )
                            fig_ts.update_yaxes(title_text=color_map[col][0], row=row_idx, col=1)
                            row_idx += 1
                if row_idx > 1:
                    fig_ts.update_layout(
                        title=dict(text="일별 스마트팜 환경 변화 (평균)", x=0.02, font=dict(size=14)),
                        xaxis_title="날짜",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                        margin=dict(l=40, r=20, t=55, b=40),
                        hovermode="x unified",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_ts, width="stretch")
            else:
                st.info("🌡️ 표시할 환경 측정 항목이 없습니다.")

        with st.expander("📋 환경 데이터 원본 테이블"):
            if env_ready:
                st.dataframe(env_df, width="stretch", hide_index=True)
                csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드",
                    data=csv_env,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
            else:
                st.info("데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# TAB 2 — 개체별 생육 차이 분석
# ═════════════════════════════════════════════
with tab2:
    try:
        if not growth_ready:
            st.info("👈 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.title("📊 개체별 생육 차이 분석")

            # ── ① 생중량 분포 히스토그램 ──
            if "생중량" in growth_df.columns:
                gw_data = growth_df["생중량"].dropna()
                if len(gw_data) >= 5:
                    st.subheader("📦 개체별 생중량 분포 (히스토그램)")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(
                        go.Histogram(
                            x=gw_data,
                            nbinsx=max(8, int(np.sqrt(len(gw_data)))),
                            marker=dict(color="#2ecc71", opacity=0.75, line=dict(color="white", width=1)),
                            name="생중량",
                        )
                    )
                    mean_gw = float(gw_data.mean())
                    fig_hist.add_vline(
                        x=mean_gw,
                        line_color="#e74c3c",
                        line_width=2,
                        line_dash="dash",
                        annotation_text=f"평균 = {round(mean_gw, 2)} g",
                        annotation_position="top right",
                    )
                    fig_hist.update_layout(
                        title=dict(text="개체별 생중량 분포 ( Histogram )", x=0.02, font=dict(size=14)),
                        xaxis_title="생중량 (g)",
                        yaxis_title="개체 수",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=50, b=40),
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_hist, width="stretch")
                else:
                    st.info("📦 분포를 표시할 생중량 데이터가 부족합니다.")
            else:
                st.info("⚖️ 생중량 열이 없어 히스토그램을 표시할 수 없습니다.")

            st.divider()

            # ── ② 잎 수·생중량의 평균·표준편차·CV ──
            st.subheader("📐 평균·표준편차·변동계수 (CV)")
            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                stat_rows = []
                for col in ["잎 수", "생중량"]:
                    vals = growth_df[col].dropna()
                    if len(vals) >= 2:
                        m = float(vals.mean())
                        s = float(vals.std(ddof=1))
                        cv = (s / m * 100) if m != 0 else float("nan")
                        stat_rows.append([col, f"{m:.2f}", f"{s:.2f}", f"{cv:.1f}%"])
                if stat_rows:
                    st.dataframe(
                        pd.DataFrame(stat_rows, columns=["측정 항목", "평균", "표준편차 (s)", "변동계수 (CV %)"]),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "📌 표준편차(s): 각 값이 평균에서 얼마나 떨어져 있는지 나타내는 산포 지표입니다.\n"
                        "📌 변동계수(CV): (표준편차 ÷ 평균) × 100. 서로 다른 단위를 가진 값의 상대적 산포를 비교할 때 사용합니다."
                    )
                else:
                    st.info("계산할 데이터가 부족합니다.")
            else:
                st.info("📐 잎 수 또는 생중량 열이 없습니다.")

            st.divider()

            # ── ③ 생중량 상위 25% vs 하위 25% 비교 ──
            if "생중량" in growth_df.columns:
                gw_all = growth_df["생중량"].dropna()
                if len(gw_all) >= 4:
                    q25 = float(gw_all.quantile(0.25))
                    q75 = float(gw_all.quantile(0.75))

                    top25_df = growth_df[growth_df["생중량"] >= q75].dropna(subset=["생중량"])
                    bot25_df = growth_df[growth_df["생중량"] <= q25].dropna(subset=["생중량"])

                    st.subheader(f"🏆 생중량 상위 25% ( ≥ {round(q75, 2)} g ) vs 하위 25% ( ≤ {round(q25, 2)} g )")

                    comp1, comp2 = st.columns(2)
                    with comp1:
                        st.markdown(f"**🔝 상위 25% ({len(top25_df)}개체)**")
                        for _, row in top25_df.sort_values("생중량", ascending=False).iterrows():
                            st.write(f"• 개체 **{row['개체번호']}** : 생중량 {round(float(row['생중량']), 2)} g")
                        if "잎 수" in top25_df.columns:
                            st.write(f"   평균 잎 수: {round(float(top25_df['잎 수'].mean()), 1)}장")
                        if "지상부 길이" in top25_df.columns:
                            st.write(f"   평균 지상부 길이: {round(float(top25_df['지상부 길이'].mean()), 1)} mm")
                    with comp2:
                        st.markdown(f"**🔻 하위 25% ({len(bot25_df)}개체)**")
                        for _, row in bot25_df.sort_values("생중량").iterrows():
                            st.write(f"• 개체 **{row['개체번호']}** : 생중량 {round(float(row['생중량']), 2)} g")
                        if "잎 수" in bot25_df.columns:
                            st.write(f"   평균 잎 수: {round(float(bot25_df['잎 수'].mean()), 1)}장")
                        if "지상부 길이" in bot25_df.columns:
                            st.write(f"   평균 지상부 길이: {round(float(bot25_df['지상부 길이'].mean()), 1)} mm")

                    # ── 그룹별 막대 그래프 ──
                    bar_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                    if bar_cols:
                        st.divider()
                        st.subheader("📊 상위 25% vs 하위 25% 생육 지표 비교")
                        n_plots = len(bar_cols)
                        fig_bar = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.07)
                        row_idx = 1
                        for col in bar_cols:
                            t_vals = top25_df[col].dropna()
                            b_vals = bot25_df[col].dropna()
                            if len(t_vals) >= 1 and len(b_vals) >= 1:
                                fig_bar.add_trace(
                                    go.Bar(
                                        x=[f"상위 25% ({len(t_vals)}개체)"],
                                        y=[round(float(t_vals.mean()), 2)],
                                        name=f"{col} (상위)",
                                        marker=dict(color="#2ecc71", opacity=0.85),
                                    ),
                                    row=row_idx,
                                    col=1,
                                )
                                fig_bar.add_trace(
                                    go.Bar(
                                        x=[f"하위 25% ({len(b_vals)}개체)"],
                                        y=[round(float(b_vals.mean()), 2)],
                                        name=f"{col} (하위)",
                                        marker=dict(color="#e74c3c", opacity=0.85),
                                    ),
                                    row=row_idx,
                                    col=1,
                                )
                                fig_bar.update_yaxes(title_text=col, row=row_idx, col=1)
                                row_idx += 1
                        fig_bar.update_layout(
                            title=dict(text="생중량 상·하위 25% 집단 평균 생육 지표 비교", x=0.02, font=dict(size=14)),
                            xaxis_title="집단",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            barmode="group",
                            margin=dict(l=40, r=20, t=55, b=50),
                            template="plotly_white",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_bar, width="stretch")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# TAB 3 — 상관관계 및 예측
# ═════════════════════════════════════════════
with tab3:
    try:
        if not growth_ready:
            st.info("👈 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.title("🔗 생육 지표와 생중량의 상관관계 및 예측")

            st.subheader("🔍 상관계수 비교")
            if "생중량" in growth_df.columns:
                r_rows = []
                x_candidates = [c for c in selected_growth_cols if c != "생중량" and c in growth_df.columns]
                for xcol in x_candidates:
                    df_xy = growth_df[[xcol, "생중량"]].dropna()
                    if len(df_xy) >= 2:
                        r = float(np.corrcoef(df_xy[xcol].values.astype(float), df_xy["생중량"].values.astype(float))[0, 1])
                        r2 = round(r**2, 3)
                        if abs(r) >= 0.7:
                            strength = "강한 상관관계"
                        elif abs(r) >= 0.4:
                            strength = "중간 상관관계"
                        else:
                            strength = "약한 상관관계"
                        r_rows.append([xcol, f"{r:.3f}", f"{r2:.3f}", strength])

                if r_rows:
                    st.dataframe(
                        pd.DataFrame(r_rows, columns=["비교 지표 X", "상관계수 (r)", "결정계수 (R²)", "상관 강도"]),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "📌 상관계수 r: -1에 가까우면 음의 상관(X↑ Y↓), +1에 가까우면 양의 상관(X↑ Y↑).\n"
                        "📌 결정계수 R² = r²: 회귀식이 데이터를 설명하는 비율(0~1)."
                    )
                else:
                    st.info("상관계수를 계산할 데이터가 충분하지 않습니다.")

                st.divider()

                # ── 산점도 + 추세선 ──
                st.subheader("📈 산점도 및 회귀 추세선")
                sc_cols_to_plot = [c for c in selected_growth_cols if c != "생중량" and c in growth_df.columns]
                if sc_cols_to_plot:
                    n_sc = len(sc_cols_to_plot)
                    fig_sc = make_subplots(rows=n_sc, cols=1, shared_xaxes=False, vertical_spacing=0.12)
                    for i, xcol in enumerate(sc_cols_to_plot, 1):
                        df_xy = growth_df[[xcol, "생중량"]].dropna()
                        if len(df_xy) >= 2:
                            x_vals = df_xy[xcol].values.astype(float)
                            y_vals = df_xy["생중량"].values.astype(float)
                            r = float(np.corrcoef(x_vals, y_vals)[0, 1])
                            r2 = round(r**2, 3)
                            fig_sc.add_trace(
                                go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode="markers",
                                    name=f"{xcol}",
                                    marker=dict(color="#e74c3c", size=7, opacity=0.7, line=dict(width=1, color="white")),
                                ),
                                row=i,
                                col=1,
                            )
                            add_trendline(fig_sc, x_vals, y_vals)
                            fig_sc.update_xaxes(title_text=xcol, row=i, col=1)
                            fig_sc.update_yaxes(title_text="생중량 (g)", row=i, col=1)
                            with st.expander(f"📐 {xcol} ↔ 생중량 상세 (r = {r:.3f}, R² = {r2})"):
                                st.write(f"• 상관계수 r = {r:.3f}")
                                st.write(f"• 결정계수 R² = {r2}")
                                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                                direction = "증가" if slope > 0 else "감소"
                                st.write(f"• 추세 방향: {xcol}이 커질수록 생중량이 {direction}")
                                st.write(
                                    f"• 추세선의 기울기: {slope:.4f} ( {xcol} 1단위 증가 시 생중량 약 {abs(slope):.3f}g 변화 )"
                                )
                    if n_sc > 0:
                        fig_sc.update_layout(
                            title=dict(text="생육 지표별 생중량 산점도 및 추세선", x=0.02, font=dict(size=14)),
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=55, b=40),
                            template="plotly_white",
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig_sc, width="stretch")
                else:
                    st.info("📈 산점도로 표시할 항목이 선택되지 않았습니다.")

                st.divider()

                # ── 잎 수 → 생중량 예측 회귀식 ──
                if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                    st.subheader("🔮 잎 수를 이용한 생중량 예측 (단순 선형 회귀)")
                    df_pred = growth_df[["잎 수", "생중량"]].dropna()
                    if len(df_pred) >= 2:
                        x_pred = df_pred["잎 수"].values.astype(float)
                        y_pred = df_pred["생중량"].values.astype(float)
                        slope, intercept = np.polyfit(x_pred, y_pred, 1)
                        y_pred_line = slope * x_pred + intercept
                        ss_res = float(np.sum((y_pred - y_pred_line) ** 2))
                        ss_tot = float(np.sum((y_pred - float(y_pred.mean())) ** 2))
                        r2_pred = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                        r2_simple = round(float(np.corrcoef(x_pred, y_pred)[0, 1]) ** 2, 3)

                        # 회귀식을 평범한 텍스트로 표시
                        pred1, pred2, pred3 = st.columns(3)
                        with pred1:
                            st.metric("기울기 (slope)", f"{slope:.4f} g/잎")
                        with pred2:
                            st.metric("절편 (intercept)", f"{intercept:.4f} g")
                        with pred3:
                            st.metric("결정계수 (R²)", f"{r2_simple:.3f}")

                        # 예측 입력
                        st.divider()
                        user_leaves = st.number_input(
                            "🔎 예측할 잎 수 (장)를 입력하세요",
                            min_value=0.0,
                            max_value=200.0,
                            value=30.0,
                            step=1.0,
                        )
                        pred_gw = slope * float(user_leaves) + intercept
                        st.info(
                            f"🌿 잎 수 **{user_leaves:.0f}장**일 때 예측 생중량: **{max(0, pred_gw):.2f} g**\n"
                            f"(회귀식: 생중량 = {slope:.4f} × 잎 수 + {intercept:.4f}, R² = {r2_simple:.3f})"
                        )

                        # 회귀선 시각화 (기존 산점도 위에)
                        fig_pred = go.Figure()
                        fig_pred.add_trace(
                            go.Scatter(
                                x=x_pred,
                                y=y_pred,
                                mode="markers",
                                name="실제 데이터",
                                marker=dict(color="#e74c3c", size=7, opacity=0.7, line=dict(width=1, color="white")),
                            )
                        )
                        x_range = np.linspace(float(x_pred.min()), float(x_pred.max()), 100)
                        y_range = slope * x_range + intercept
                        fig_pred.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_range,
                                mode="lines",
                                name=f"회귀선 (R²={r2_simple:.3f})",
                                line=dict(color="#2980b9", width=2.5),
                            )
                        )
                        fig_pred.add_vline(
                            x=float(user_leaves),
                            line_color="#2ecc71",
                            line_width=2,
                            line_dash="dot",
                            annotation_text=f"예측: {max(0, pred_gw):.2f} g",
                            annotation_position="top right",
                        )
                        fig_pred.update_layout(
                            title=dict(text=f"잎 수 → 생중량 예측 회귀 (R² = {r2_simple:.3f})", x=0.02, font=dict(size=14)),
                            xaxis_title="잎 수 (장)",
                            yaxis_title="생중량 (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=55, b=40),
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_pred, width="stretch")
                    else:
                        st.info("📐 잎 수와 생중량 데이터가 충분하지 않아 회귀식을 구할 수 없습니다.")
                else:
                    st.info("📐 잎 수 또는 생중량 열이 없어 회귀 예측을 표시할 수 없습니다.")

            else:
                st.info("⚖️ 생중량 열이 없어 상관관계·예측 분석을 할 수 없습니다.")

            with st.expander("📋 생육 데이터 원본 테이블"):
                if growth_ready:
                    st.dataframe(growth_df, width="stretch", hide_index=True)
                    csv_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 생육 결과 CSV 다운로드",
                        data=csv_g,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
