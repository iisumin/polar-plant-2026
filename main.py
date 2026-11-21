import io
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

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
# 데이터 로딩
# ─────────────────────────────────────────────
@st.cache_data
def load_csv_bytes(file):
    """업로드된 파일에서 바이트를 꺼내 인코딩 시도."""
    if file is None:
        return None
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


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
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
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


def file_school_name(filename):
    """파일명 앞부분에서 학교명 추정."""
    if filename is None:
        return "우리 학교"
    name = os.path.basename(filename)
    name = name.rsplit(".", 1)[0]
    parts = name.replace("-", "_").split("_")
    school = ""
    for p in parts:
        if p and (any(ord(ch) > 127 for ch in p) or p.isalpha()):
            school = p
            break
    return school if school else "우리 학교"


def make_line(fig, dates, values, name, color):
    """시계열 꺾은선 트레이스 추가."""
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", name=name, line=dict(color=color)))


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

env_df_raw = load_csv_bytes(env_file)
growth_df_raw = load_csv_bytes(growth_file)

if env_df_raw is not None and not env_df_raw.empty:
    env_df = std_cols(env_df_raw)
else:
    env_df = pd.DataFrame()

if growth_df_raw is not None and not growth_df_raw.empty:
    growth_df = std_cols(growth_df_raw)
else:
    growth_df = pd.DataFrame()

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
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ═════════════════════════════════════════════
# TAB 1 — 실험 개요
# ═════════════════════════════════════════════
with tab1:
    try:
        st.header(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

        col_bg, col_purpose = st.columns([1, 2])
        with col_bg:
            st.subheader("🧪 연구 배경 및 목적")
            st.markdown(
                f"""
- 극지식물은 건조·고온·저온 등 척박한 환경에서도 살아남는 식물입니다.
- 본 대시보드는 **{school_name}**에서 재배한 극지식물 개체들의 생육 데이터를 분석해,
  개체별로 생장 차이가 얼마나 나는지, 어떤 환경 요인이 영향을 주는지 살핍니다.
- **평균**(여러 값을 더해 개수로 나눈 값), **표준편차**(값들이 평균에서 얼마나 퍼져 있는지),
  **상관계수**(두 변수가 함께 움직이는 정도, -1~1) 등을 사용해 분석합니다.
"""
            )

        with col_purpose:
            if growth_ready:
                n_individuals = growth_df["개체번호"].nunique()
                g_cols = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
                    if c in growth_df.columns
                ]
                measure_str = ", ".join(g_cols) if g_cols else "측정 항목 없음"

                if env_ready and "time" in env_df.columns:
                    dmin = env_df["time"].min()
                    dmax = env_df["time"].max()
                    if pd.notna(dmin) and pd.notna(dmax):
                        date_str_env = f"{dmin.strftime('%Y-%m-%d')} ~ {dmax.strftime('%Y-%m-%d')}"
                    else:
                        date_str_env = "측정 기간 확인 불가"
                else:
                    date_str_env = "환경 데이터 없음"

                st.subheader("📋 데이터 요약")
                summary_rows = []
                summary_rows.append(["개체 수", f"{n_individuals}개"])
                summary_rows.append(["측정 기간 (환경)", date_str_env])
                summary_rows.append(["생육 측정 항목", measure_str if measure_str else "측정 항목 없음"])
                summary_df = pd.DataFrame(summary_rows, columns=["구분", "값"])
                st.dataframe(summary_df, width="stretch", hide_index=True)

                st.divider()

                avg_temp_val = None
                avg_hum_val = None
                avg_growth_val = None

                if env_ready and "temperature" in env_df.columns:
                    temps = env_df["temperature"].dropna()
                    if len(temps) >= 1:
                        avg_temp_val = round(float(temps.mean()), 1)

                if env_ready and "humidity" in env_df.columns:
                    hums = env_df["humidity"].dropna()
                    if len(hums) >= 1:
                        avg_hum_val = round(float(hums.mean()), 1)

                if growth_ready and "생중량" in growth_df.columns:
                    gw = growth_df["생중량"].dropna()
                    if len(gw) >= 1:
                        avg_growth_val = round(float(gw.mean()), 2)

                st.subheader("📊 주요 지표")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("총 개체수", f"{n_individuals}개")
                with c2:
                    if avg_temp_val is not None:
                        st.metric("평균 온도", f"{avg_temp_val} °C")
                    else:
                        st.info("🌡️ 온도 데이터 없음")
                with c3:
                    if avg_hum_val is not None:
                        st.metric("평균 습도", f"{avg_hum_val} %")
                    else:
                        st.info("💧 습도 데이터 없음")
                with c4:
                    if avg_growth_val is not None:
                        st.metric("평균 생중량", f"{avg_growth_val} g")
                    else:
                        st.info("⚖️ 생중량 데이터 없음")
            else:
                st.info("📤 생육 결과 CSV를 업로드하면 실험 개요가 표시됩니다.")

        if not env_ready and not growth_ready:
            st.info("👈 사이드바에서 환경 CSV와 생육 결과 CSV를 각각 업로드해 주세요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# TAB 2 — 환경 데이터
# ═════════════════════════════════════════════
with tab2:
    try:
        if not env_ready:
            st.info("👈 사이드바에서 환경 CSV를 업로드해 주세요.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            env_cols_available = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if not env_cols_available:
                st.info("📊 표시할 환경 측정 항목이 없습니다.")
            else:
                st.subheader("📈 환경 요약 통계")
                summary_env_rows = []
                for col in env_cols_available:
                    vals = env_df[col].dropna()
                    if len(vals) >= 1:
                        summary_env_rows.append([
                            col,
                            round(float(vals.mean()), 2),
                            round(float(vals.min()), 2),
                            round(float(vals.max()), 2),
                        ])
                if summary_env_rows:
                    env_summary_df = pd.DataFrame(
                        summary_env_rows, columns=["항목", "평균", "최소", "최대"]
                    )
                    st.dataframe(env_summary_df, width="stretch", hide_index=True)
                else:
                    st.info("📊 계산할 환경 데이터가 부족합니다.")

                st.divider()
                st.subheader("📅 날짜별 시계열")

                ts_cols_map = {"temperature": ("온도", "#e74c3c"), "humidity": ("습도", "#3498db")}
                if "ph" in env_df.columns:
                    ts_cols_map["ph"] = ("pH", "#2ecc71")
                if "ec" in env_df.columns:
                    ts_cols_map["ec"] = ("EC", "#9b59b6")

                if "time" in env_df.columns and len(ts_cols_map) > 0:
                    env_daily = env_df.set_index("time").resample("D").mean(numeric_only=True).reset_index()

                    n_ts = sum(1 for ck in ts_cols_map if ck in env_daily.columns)
                    if n_ts >= 1:
                        fig_ts = make_subplots(rows=n_ts, cols=1, shared_xaxes=True, vertical_spacing=0.08)
                        row_idx = 1
                        for col_key, (label, color) in ts_cols_map.items():
                            if col_key in env_daily.columns:
                                vals = env_daily[col_key].dropna()
                                dates = env_daily.loc[vals.index, "time"]
                                if len(vals) >= 2:
                                    fig_ts.add_trace(
                                        go.Scatter(
                                            x=dates, y=vals, mode="lines",
                                            name=label, line=dict(color=color, width=2),
                                        ),
                                        row=row_idx, col=1,
                                    )
                                    fig_ts.update_yaxes(title_text=label, row=row_idx, col=1)
                                    row_idx += 1
                        fig_ts.update_layout(
                            title=dict(text="일별 환경 변화 (평균)", x=0.02, font=dict(size=14)),
                            xaxis_title="날짜",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                            margin=dict(l=40, r=20, t=50, b=40),
                            hovermode="x unified",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_ts, width="stretch")
                    else:
                        st.info("📈 일별 집계할 데이터가 충분하지 않습니다.")

            with st.expander("📋 환경 데이터 원본 테이블"):
                if env_ready:
                    st.dataframe(env_df, width="stretch", hide_index=True)
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name}_환경데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# TAB 3 — 생육 결과
# ═════════════════════════════════════════════
with tab3:
    try:
        if not growth_ready:
            st.info("👈 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.header(f"📊 {school_name} 생육 결과 분석")

            growth_cols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
                if c in growth_df.columns
            ]

            if not growth_cols:
                st.info("📊 표시할 생육 측정 항목이 없습니다.")
            else:
                # ── 핵심 결과 카드 ──
                st.subheader("🥇 핵심 결과")
                if "생중량" in growth_df.columns:
                    gw = growth_df["생중량"].dropna()
                    if len(gw) >= 1:
                        avg_gw = float(gw.mean())
                        max_idx = gw.idxmax()
                        min_idx = gw.idxmin()
                        max_id = growth_df.loc[max_idx, "개체번호"]
                        min_id = growth_df.loc[min_idx, "개체번호"]
                        max_val = float(growth_df.loc[max_idx, "생중량"])
                        min_val = float(growth_df.loc[min_idx, "생중량"])

                        kc1, kc2, kc3 = st.columns(3)
                        with kc1:
                            st.metric("평균 생중량", f"{round(avg_gw, 2)} g")
                        with kc2:
                            st.metric("최대 생중량 개체", f"개체 {max_id}", f"{round(max_val, 2)} g")
                        with kc3:
                            st.metric("최소 생중량 개체", f"개체 {min_id}", f"{round(min_val, 2)} g")

                st.divider()

                # ── 개체별 생육 비교 막대 그래프 ──
                st.subheader("📊 개체별 생육 비교")
                bar_metrics = [
                    c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                    if c in growth_df.columns
                ]
                if bar_metrics:
                    n_plots = len(bar_metrics)
                    fig_bar = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.06)
                    for i, metric in enumerate(bar_metrics, 1):
                        series = growth_df[["개체번호", metric]].dropna()
                        if len(series) >= 1:
                            ids = series["개체번호"].astype(str).tolist()
                            vals = series[metric].tolist()
                            is_gw = (metric == "생중량")
                            color = "#2ecc71" if is_gw else "#3498db"
                            fig_bar.add_trace(
                                go.Bar(
                                    x=ids, y=vals,
                                    name=metric,
                                    marker=dict(color=color, opacity=0.85),
                                ),
                                row=i, col=1,
                            )
                            fig_bar.update_yaxes(title_text=metric, row=i, col=1)
                    fig_bar.update_layout(
                        title=dict(text="개체별 생육 지표 비교", x=0.02, font=dict(size=14)),
                        xaxis_title="개체번호",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=50, b=50),
                        template="plotly_white",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("📊 막대 그래프로 표시할 생육 항목이 없습니다.")

                st.divider()

                # ── 생중량 분포 (박스플롯) ──
                if "생중량" in growth_df.columns:
                    gw_vals = growth_df["생중량"].dropna()
                    if len(gw_vals) >= 3:
                        st.subheader("📦 생중량 분포 (박스플롯)")
                        fig_box = go.Figure()
                        fig_box.add_trace(
                            go.Box(
                                y=gw_vals,
                                name="생중량",
                                fillcolor="#2ecc71",
                                boxpoints="outliers",
                                marker=dict(color="#2ecc71", size=6),
                            )
                        )
                        fig_box.update_layout(
                            title=dict(text="개체별 생중량 분포", x=0.02, font=dict(size=14)),
                            yaxis_title="생중량 (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=50, b=40),
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_box, width="stretch")

                        # 상위 25% / 하위 25% 비교
                        q75 = float(gw_vals.quantile(0.75))
                        q25 = float(gw_vals.quantile(0.25))
                        top25 = growth_df[growth_df["생중량"] >= q75][["개체번호", "생중량"]].dropna()
                        bot25 = growth_df[growth_df["생중량"] <= q25][["개체번호", "생중량"]].dropna()

                        if len(top25) >= 1 and len(bot25) >= 1:
                            st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")
                            comp1, comp2 = st.columns(2)
                            with comp1:
                                st.markdown(f"**🔝 상위 25% (생중량 ≥ {round(q75, 2)} g)**")
                                for _, row in top25.sort_values("생중량", ascending=False).iterrows():
                                    st.write(f"• 개체 **{row['개체번호']}** : {round(float(row['생중량']), 2)} g")
                            with comp2:
                                st.markdown(f"**🔻 하위 25% (생중량 ≤ {round(q25, 2)} g)**")
                                for _, row in bot25.sort_values("생중량").iterrows():
                                    st.write(f"• 개체 **{row['개체번호']}** : {round(float(row['생중량']), 2)} g")
                    else:
                        st.info("📦 분포를 표시할 생중량 데이터가 부족합니다.")
                else:
                    st.info("⚖️ 생중량 열이 없어 분포 분석을 건너뛰었습니다.")

                st.divider()

                # ── 상관관계 분석 ──
                if "생중량" in growth_df.columns:
                    sc_cols = [c for c in ["잎 수", "지상부 길이"] if c in growth_df.columns]
                    if sc_cols:
                        st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
                        fig_sc = make_subplots(rows=len(sc_cols), cols=1, shared_xaxes=False, vertical_spacing=0.12)
                        for i, xcol in enumerate(sc_cols, 1):
                            df_xy = growth_df[[xcol, "생중량"]].dropna()
                            if len(df_xy) >= 2:
                                x_vals = df_xy[xcol].values.astype(float)
                                y_vals = df_xy["생중량"].values.astype(float)
                                r = float(np.corrcoef(x_vals, y_vals)[0, 1])
                                r2 = round(r**2, 3)
                                fig_sc.add_trace(
                                    go.Scatter(
                                        x=x_vals, y=y_vals,
                                        mode="markers",
                                        name=f"{xcol} vs 생중량",
                                        marker=dict(color="#e74c3c", size=7, opacity=0.75, line=dict(width=1, color="white")),
                                    ),
                                    row=i, col=1,
                                )
                                # 추세선 (np.polyfit)
                                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                                x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
                                y_line = slope * x_line + intercept
                                fig_sc.add_trace(
                                    go.Scatter(
                                        x=x_line, y=y_line,
                                        mode="lines",
                                        name=f"추세선 (r²={r2})",
                                        line=dict(color="#2980b9", width=2, dash="dash"),
                                    ),
                                    row=i, col=1,
                                )
                                fig_sc.update_xaxes(title_text=xcol, row=i, col=1)
                                fig_sc.update_yaxes(title_text="생중량 (g)", row=i, col=1)

                                cc1, cc2 = st.columns([1, 3])
                                with cc1:
                                    st.metric(f"{xcol} ↔ 생중량 상관계수 (r)", f"{round(r, 3)}")
                                with cc2:
                                    if abs(r) >= 0.7:
                                        note = "→ 강한 상관관계 (두 변수가 함께 크게 움직임)"
                                    elif abs(r) >= 0.4:
                                        note = "→ 중간 정도의 상관관계"
                                    else:
                                        note = "→ 약한 상관관계 (뚜렷한 관계 없음)"
                                    st.info(f"📐 r² = {r2}  {note}")
                        fig_sc.update_layout(
                            title=dict(text="잎 수·지상부 길이 vs 생중량 상관관계", x=0.02, font=dict(size=14)),
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=50, b=40),
                            template="plotly_white",
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig_sc, width="stretch")
                    else:
                        st.info("🔗 산점도로 표시할 비교 항목이 없습니다.")
                else:
                    st.info("⚖️ 생중량 열이 없어 상관관계 분석을 건너뛰었습니다.")

            with st.expander("📋 생육 데이터 원본 테이블"):
                if growth_ready:
                    st.dataframe(growth_df, width="stretch", hide_index=True)
                    csv_bytes_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 생육 결과 CSV 다운로드",
                        data=csv_bytes_g,
                        file_name=f"{school_name}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
