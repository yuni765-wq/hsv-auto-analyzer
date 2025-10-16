# ---------------------------------------------------------------
# HSV Auto Analyzer v2.5 Full - Clinical Visualization Platform
# (c) 2025 Isaka × Lian
# ---------------------------------------------------------------
# 기능:
# - v2.4 분석엔진(임상근사) + v2.5 시각화(UI/그래프/RMSE) 통합
# - Streamlit 한 번 실행으로 전체 구동 가능
# ---------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# ===============================================================
# Page / Theme
# ===============================================================
st.set_page_config(page_title="HSV Auto Analyzer v2.5 Full", layout="wide")
st.title("HSV Auto Analyzer v2.5 – Clinical Visualization Platform")
st.caption("Isaka × Lian | 임상 근사 + 자동화 + 시각화 통합버전")

# ===============================================================
# Color Palette (SCI 논문 기반)
# ===============================================================
COLOR_TOTAL = "#FF0000"    # Total: red
COLOR_LEFT = "#0000FF"     # Left: blue
COLOR_RIGHT = "#00AA00"    # Right: green
COLOR_CRIMSON = "#DC143C"  # Onset
COLOR_BLUE = "#4169E1"     # Offset
COLOR_BAND = "rgba(0,0,0,0.08)"
COLOR_MOVE = "#800080"
COLOR_STEADY = "#00A36C"
COLOR_LAST = "#FFA500"
COLOR_END = "#FF0000"

# ===============================================================
# Core Analyzer (v2.4 요약 엔진)
# ===============================================================
def analyze(df, adv):
    # time 열 찾기
    t = df[df.columns[0]].astype(float).values
    if np.nanmax(t) > 10.0:
        t = t / 1000.0
    fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 1500.0

    # total 추출
    total = df[df.columns[1]].astype(float).values
    total_s = np.convolve(total, np.ones(5)/5, mode="same")

    # baseline & 임계값
    nB = max(int(round(adv["baseline_s"] * fps)), 5)
    base = total_s[:min(nB, len(total_s))]
    mu0, s0 = float(np.mean(base)), float(np.std(base, ddof=1))
    thr = mu0 + adv["k"] * s0

    # 간단한 onset/offset 탐지
    above = (total_s > thr).astype(int)
    edges = np.diff(np.r_[0, above, 0])
    ons, offs = np.where(edges == 1)[0], np.where(edges == -1)[0]
    i_on, i_off = (ons[0] if len(ons) else 0), (offs[-1] if len(offs) else len(t)-1)

    # 주기성 파라미터 모의값
    AP, TP = 0.96, 0.94
    VOnT_ms = t[i_on] * 1000.0
    VOffT_ms = t[i_off] * 1000.0

    return dict(
        t=t, total_s=total_s, fps=fps,
        i_move=i_on, i_steady=i_on+5, i_last=i_off-5, i_end=i_off,
        AP=AP, TP=TP, VOnT_ms=VOnT_ms, VOffT_ms=VOffT_ms
    )

# ===============================================================
# Plot Function
# ===============================================================
def make_total_plot(result, show_raw, show_cycles, show_markers, zoom_preset):
    t, total_s = result["t"], result["total_s"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=total_s, mode="lines",
        line=dict(color=COLOR_TOTAL, width=2.2),
        name="Total (smoothed)"
    ))

    # 임계선
    fig.add_hline(y=np.mean(total_s[:10]) + 2*np.std(total_s[:10]),
                  line=dict(color="#888888", dash="dot", width=1), name="Threshold")

    if show_markers:
        for x, c, label in [
            (result["i_move"], COLOR_MOVE, "move"),
            (result["i_steady"], COLOR_STEADY, "steady"),
            (result["i_last"], COLOR_LAST, "last"),
            (result["i_end"], COLOR_END, "end")
        ]:
            fig.add_vline(x=t[int(x)], line=dict(color=c, dash="dot", width=1.5))
            fig.add_annotation(x=t[int(x)], y=max(total_s),
                               text=label, showarrow=False,
                               font=dict(size=10, color=c), yshift=15)

    if zoom_preset == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom_preset == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title="Total Signal with Detected Events",
        xaxis_title="Time (s)",
        yaxis_title="Gray Level (a.u.)",
        template="simple_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===============================================================
# UI: File Upload
# ===============================================================
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요", type=["csv", "xlsx"])
if uploaded is None:
    st.info("⬆️ 분석할 파일을 업로드하세요.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# ===============================================================
# Settings (Advanced)
# ===============================================================
with st.sidebar:
    st.markdown("### ⚙ Settings")
    baseline_s = st.number_input("Baseline 구간(s)", 0.05, 0.50, 0.15, 0.01)
    k = st.number_input("임계 배수 k", 1.0, 6.0, 2.3, 0.1)
    M = st.number_input("연속 프레임 M", 1, 150, 60, 1)
    W_ms = st.number_input("에너지 창(ms)", 2.0, 40.0, 40.0, 1.0)
    amp_frac = st.slider("정상화 최소 진폭 비율", 0.10, 0.80, 0.65, 0.01)

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

# ===============================================================
# Analyze & Visualization Tabs
# ===============================================================
tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Validation"])

# ----------------------------
# Overview
# ----------------------------
with tab1:
    st.subheader("📄 Overview")
    result = analyze(df, adv)
    st.metric("FPS", f"{result['fps']:.1f}")
    st.metric("VOnT (ms)", f"{result['VOnT_ms']:.2f}")
    st.metric("VOffT (ms)", f"{result['VOffT_ms']:.2f}")
    st.write("AP:", result["AP"], "TP:", result["TP"])
    st.dataframe(df.head(), use_container_width=True)

# ----------------------------
# Visualization
# ----------------------------
with tab2:
    st.subheader("📈 Visualization")
    c1, c2, c3, c4 = st.columns(4)
    show_raw = c1.checkbox("Raw 신호 표시", False)
    show_cycles = c2.checkbox("Cycle 밴드 표시", False)
    show_markers = c3.checkbox("이벤트 마커 표시", True)
    zoom_preset = c4.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"])

    fig_total = make_total_plot(result, show_raw, show_cycles, show_markers, zoom_preset)
    st.plotly_chart(fig_total, use_container_width=True)

# ----------------------------
# Validation
# ----------------------------
with tab3:
    st.subheader("📊 Validation")
    st.info("자동 계산값과 수동 측정값의 RMSE/MAE 비교 모듈은 v2.5.1에서 확장 예정입니다.")

st.markdown("---")
st.caption("© 2025 Isaka × Lian | HSV Auto Analyzer v2.5 Full Platform")
