# ---------------------------------------------------------------
# HSV Auto Analyzer v2.5 Full - Clinical Visualization Platform
# (Isaka × Lian)
# ---------------------------------------------------------------
# 구조: v2.4 엔진(import) + v2.5 UI/그래프/요약 메트릭
# 사용법: streamlit run app_v2.5_Full.py
# ---------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============ ❶ v2.4 분석엔진을 가져옵니다 (동일 폴더에 존재해야 함) ============
# 파일명/함수명은 어제 쓰신 이름에 맞춰주세요.
# 예) app_v2_4_Final.py 에 analyze(df, adv) 가 정의되어 있다고 가정
from app_v2_4_Final import analyze  # ★★ 핵심: v2.4 엔진 그대로 사용 ★★

# ============ 페이지 / 레이아웃 기본 ============ #
st.set_page_config(page_title="HSV Auto Analyzer v2.5 – Clinical Visualization", layout="wide")
st.title("HSV Auto Analyzer v2.5 – Clinical Visualization Platform")
st.caption("Isaka × Lian | 임상 근사 + 자동화 + 시각화 통합버전 (v2.4 엔진 + v2.5 UI)")

# ============ 색상 팔레트 (SCI 논문 팔레트 맵핑) ============ #
COLOR_TOTAL   = "#FF0000"   # Total: red
COLOR_LEFT    = "#0000FF"   # Left: blue
COLOR_RIGHT   = "#00AA00"   # Right: green
COLOR_CRIMSON = "#DC143C"   # Onset 계열
COLOR_BLUE    = "#4169E1"   # Offset 계열
COLOR_BAND    = "rgba(0,0,0,0.08)"  # cycle bands
COLOR_MOVE    = "#800080"   # move marker (purple)
COLOR_STEADY  = "#00A36C"   # steady marker (green)
COLOR_LAST    = "#FFA500"   # last steady marker (orange)
COLOR_END     = "#FF0000"   # end marker (red)

# ============ 사이드바: Settings (v2.4 파라미터 그대로) ============ #
with st.sidebar:
    st.markdown("### ⚙ Settings")
    baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.08, step=0.01)
    k          = st.number_input("임계 배수 k",      min_value=0.50, max_value=6.00, value=2.30, step=0.10)
    M          = st.number_input("연속 프레임 M",     min_value=1,    max_value=150,  value=60,   step=1)
    W_ms       = st.number_input("에너지 창(ms)",     min_value=2.0,  max_value=40.0, value=40.0, step=1.0)
    amp_frac   = st.slider("정상화 최소 진폭 비율", 0.10, 0.80, 0.65, 0.01)

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

# ============ 파일 업로드 ============ #
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요", type=["csv", "xlsx"])
if uploaded is None:
    st.info("⬆️ 분석할 파일을 업로드하세요.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# ============ ❷ v2.4 엔진으로 분석 실행 ============ #
# v2.4의 analyze()는 아래 3개를 반환한다고 가정:
# summary:  Parameter/Value 표 (AP/TP/AS/PS/VOnT/VOffT 포함)
# per_cycle: (옵션) 사이클 상세
# extras:   fps, n_cycles, 그리고 viz(dict) = 그래프 재료
summary, per_cycle, extras = analyze(df, adv)

# viz 파트(시간축, 신호, 에너지, 임계선/히스테리시스, 마커 등)
viz = extras.get("viz", {})
t         = viz.get("t", None)
total_s   = viz.get("total_s", None)
left_s    = viz.get("left_s", None)
right_s   = viz.get("right_s", None)
E_on      = viz.get("E_on", None)
E_off     = viz.get("E_off", None)
thr_on    = viz.get("thr_on", None)
thr_off   = viz.get("thr_off", None)
Tlow_on   = viz.get("Tlow_on", None)
Tlow_off  = viz.get("Tlow_off", None)
i_move    = viz.get("i_move", None)
i_steady  = viz.get("i_steady", None)
i_last    = viz.get("i_last", None)
i_end     = viz.get("i_end", None)
cycles    = viz.get("cycles", [])

# ============ 유틸: summary에서 값 뽑기 ============ #
def _get_val(param_name, default=np.nan):
    try:
        return float(summary.loc[summary["Parameter"] == param_name, "Value"].iloc[0])
    except Exception:
        return default

AP   = _get_val("Amplitude Periodicity (AP)")
TP   = _get_val("Time Periodicity (TP)")
AS   = _get_val("Amplitude Symmetry (AS)")
PS   = _get_val("Phase Symmetry (PS)")
VOnT = _get_val("Voice Onset Time (VOnT, ms)")
VOffT= _get_val("Voice Offset Time (VOffT, ms)")
fps  = float(extras.get("fps", np.nan))
ncyc = int(extras.get("n_cycles", 0))

# ============ 그래프 함수들 ============ #
def make_total_plot(show_cycles=True, show_markers=True, zoom="전체"):
    if t is None or total_s is None:
        fig = go.Figure()
        fig.update_layout(template="simple_white", height=360)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=total_s, mode="lines",
        line=dict(color=COLOR_TOTAL, width=2.2),
        name="Total (smoothed)"
    ))

    if show_cycles and cycles:
        for s, e in cycles[:80]:
            fig.add_vrect(x0=t[s], x1=t[e], fillcolor=COLOR_BAND, opacity=0.08, line_width=0)

    if show_markers:
        marks = [(i_move,   COLOR_MOVE,   "move"),
                 (i_steady, COLOR_STEADY, "steady"),
                 (i_last,   COLOR_LAST,   "last"),
                 (i_end,    COLOR_END,    "end")]
        for idx, col, label in marks:
            if idx is not None and 0 <= int(idx) < len(t):
                xval = t[int(idx)]
                fig.add_vline(x=xval, line=dict(color=col, dash="dot", width=1.6))
                fig.add_annotation(x=xval, y=float(np.nanmax(total_s)), text=label,
                                   showarrow=False, font=dict(size=10, color=col), yshift=14)

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title="Total Signal with Detected Events",
        xaxis_title="Time (s)", yaxis_title="Gray Level (a.u.)",
        template="simple_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_lr_plot(normalize=False, zoom="전체"):
    if t is None or (left_s is None and right_s is None):
        fig = go.Figure(); fig.update_layout(template="simple_white", height=340); return fig

    def _norm(x):
        if x is None: return None
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)

    L = _norm(left_s) if normalize else left_s
    R = _norm(right_s) if normalize else right_s

    fig = go.Figure()
    if L is not None:
        fig.add_trace(go.Scatter(x=t, y=L, name="Left",  mode="lines",
                                 line=dict(color=COLOR_LEFT, width=2.0)))
    if R is not None:
        fig.add_trace(go.Scatter(x=t, y=R, name="Right", mode="lines",
                                 line=dict(color=COLOR_RIGHT, width=2.0, dash="dot")))

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Left vs Right (AS {AS:.2f} · PS {PS:.2f})",
        xaxis_title="Time (s)", yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"),
        template="simple_white", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_energy_plot(mode="on", show_markers=True, zoom="전체"):
    if t is None: 
        fig = go.Figure(); fig.update_layout(template="simple_white", height=320); return fig

    if mode == "on":
        E, thr, tlow, color, label = E_on, thr_on, Tlow_on, COLOR_CRIMSON, "Onset"
        event_idx = i_move
    else:
        E, thr, tlow, color, label = E_off, thr_off, Tlow_off, COLOR_BLUE, "Offset"
        event_idx = i_end

    fig = go.Figure()
    if E is not None:
        fig.add_trace(go.Scatter(x=t, y=E, name=f"E_{label.lower()}", mode="lines",
                                 line=dict(color=color, width=2.0)))
    if thr is not None:
        fig.add_hline(y=float(thr), line=dict(color=color, width=1.5),
                      annotation_text=f"thr_{label.lower()}", annotation_position="top left")
    if tlow is not None:
        fig.add_hline(y=float(tlow), line=dict(color=color, width=1.2, dash="dot"),
                      annotation_text=f"Tlow_{label.lower()}", annotation_position="bottom left")

    if show_markers and event_idx is not None and 0 <= int(event_idx) < len(t):
        xval = t[int(event_idx)]
        fig.add_vline(x=xval, line=dict(color=color, dash="dot", width=1.6))
        fig.add_annotation(x=xval, y=(np.nanmax(E) if E is not None else 0),
                           text=f"{label} @ {xval*1000.0:.2f} ms",
                           showarrow=False, font=dict(size=10, color=color), yshift=14)

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Energy & Thresholds – {label}",
        xaxis_title="Time (s)", yaxis_title="Energy (a.u.)",
        template="simple_white", height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ============ 탭 ============ #
tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Validation"])

# ---------- Overview: 결과 메트릭 카드 + 표 ---------- #
with tab1:
    st.subheader("🩺 Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("AP", f"{AP:.4f}")
    c2.metric("TP", f"{TP:.4f}")
    c3.metric("AS", f"{AS:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("PS", f"{PS:.4f}")
    c5.metric("VOnT (ms)", f"{VOnT:.2f}")
    c6.metric("VOffT (ms)", f"{VOffT:.2f}")

    st.caption(f"FPS: {fps:.1f} | 검출된 사이클 수: {ncyc}")
    st.dataframe(summary, use_container_width=True)

# ---------- Visualization: 3패널 + 컨트롤 ---------- #
with tab2:
    st.subheader("📈 Visualization")

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    show_cycles   = cc1.checkbox("Cycle 밴드 표시", True)
    show_markers  = cc2.checkbox("이벤트 마커 표시", True)
    zoom_preset   = cc3.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"])
    normalize_lr  = cc4.checkbox("좌/우 정규화", False)
    energy_mode   = cc5.radio("에너지 뷰", ["Onset", "Offset"], horizontal=True)

    st.markdown("#### A) Total")
    fig_total = make_total_plot(show_cycles=show_cycles, show_markers=show_markers, zoom=zoom_preset)
    st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("#### B) Left vs Right")
    fig_lr = make_lr_plot(normalize=normalize_lr, zoom=zoom_preset)
    st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown("#### C) Energy + Thresholds")
    fig_en = make_energy_plot(mode=("on" if energy_mode=="Onset" else "off"),
                              show_markers=show_markers, zoom=zoom_preset)
    st.plotly_chart(fig_en, use_container_width=True)

# ---------- Validation: RMSE/MAE (placeholder; v2.5.1에서 확장) ---------- #
with tab3:
    st.subheader("📊 Validation (RMSE / MAE / Bias)")
    st.info("자동 vs 수동 측정치 정량검증 테이블/그래프는 v2.5.1에서 통합됩니다.")
    # 예: 여러 케이스 업로드 버퍼 + RMSE 요약 테이블 + Bias 히스토그램
    # (필요 시 여기로 확장)
    
st.markdown("---")
st.caption("© 2025 Isaka × Lian | HSV Auto Analyzer v2.5 (v2.4 engine + v2.5 UI)")
