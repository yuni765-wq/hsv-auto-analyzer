# ---------------------------------------------------------------
# HSV Auto Analyzer v2.5 – Clinical Visualization Platform (app)
# Isaka × Lian
# ---------------------------------------------------------------
# 이 파일은 "실행 전용" 통합본입니다.
# 분석 로직은 같은 폴더의 app_v2_4_Final.py 에서 import 합니다.
# 실행: streamlit run app.py
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ❶ v2.4 엔진 그대로 사용 (파일명/함수명 반드시 일치)
from app_v2_4_Final import analyze  # ← 같은 폴더에 app_v2_4_Final.py 필수

# ──────────────────────────────────────────────────────────────
# 페이지 세팅
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="HSV Auto Analyzer v2.5 – Clinical Visualization", layout="wide")
st.title("HSV Auto Analyzer v2.5 – Clinical Visualization Platform")
st.caption("Isaka × Lian | 임상 근사 + 자동화 + 시각화 통합버전 (v2.4 엔진 + v2.5 UI)")

# 색상 (SCI 팔레트 매핑)
COLOR_TOTAL   = "#FF0000"   # Total: red
COLOR_LEFT    = "#0000FF"   # Left: blue
COLOR_RIGHT   = "#00AA00"   # Right: green
COLOR_CRIMSON = "#DC143C"   # Onset 계열
COLOR_BLUE    = "#4169E1"   # Offset 계열
COLOR_BAND    = "rgba(0,0,0,0.08)"
COLOR_MOVE    = "#800080"
COLOR_STEADY  = "#00A36C"
COLOR_LAST    = "#FFA500"
COLOR_END     = "#FF0000"

# ──────────────────────────────────────────────────────────────
# 사이드바: v2.4 파라미터 그대로
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Settings")
    baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.08, step=0.01)
    k          = st.number_input("임계 배수 k",      min_value=0.50, max_value=6.00, value=2.30, step=0.10)
    M          = st.number_input("연속 프레임 M",     min_value=1,    max_value=150,  value=60,   step=1)
    W_ms       = st.number_input("에너지 창(ms)",     min_value=2.0,  max_value=40.0, value=40.0, step=1.0)
    amp_frac   = st.slider("정상화 최소 진폭 비율", 0.10, 0.80, 0.65, 0.01)

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

# ──────────────────────────────────────────────────────────────
# 파일 업로드
# ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요", type=["csv", "xlsx"])
if uploaded is None:
    st.info("⬆️ 분석할 파일을 업로드하세요.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# ──────────────────────────────────────────────────────────────
# ❷ v2.4 엔진으로 분석 실행
#   analyze(df, adv) → (summary, per_cycle, extras)
#   extras['viz'] 안에 그래프 재료(시계열, 임계선, 마커 등)가 포함되어야 함
# ──────────────────────────────────────────────────────────────
summary, per_cycle, extras = analyze(df, adv)
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

# ──────────────────────────────────────────────────────────────
# 그래프 함수
# ──────────────────────────────────────────────────────────────
def make_total_plot(show_cycles=True, show_markers=True, zoom="전체"):
    fig = go.Figure()
    if t is None or total_s is None:
        fig.update_layout(template="simple_white", height=360)
        return fig

    fig.add_trace(go.Scatter(x=t, y=total_s, mode="lines",
                             line=dict(color=COLOR_TOTAL, width=2.2),
                             name="Total (smoothed)"))

    if show_cycles and cycles:
        for s, e in cycles[:120]:
            fig.add_vrect(x0=t[s], x1=t[e], fillcolor=COLOR_BAND, opacity=0.08, line_width=0)

    if show_markers:
        for idx, col, label in [
            (i_move,   COLOR_MOVE,   "move"),
            (i_steady, COLOR_STEADY, "steady"),
            (i_last,   COLOR_LAST,   "last"),
            (i_end,    COLOR_END,    "end"),
        ]:
            if idx is not None and 0 <= int(idx) < len(t):
                xval = t[int(idx)]
                fig.add_vline(x=xval, line=dict(color=col, dash="dot", width=1.6))
                fig.add_annotation(x=xval, y=float(np.nanmax(total_s)),
                                   text=label, showarrow=False,
                                   font=dict(size=10, color=col), yshift=14)

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title="Total Signal with Detected Events",
        xaxis_title="Time (s)", yaxis_title="Gray Level (a.u.)",
        template="simple_white", height=380,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig

def make_lr_plot(normalize=False, zoom="전체"):
    fig = go.Figure()
    if t is None or (left_s is None and right_s is None):
        fig.update_layout(template="simple_white", height=340); return fig

    def _norm(x):
        if x is None: return None
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)

    L = _norm(left_s) if normalize else left_s
    R = _norm(right_s) if normalize else right_s

    if L is not None:
        fig.add_trace(go.Scatter(x=t, y=L, name="Left",
                                 mode="lines", line=dict(color=COLOR_LEFT, width=2.0)))
    if R is not None:
        fig.add_trace(go.Scatter(x=t, y=R, name="Right",
                                 mode="lines", line=dict(color=COLOR_RIGHT, width=2.0, dash="dot")))

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Left vs Right (AS {AS:.2f} · PS {PS:.2f})",
        xaxis_title="Time (s)",
        yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"),
        template="simple_white", height=340,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig

def make_energy_plot(mode="on", show_markers=True, zoom="전체"):
    fig = go.Figure()
    if t is None: 
        fig.update_layout(template="simple_white", height=320); return fig

    if mode == "on":
        E, thr, tlow, color, label, event_idx = E_on, thr_on, Tlow_on, COLOR_CRIMSON, "Onset", i_move
    else:
        E, thr, tlow, color, label, event_idx = E_off, thr_off, Tlow_off, COLOR_BLUE, "Offset", i_end

    if E is not None:
        fig.add_trace(go.Scatter(x=t, y=E, name=f"E_{label.lower()}",
                                 mode="lines", line=dict(color=color, width=2.0)))
    if thr is not None:
        fig.add_hline(y=float(thr), line=dict(color=color, width=1.5),
                      annotation_text=f"thr_{label.lower()}", annotation_position="top left")
    if tlow is not None:
        fig.add_hline(y=float(tlow), line=dict(color=color, dash="dot", width=1.2),
                      annotation_text=f"Tlow_{label.lower()}", annotation_position="bottom left")

    if show_markers and event_idx is not None and 0 <= int(event_idx) < len(t):
        xval = t[int(event_idx)]
        fig.add_vline(x=xval, line=dict(color=color, dash="dot", width=1.6))
        if E is not None:
            fig.add_annotation(x=xval, y=float(np.nanmax(E)), text=f"{label} @ {xval*1000.0:.2f} ms",
                               showarrow=False, font=dict(size=10, color=color), yshift=14)

    if zoom == "0–0.2s":
        fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s":
        fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Energy & Thresholds – {label}",
        xaxis_title="Time (s)", yaxis_title="Energy (a.u.)",
        template="simple_white", height=320,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig

# ──────────────────────────────────────────────────────────────
# 탭
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Validation"])

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

with tab2:
    st.subheader("📈 Visualization")
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    show_cycles   = cc1.checkbox("Cycle 밴드 표시", True)
    show_markers  = cc2.checkbox("이벤트 마커 표시", True)
    zoom_preset   = cc3.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"])
    normalize_lr  = cc4.checkbox("좌/우 정규화", False)
    energy_mode   = cc5.radio("에너지 뷰", ["Onset", "Offset"], horizontal=True)

    st.markdown("#### A) Total")
    st.plotly_chart(make_total_plot(show_cycles, show_markers, zoom_preset), use_container_width=True)

    st.markdown("#### B) Left vs Right")
    st.plotly_chart(make_lr_plot(normalize_lr, zoom_preset), use_container_width=True)

    st.markdown("#### C) Energy + Thresholds")
    st.plotly_chart(make_energy_plot("on" if energy_mode == "Onset" else "off",
                                     show_markers, zoom_preset), use_container_width=True)

with tab3:
    st.subheader("📊 Validation (RMSE / MAE / Bias)")
    st.info("자동 vs 수동 측정치 정량검증은 v2.5.1에서 확장 예정입니다.")
