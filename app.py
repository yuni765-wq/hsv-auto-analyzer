# ---------------------------------------------------------------
# HSV Auto Analyzer v3 – Merged (v2.5 engine + v3 features)
# Isaka × Lian
# ---------------------------------------------------------------
# 실행: streamlit run app_v3.py
# 요구: streamlit, plotly, pandas, numpy, (optional) scipy
# ---------------------------------------------------------------

import math
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# optional: scipy savgol (없어도 동작)
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

# ================= UI 기본 =================
st.set_page_config(page_title="HSV Auto Analyzer v3 – Adaptive Clinical Engine", layout="wide")
st.title("HSV Auto Analyzer v3 – Adaptive Clinical Engine (Merged)")
st.caption("v2.5 임상 엔진 + v3 프로필/배치/비교 기능 통합")

# 색상 팔레트 (v2.5 유지)
COLOR_TOTAL   = "#FF0000"
COLOR_LEFT    = "#0066FF"
COLOR_RIGHT   = "#00AA00"
COLOR_CRIMSON = "#DC143C"
COLOR_ROYAL   = "#4169E1"
COLOR_BAND    = "rgba(0,0,0,0.08)"
COLOR_MOVE    = "#800080"
COLOR_STEADY  = "#00A36C"
COLOR_LAST    = "#FFA500"
COLOR_END     = "#FF0000"

# ================= 유틸 =================
def _norm_cols(cols):
    return [c.lower().strip().replace(" ", "_") for c in cols]

def _moving_rms(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return np.sqrt(np.maximum(x * x, 0.0))
    w = int(w)
    pad = w // 2
    xx = np.pad(x.astype(float), (pad, pad), mode="edge")
    ker = np.ones(w) / float(w)
    m2 = np.convolve(xx * xx, ker, mode="valid")
    return np.sqrt(np.maximum(m2, 0.0))

def _smooth(signal: np.ndarray, fps: float) -> np.ndarray:
    if signal is None: return None
    n = len(signal)
    if n < 7:
        return signal.astype(float)
    base_w = int(max(7, min(21, round(fps * 0.007))))  # ~7ms 근처
    win = base_w if (base_w % 2 == 1) else base_w + 1
    win = min(win, n - 1) if n % 2 == 0 and win >= n else min(win, n - (1 - (n % 2)))
    if _HAS_SAVGOL:
        try:
            return savgol_filter(signal.astype(float), window_length=win, polyorder=3, mode="interp")
        except Exception:
            pass
    # fallback: moving average
    pad = win // 2
    xx = np.pad(signal.astype(float), (pad, pad), mode="edge")
    ker = np.ones(win) / float(win)
    return np.convolve(xx, ker, mode="valid")

def _detect_peaks(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=int)
    y1 = y[1:] - y[:-1]
    s = np.sign(y1)
    idx = np.where((s[:-1] > 0) & (s[1:] <= 0))[0] + 1
    return idx.astype(int)

def _build_cycles(t: np.ndarray, signal: np.ndarray, min_frames: int = 5) -> list:
    peaks = _detect_peaks(signal)
    cycles = []
    if len(peaks) < 2:
        return cycles
    for i in range(len(peaks) - 1):
        s = int(peaks[i]); e = int(peaks[i + 1])
        if (e - s) >= max(2, min_frames):
            cycles.append((s, e))
    return cycles

def _clamp01(x):
    if x is None or np.isnan(x): return np.nan
    return float(max(0.0, min(1.0, x)))

def _nanmean0(x):
    v = np.nanmean(x) if len(x) else np.nan
    return 0.0 if (v is None or np.isnan(v)) else float(v)

# ================= Metrics (v2.5) =================

def _ap_tp(t: np.ndarray, total: np.ndarray, cycles: list) -> tuple:
    if len(cycles) < 3:
        return (np.nan, np.nan)
    amps, periods = [], []
    for s, e in cycles:
        seg = total[s:e]
        amp = float(np.nanmax(seg) - np.nanmin(seg))
        Ti = float(t[e] - t[s])
        amps.append(amp)
        periods.append(max(Ti, 1e-9))
    amps = np.array(amps, float); periods = np.array(periods, float)

    def _periodicity(v):
        m = np.nanmean(v)
        s = np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
        if not np.isfinite(m) or m <= 0:
            return np.nan
        return _clamp01(1.0 - (s / m))

    TP = _periodicity(periods)
    AP = _periodicity(amps)
    return (AP, TP)

def _as_range(left: np.ndarray, right: np.ndarray, cycles: list) -> float:
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    ratios = []
    for s, e in cycles:
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        m = max(L, R)
        ratios.append((min(L, R) / m) if m > 0 else np.nan)
    return _clamp01(_nanmean0(ratios))

def _ps(left: np.ndarray, right: np.ndarray, t: np.ndarray, cycles: list) -> float:
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    diffs = []
    for s, e in cycles:
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s]) if (t is not None) else 1.0
        if Ti <= 0: continue
        d = abs(float(t[li] - t[ri])) / Ti
        diffs.append(min(1.0, d))
    if not len(diffs): return np.nan
    return _clamp01(1.0 - _nanmean0(diffs))

# ================= v2.5 엔진 =================

def analyze_v25(df: pd.DataFrame, adv: dict):
    cols = _norm_cols(df.columns.tolist())
    df.columns = cols

    def pick(key):
        for c in cols:
            if key in c: return c
        return None

    time_col   = pick("time")
    left_col   = pick("left")
    right_col  = pick("right")
    total_col  = pick("total")
    onset_col  = pick("onset")
    offset_col = pick("offset")

    if time_col is None:
        empty = pd.DataFrame()
        return pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0, viz={})

    t = df[time_col].astype(float).values
    if np.nanmax(t) > 10.0:  # ms → s
        t = t / 1000.0

    if total_col is not None:
        total = df[total_col].astype(float).values
    elif left_col is not None and right_col is not None:
        total = (df[left_col].astype(float).values + df[right_col].astype(float).values) / 2.0
    else:
        empty = pd.DataFrame()
        return pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0, viz={})

    left  = df[left_col].astype(float).values  if left_col  else None
    right = df[right_col].astype(float).values if right_col else None

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0

    total_s = _smooth(total, fps)
    left_s  = _smooth(left, fps)  if left  is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    min_frames = max(int(0.002 * fps), 5)
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    AP, TP = _ap_tp(t, total_s, cycles)
    AS     = _as_range(left_s, right_s, cycles)
    PS     = _ps(left_s, right_s, t, cycles)

    W_ms       = float(adv.get("W_ms", 35.0))
    baseline_s = float(adv.get("baseline_s", 0.06))
    k          = float(adv.get("k", 1.10))
    amp_frac   = float(adv.get("amp_frac", 0.70))

    hysteresis_ratio = 0.70
    min_event_ms     = 40.0
    refractory_ms    = 30.0

    W = max(int(round((W_ms / 1000.0) * fps)), 3)

    def _energy(trace):
        return _moving_rms(np.abs(np.diff(trace, prepend=trace[0])), W)

    onset_series  = df[onset_col].astype(float).values  if onset_col  else total_s
    offset_series = df[offset_col].astype(float).values if offset_col else total_s

    E_on  = _energy(onset_series)
    E_off = _energy(offset_series)

    nB = max(int(round(baseline_s * fps)), 5)
    def _thr(E):
        base = E[:min(nB, len(E))]
        mu0 = float(np.mean(base)) if len(base) else 0.0
        s0  = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        Th  = mu0 + k * s0
        Tl  = hysteresis_ratio * Th
        return Th, Tl

    Th_on,  Tl_on  = _thr(E_on)
    Th_off, Tl_off = _thr(E_off)

    def _hyst_detect(E, Th, Tl):
        above = (E >= Th).astype(int)
        low   = (E >= Tl).astype(int)
        min_frames_ev = max(1, int(round((min_event_ms/1000.0) * fps)))
        refr_frames   = max(1, int(round((refractory_ms/1000.0) * fps)))
        starts, ends = [], []
        i = 0; N = len(E); state = 0
        while i < N:
            if state == 0:
                if i + min_frames_ev <= N and np.all(above[i:i+min_frames_ev] == 1):
                    state = 1; starts.append(i); i += min_frames_ev; i += refr_frames; continue
                i += 1
            else:
                if low[i] == 1: i += 1
                else: ends.append(i); state = 0; i += refr_frames
        return np.array(starts, int), np.array(ends, int)

    on_starts, on_ends   = _hyst_detect(E_on,  Th_on,  Tl_on)
    off_starts, off_ends = _hyst_detect(E_off, Th_off, Tl_off)

    i_move = int(on_starts[0]) if len(on_starts) else (cycles[0][0] if len(cycles) else None)

    VOnT = np.nan; VOffT = np.nan; i_steady = None; i_last = None; i_end = None
    if len(cycles) >= 1 and i_move is not None:
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles])) if cycles else 0.0
        for s, e in cycles:  # steady
            if s <= i_move:   continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp):
                i_steady = int(s); break
        MIN_VONT_GAP = int(round(0.004 * fps))
        if i_steady is not None and (i_steady - i_move) < MIN_VONT_GAP:
            for s, e in cycles:
                if s <= i_move + MIN_VONT_GAP: continue
                amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
                if g_amp <= 0 or (amp >= amp_frac * g_amp): i_steady = int(s); break
        if i_steady is None: i_steady = cycles[0][0] if cycles else i_move
        for s, e in reversed(cycles):  # last steady
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp): i_last = int(s); break
        if i_last is None: i_last = cycles[-1][0] if cycles else (len(t)-1)
        idxs = np.where(off_ends >= i_last)[0] if len(off_ends) else []
        if len(idxs): i_end = int(off_ends[idxs[-1]])
        else: i_end = cycles[-1][1] if cycles else (len(t)-1)
        t_move, t_steady, t_last, t_end = map(lambda i: float(t[i]) if i is not None else np.nan,
                                              [i_move, i_steady, i_last, i_end])
        VOnT  = (t_steady - t_move) * 1000.0 if (np.isfinite(t_steady) and np.isfinite(t_move)) else np.nan
        VOffT = (t_end - t_last)   * 1000.0 if (np.isfinite(t_end) and np.isfinite(t_last)) else np.nan

    summary = pd.DataFrame({
        "Parameter": [
            "Amplitude Periodicity (AP)",
            "Time Periodicity (TP)",
            "Amplitude Symmetry (AS)",
            "Phase Symmetry (PS)",
            "Voice Onset Time (VOnT, ms)",
            "Voice Offset Time (VOffT, ms)",
        ],
        "Value": [AP, TP, AS, PS, VOnT, VOffT]
    })

    viz = dict(
        t=t, total_s=total_s, left_s=left_s, right_s=right_s,
        E_on=E_on, E_off=E_off, thr_on=Th_on, thr_off=Th_off,
        Tlow_on=Tl_on, Tlow_off=Tl_off,
        i_move=i_move, i_steady=i_steady, i_last=i_last, i_end=i_end,
        cycles=cycles, fps=fps
    )
    extras = dict(fps=fps, n_cycles=len(cycles), viz=viz)
    return summary, pd.DataFrame(), extras

# ================= 프리셋/프로필 =================
PRESETS = {
    "Normal": dict(baseline_s=0.06, k=1.10, M=40, W_ms=35, amp_frac=0.70),
    "ULP":    dict(baseline_s=0.06, k=1.50, M=40, W_ms=35, amp_frac=0.60),
    "SD":     dict(baseline_s=0.06, k=1.75, M=50, W_ms=40, amp_frac=0.75),
}

# ================= 사이드바 =================
with st.sidebar:
    st.subheader("분석 프로필")
    profile_name = st.selectbox("Preset", list(PRESETS.keys()) + ["Custom"], index=0)
    base = PRESETS.get(profile_name, PRESETS["Normal"]).copy()

    # v2.5 파라미터 노출
    baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=float(base.get('baseline_s',0.06)), step=0.01)
    k          = st.number_input("임계 배수 k", min_value=0.50, max_value=6.00, value=float(base.get('k',1.10)), step=0.10)
    M          = st.number_input("연속 프레임 M (참고)", min_value=1, max_value=150, value=int(base.get('M',40)), step=1)
    W_ms       = st.number_input("에너지 창(ms)", min_value=2.0, max_value=60.0, value=float(base.get('W_ms',35.0)), step=1.0)
    amp_frac   = st.slider("정상화 최소 진폭 비율", 0.10, 0.95, float(base.get('amp_frac',0.70)), 0.01)
    st.caption("히스테리시스/디바운스/불응기간: 0.7 / 40ms / 30ms (v2.5)")

    ADV = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

    st.markdown("---")
    st.subheader("자동 권고 (라이트)")
    st.caption("파일 로드 후 0.3–0.6s 구간의 비대칭/변동성으로 Normal/ULP/SD 권고")

# ================= 공통 로더 =================
def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith('.csv'): return pd.read_csv(file)
    return pd.read_excel(file)

# ================= 자동 추천 (라이트) =================

def proxy_as(df: pd.DataFrame) -> float:
    cols = _norm_cols(df.columns.tolist()); df.columns = cols
    if 'left' in cols and 'right' in cols:
        L = df['left'].astype(float).values
        R = df['right'].astype(float).values
        denom = np.maximum((np.abs(L) + np.abs(R)), 1e-6)
        return float(np.nanmean(np.abs(L - R) / denom))
    return np.nan

def auto_suggest_profile(df: pd.DataFrame) -> str:
    cols = _norm_cols(df.columns.tolist()); df.columns = cols
    t = df['time'].astype(float).values if 'time' in cols else np.arange(len(df))
    if np.nanmax(t) > 10.0: t = t/1000.0
    if 'total' in cols: sig = df['total'].astype(float).values
    elif ('left' in cols and 'right' in cols): sig = (df['left'].astype(float).values + df['right'].astype(float).values)/2.0
    else: return "Normal"

    # 0.3–0.6s 윈도우
    mask = (t >= 0.3) & (t <= 0.6)
    if mask.sum() < 5: mask = np.ones_like(t, dtype=bool)
    ts = t[mask]; ys = sig[mask]

    # onset proxy (70% crossing)
    thr = 0.70 * (np.nanmax(ys) if np.nanmax(ys)!=0 else 1.0)
    abv = ys >= thr
    von_proxy = np.inf
    if np.any(abv):
        von_proxy = float(ts[np.argmax(abv)] - ts[0])

    AS_p = proxy_as(df[mask])
    cv = float(np.std(ys) / (np.mean(np.abs(ys)) + 1e-6)) if len(ys)>0 else np.nan

    if (not np.isnan(AS_p) and AS_p > 0.25) or (von_proxy > 0.12):
        return "ULP"
    if (not np.isnan(cv) and cv > 0.8):
        return "SD"
    return "Normal"

# ================= 플롯 빌더 (v2.5) =================

def build_total_plot(viz, zoom="전체", show_cycles=True, show_markers=True):
    t = viz.get('t'); total_s = viz.get('total_s'); cycles = viz.get('cycles', [])
    i_move = viz.get('i_move'); i_steady = viz.get('i_steady'); i_last = viz.get('i_last'); i_end = viz.get('i_end')
    fig = go.Figure()
    if t is None or total_s is None:
        fig.update_layout(template="simple_white", height=360); return fig
    fig.add_trace(go.Scatter(x=t, y=total_s, mode="lines", line=dict(color=COLOR_TOTAL, width=2.2), name="Total (smoothed)"))
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
                fig.add_annotation(x=xval, y=float(np.nanmax(total_s)), text=label,
                                   showarrow=False, font=dict(size=10, color=col), yshift=14)
    if zoom == "0–0.2s": fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])
    fig.update_layout(title="Total Signal with Detected Events", xaxis_title="Time (s)", yaxis_title="Gray Level (a.u.)",
                      template="simple_white", height=380, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig

def build_lr_plot(viz, AS, PS, normalize=False, zoom="전체"):
    t = viz.get('t'); left_s = viz.get('left_s'); right_s = viz.get('right_s')
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
        fig.add_trace(go.Scatter(x=t, y=L, name="Left", mode="lines", line=dict(color=COLOR_LEFT, width=2.0)))
    if R is not None:
        fig.add_trace(go.Scatter(x=t, y=R, name="Right", mode="lines", line=dict(color=COLOR_RIGHT, width=2.0, dash="dot")))
    if zoom == "0–0.2s": fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])
    fig.update_layout(title=f"Left vs Right (AS {AS:.2f} · PS {PS:.2f})", xaxis_title="Time (s)",
                      yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"), template="simple_white", height=340,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig

def build_energy_plot(viz, mode="on", show_markers=True, zoom="전체"):
    t = viz.get('t')
    if t is None:
        fig = go.Figure(); fig.update_layout(template="simple_white", height=320); return fig
    if mode == "on":
        E, Th, Tl, color, label, event_idx = viz.get('E_on'), viz.get('thr_on'), viz.get('Tlow_on'), COLOR_CRIMSON, "Onset", viz.get('i_move')
    else:
        E, Th, Tl, color, label, event_idx = viz.get('E_off'), viz.get('thr_off'), viz.get('Tlow_off'), COLOR_ROYAL, "Offset", viz.get('i_end')
    fig = go.Figure()
    if E is not None:
        fig.add_trace(go.Scatter(x=t, y=E, name=f"E_{label.lower()}", mode="lines", line=dict(color=color, width=2.0)))
    if Th is not None:
        fig.add_hline(y=float(Th), line=dict(color=color, width=1.5), annotation_text=f"thr_{label.lower()}", annotation_position="top left")
    if Tl is not None:
        fig.add_hline(y=float(Tl), line=dict(color=color, dash="dot", width=1.2), annotation_text=f"Tlow_{label.lower()}", annotation_position="bottom left")
    if show_markers and event_idx is not None and 0 <= int(event_idx) < len(t):
        xval = t[int(event_idx)]
        fig.add_vline(x=xval, line=dict(color=color, dash="dot", width=1.6))
        if E is not None:
            fig.add_annotation(x=xval, y=float(np.nanmax(E)), text=f"{label} @ {xval*1000.0:.2f} ms", showarrow=False, font=dict(size=10, color=color), yshift=14)
    if zoom == "0–0.2s": fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])
    fig.update_layout(title=f"Energy & Thresholds – {label}", xaxis_title="Time (s)", yaxis_title="Energy (a.u.)",
                      template="simple_white", height=320, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig

# ================= 탭 =================
T1, T2, T3, T4 = st.tabs(["① 단일 분석", "② 시각화", "③ 배치 검증", "④ 프리셋 비교(수동값 없음)"])

# ---------- 탭1: 단일 분석 ----------
with T1:
    st.markdown("### CSV 업로드 (단일)")
    file_single = st.file_uploader("CSV/XLSX 1개", type=["csv","xlsx"], key="single")
    if file_single is not None:
        df = load_df(file_single)
        # 자동 권고 프로필
        sug = auto_suggest_profile(df)
        st.info(f"자동 권고 프로필: **{sug}** (현재 선택: {profile_name})")

        summary, _, extras = analyze_v25(df, ADV)
        viz = extras.get("viz", {})
        fps  = float(extras.get("fps", np.nan))
        ncyc = int(extras.get("n_cycles", 0))

        # 개요
        st.markdown("#### 개요")
        def _get(summary, key, default=np.nan):
            try:
                return float(summary.loc[summary["Parameter"]==key, "Value"].iloc[0])
            except Exception:
                return default
        AP = _get(summary, "Amplitude Periodicity (AP)")
        TP = _get(summary, "Time Periodicity (TP)")
        AS = _get(summary, "Amplitude Symmetry (AS)")
        PS = _get(summary, "Phase Symmetry (PS)")
        VOnT = _get(summary, "Voice Onset Time (VOnT, ms)")
        VOffT= _get(summary, "Voice Offset Time (VOffT, ms)")

        c1,c2,c3 = st.columns(3); c1.metric("AP", f"{AP:.4f}"); c2.metric("TP", f"{TP:.4f}"); c3.metric("AS", f"{AS:.4f}")
        d1,d2,d3 = st.columns(3); d1.metric("PS", f"{PS:.4f}"); d2.metric("VOnT(ms)", f"{VOnT:.2f}"); d3.metric("VOffT(ms)", f"{VOffT:.2f}")
        st.caption(f"FPS: {fps:.1f} | 검출된 사이클 수: {ncyc}")
        st.dataframe(summary, use_container_width=True)

        # QC flags
        qc_msgs = []
        if ncyc < 3: qc_msgs.append("cycles < 3")
        if AS is not None and not np.isnan(AS) and AS < 0.6: qc_msgs.append("AS < 0.6 (비대칭 가능)")
        VLast = _get(summary, "Voice Offset Time (VOffT, ms)")
        if (VOnT is not None) and (VLast is not None) and np.isfinite(VOnT) and np.isfinite(VLast) and (VOnT <= 0):
            qc_msgs.append("VOnT ≤ 0 → 프리셋 상향 또는 k↑/W↑ 권장")
        if qc_msgs: st.warning("QC 경고: " + "; ".join(qc_msgs))

# ---------- 탭2: 시각화 ----------
with T2:
    st.markdown("### 📈 Visualization")
    st.caption("A: Total / B: L–R / C: Energy")
    zoom_preset   = st.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"], index=0)
    colA, colB = st.columns([1,1])
    show_cycles   = colA.checkbox("Cycle 밴드 표시", True)
    show_markers  = colA.checkbox("이벤트 마커 표시", True)
    normalize_lr  = colB.checkbox("좌/우 정규화", False)
    energy_mode   = colB.radio("에너지 뷰", ["Onset","Offset"], index=0, horizontal=True)

    st.markdown("#### A) Total")
    st.plotly_chart(build_total_plot(viz, zoom_preset, show_cycles, show_markers), use_container_width=True)
    st.markdown("#### B) Left vs Right")
    st.plotly_chart(build_lr_plot(viz, AS if 'AS' in locals() else np.nan, PS if 'PS' in locals() else np.nan, normalize_lr, zoom_preset), use_container_width=True)
    st.markdown("#### C) Energy + Thresholds")
    st.plotly_chart(build_energy_plot(viz, "on" if energy_mode=="Onset" else "off", show_markers, zoom_preset), use_container_width=True)

# ---------- 탭3: 배치 검증 (RMSE/MAE/Bias) ----------
with T3:
    st.markdown("### 배치 검증")
    st.caption("최대 8개 권장. 수동 라벨(onset_manual/offset_manual)이 있을 때 RMSE/MAE/Bias 산출")
    files = st.file_uploader("CSV/XLSX 여러 개", type=["csv","xlsx"], accept_multiple_files=True, key="batch")
    if files:
        rows = []
        for ff in files[:8]:
            try:
                d = load_df(ff)
                # 자동 라벨
                label = None
                n = ff.name.lower()
                if n.startswith('ulp_'): label = 'ULP'
                elif n.startswith('sd_'): label = 'SD'
                else: label = 'Normal'

                sm, _, ex = analyze_v25(d, ADV)
                def _get(sm, key):
                    try: return float(sm.loc[sm["Parameter"]==key, "Value"].iloc[0])
                    except: return np.nan
                on_a = _get(sm, "Voice Onset Time (VOnT, ms)")/1000.0
                off_a= _get(sm, "Voice Offset Time (VOffT, ms)")/1000.0
                # 수동값(있다면 첫행 기준 ms→s)
                cols = _norm_cols(d.columns.tolist()); d.columns = cols
                on_m  = d['onset_manual'].iloc[0]/1000.0 if 'onset_manual' in cols else np.nan
                off_m = d['offset_manual'].iloc[0]/1000.0 if 'offset_manual' in cols else np.nan
                # RMSE/MAE/Bias
                def _err(a, m):
                    msk = np.isfinite([a,m]).all() if False else (np.isfinite(a) and np.isfinite(m))
                    if not msk: return (np.nan, np.nan, np.nan)
                    diff = a - m
                    return (abs(diff), abs(diff), diff)
                rmse_on, mae_on, bias_on = _err(on_a, on_m)
                rmse_off,mae_off,bias_off= _err(off_a, off_m)
                rows.append(dict(file=ff.name, label=label, preset=profile_name,
                                 baseline_s=ADV['baseline_s'], k=ADV['k'], M=ADV['M'], W_ms=ADV['W_ms'], amp_frac=ADV['amp_frac'],
                                 onset_auto_s=on_a, offset_auto_s=off_a,
                                 onset_manual_s=on_m, offset_manual_s=off_m,
                                 RMSE_on_s=rmse_on, MAE_on_s=mae_on, Bias_on_s=bias_on,
                                 RMSE_off_s=rmse_off, MAE_off_s=mae_off, Bias_off_s=bias_off))
            except Exception as e:
                rows.append(dict(file=getattr(ff,'name','?'), error=str(e)))
        res = pd.DataFrame(rows)
        st.dataframe(res, use_container_width=True)
        # 다운로드
        buf = io.StringIO(); res.to_csv(buf, index=False)
        st.download_button("CSV 로그 다운로드", buf.getvalue(), file_name="v3_batch_validation_log.csv", mime="text/csv")

# ---------- 탭4: 프리셋 비교 (수동값 없음) ----------
with T4:
    st.markdown("### 프리셋 비교 (수동값 없어도)")
    st.caption("동일 CSV에 Normal vs ULP/SD 적용 → 안정성/민감도 지표 비교")
    fcmp = st.file_uploader("CSV/XLSX 1개", type=["csv","xlsx"], key="cmp")
    col1, col2 = st.columns(2)
    p1 = col1.selectbox("프리셋 A", list(PRESETS.keys()), index=0)
    p2 = col2.selectbox("프리셋 B", list(PRESETS.keys()), index=1)

    def compute_no_manual_metrics(df: pd.DataFrame, adv: dict):
        sm, _, ex = analyze_v25(df, adv)
        def _get(sm, key):
            try: return float(sm.loc[sm["Parameter"]==key, "Value"].iloc[0])
            except: return np.nan
        VOnT = _get(sm, "Voice Onset Time (VOnT, ms)")/1000.0
        VOffT= _get(sm, "Voice Offset Time (VOffT, ms)")/1000.0
        AP   = _get(sm, "Amplitude Periodicity (AP)")
        TP   = _get(sm, "Time Periodicity (TP)")
        AS   = _get(sm, "Amplitude Symmetry (AS)")
        PS   = _get(sm, "Phase Symmetry (PS)")
        viz  = ex.get('viz', {})
        t    = viz.get('t'); sig = viz.get('total_s')
        if t is None or sig is None:
            cv = np.nan
        else:
            sig_abs = np.abs(sig)
            cv = float(np.std(sig_abs) / (np.mean(sig_abs) + 1e-6))
        # QC: 간단
        qc = 0
        if np.isnan(VOnT) or np.isnan(VOffT): qc += 1
        if not np.isnan(AS) and AS < 0.6: qc += 1
        return dict(preset=adv, VOnT_s=VOnT, VOffT_s=VOffT, AP=AP, TP=TP, AS=AS, PS=PS, energy_CV=cv, QC_flags=qc)

    if fcmp is not None:
        d = load_df(fcmp)
        m1 = compute_no_manual_metrics(d, PRESETS[p1].copy())
        m2 = compute_no_manual_metrics(d, PRESETS[p2].copy())
        r = pd.DataFrame([{**{"metric":"VOnT_s"}, **{"A":m1['VOnT_s'],"B":m2['VOnT_s'],"Δ(B−A)":m2['VOnT_s']-m1['VOnT_s']},
                          **{} }])
        out = pd.DataFrame([
            dict(metric="VOnT_s",   A=m1['VOnT_s'],   B=m2['VOnT_s'],   delta=m2['VOnT_s']-m1['VOnT_s']),
            dict(metric="VOffT_s",  A=m1['VOffT_s'],  B=m2['VOffT_s'],  delta=m2['VOffT_s']-m1['VOffT_s']),
            dict(metric="AP",        A=m1['AP'],       B=m2['AP'],       delta=(m2['AP']-m1['AP'])),
            dict(metric="TP",        A=m1['TP'],       B=m2['TP'],       delta=(m2['TP']-m1['TP'])),
            dict(metric="AS",        A=m1['AS'],       B=m2['AS'],       delta=(m2['AS']-m1['AS'])),
            dict(metric="PS",        A=m1['PS'],       B=m2['PS'],       delta=(m2['PS']-m1['PS'])),
            dict(metric="energy_CV", A=m1['energy_CV'],B=m2['energy_CV'],delta=(m2['energy_CV']-m1['energy_CV'])),
            dict(metric="QC_flags",  A=m1['QC_flags'], B=m2['QC_flags'], delta=(m2['QC_flags']-m1['QC_flags'])),
        ])
        st.dataframe(out, use_container_width=True)
        # 미리보기 플롯
        sm, _, ex = analyze_v25(d, PRESETS[p1])
        st.plotly_chart(build_total_plot(ex.get('viz',{}), show_cycles=True, show_markers=True), use_container_width=True)

st.markdown("---")
st.caption("NOTE: v3 정식본에서는 Adaptive Threshold Engine과 Pattern-based Detector, ROI/EGG 동기화가 추가됩니다.")
