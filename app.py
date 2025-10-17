# ---------------------------------------------------------------
# HSV Auto Analyzer v3-alpha – PS/AS 패치 버전 (single file)
# Isaka × Lian
# ---------------------------------------------------------------
# 실행: streamlit run app.py
# 요구: streamlit, plotly, pandas, numpy, (optional) scipy, openpyxl
# ---------------------------------------------------------------

import math
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

# ============== UI 기본 ==============
st.set_page_config(page_title="HSV Auto Analyzer v3-alpha – PS/AS 패치", layout="wide")
st.title("HSV Auto Analyzer v3-alpha – PS/AS 패치")
st.caption("Isaka × Lian | 조명 독립형 AS + PS(circular distance) 적용 · v2.5 기반 엔진")

# 색상 팔레트
COLOR_TOTAL   = "#FF0000"   # red (total)
COLOR_LEFT    = "#0066FF"   # blue-ish (left)
COLOR_RIGHT   = "#00AA00"   # green (right)
COLOR_CRIMSON = "#DC143C"   # onset energy/marker
COLOR_ROYAL   = "#4169E1"   # offset energy/marker
COLOR_BAND    = "rgba(0,0,0,0.08)"
COLOR_MOVE    = "#800080"   # move
COLOR_STEADY  = "#00A36C"   # steady
COLOR_LAST    = "#FFA500"   # last steady
COLOR_END     = "#FF0000"   # end

# ============== Low-level utils ==============
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

# 새: peak-to-peak helper
_def_pp = lambda x: float(np.nanmax(x) - np.nanmin(x)) if (x is not None and len(x)) else np.nan

# ============== Metrics ==============

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

# 기존 AS(범위 기반 평균)는 유지하되, 보완 지표 3종을 병기한다.

def _as_range_median(left: np.ndarray, right: np.ndarray, cycles: list) -> float:
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    ratios = []
    for s, e in cycles:
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        m = max(L, R)
        if m > 0:
            ratios.append(min(L, R) / m)
    return _clamp01(float(np.median(ratios))) if len(ratios) else np.nan

# 새: 게인 정규화 + AS_area/AS_corr 계산

def _compute_as_triplet(left_s, right_s, total_s, cycles, fps, amp_frac):
    if left_s is None or right_s is None or len(cycles) == 0:
        return dict(AS_range=np.nan, AS_area=np.nan, AS_corr=np.nan)

    # steady cycle 선택 (전역 진폭의 amp_frac 이상)
    g_amp = float(np.nanmax([_def_pp(total_s[s:e]) for s,e in cycles])) if cycles else 0.0
    steady = [(s,e) for (s,e) in cycles if g_amp<=0 or _def_pp(total_s[s:e]) >= amp_frac * g_amp]
    use_cyc = steady if len(steady)>=3 else cycles

    eps = 1e-12
    # 게인 보정 계수: steady 구간의 p2p 중앙값
    A_L = []; A_R = []
    for s,e in use_cyc:
        A_L.append(_def_pp(left_s[s:e])); A_R.append(_def_pp(right_s[s:e]))
    sL = float(np.nanmedian(A_L)) if len(A_L) else 1.0
    sR = float(np.nanmedian(A_R)) if len(A_R) else 1.0
    if not np.isfinite(sL) or sL<=0: sL = 1.0
    if not np.isfinite(sR) or sR<=0: sR = 1.0

    Lp = left_s  / (sL + eps)
    Rp = right_s / (sR + eps)

    r_list, q_list, c_list = [], [], []
    for s,e in use_cyc:
        # range ratio
        aL = _def_pp(Lp[s:e]); aR = _def_pp(Rp[s:e])
        if max(aL,aR) > eps:
            r_list.append(min(aL,aR)/max(aL,aR))
        # area ratio (trapz of abs)
        EL = float(np.trapz(np.abs(Lp[s:e]), dx=1.0/max(fps,1e-9)))
        ER = float(np.trapz(np.abs(Rp[s:e]), dx=1.0/max(fps,1e-9)))
        if max(EL,ER) > eps:
            q_list.append(min(EL,ER)/max(EL,ER))
        # correlation (shape)
        x = Lp[s:e]; y = Rp[s:e]
        if len(x)>=5 and np.nanstd(x)>0 and np.nanstd(y)>0:
            c = np.corrcoef(x,y)[0,1]
            c_list.append(max(0.0, float(c)))

    AS_range = float(np.median(r_list)) if len(r_list) else np.nan
    AS_area  = float(np.median(q_list)) if len(q_list) else np.nan
    AS_corr  = float(np.median(c_list)) if len(c_list) else np.nan
    return dict(AS_range=AS_range, AS_area=AS_area, AS_corr=AS_corr)

# 새: PS – circular distance 기반 + 표시용 distance 스케일 (0=정상, 1=비정상)

def _ps_circular(left: np.ndarray, right: np.ndarray, t: np.ndarray, cycles: list) -> tuple:
    if left is None or right is None or len(cycles) < 1:
        return (np.nan, np.nan)
    dists = []
    for s, e in cycles:
        segL = left[s:e]; segR = right[s:e]
        if len(segL) < 3 or len(segR) < 3: continue
        li = s + int(np.nanargmax(segL))
        ri = s + int(np.nanargmax(segR))
        Ti = float(t[e] - t[s]) if (t is not None) else np.nan
        if not np.isfinite(Ti) or Ti <= 0: continue
        dt = abs(float(t[li] - t[ri]))
        # circular (modulo) distance: 경계 래핑 보정
        dt_circ = min(dt, max(Ti - dt, 0.0))
        dists.append(min(1.0, dt_circ / Ti))
    if not len(dists):
        return (np.nan, np.nan)
    PS_dist = float(np.nanmean(dists))           # 0=동위상(정상), 1=완전 반위상
    PS_sim  = float(1.0 - PS_dist)               # 1=좋음, 0=나쁨 (과거 스케일)
    return ( _clamp01(PS_sim), _clamp01(PS_dist) )

# ============== v2.4 엔진: analyze(df, adv) ==============

def analyze(df: pd.DataFrame, adv: dict):
    """
    입력 df: time + (left/right 또는 total) + (선택) onset/offset.
    adv: dict(baseline_s, k, M, W_ms, amp_frac)
    반환: summary(DataFrame), per_cycle(빈), extras(dict: fps, n_cycles, viz)
    """
    # ---- 컬럼 매핑 ----
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

    # ---- fps ----
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0

    # ---- smoothing ----
    total_s = _smooth(total, fps)
    left_s  = _smooth(left, fps)  if left  is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    # ---- cycles ----
    min_frames = max(int(0.002 * fps), 5)
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    # ---- Metrics (AP/TP) ----
    AP, TP = _ap_tp(t, total_s, cycles)

    # ---- 에너지/임계/히스테리시스 기반 VOnT/VOffT ----
    W_ms       = float(adv.get("W_ms", 35.0))
    baseline_s = float(adv.get("baseline_s", 0.06))
    k          = float(adv.get("k", 1.10))
    amp_frac   = float(adv.get("amp_frac", 0.70))

    # 고정 규칙
    hysteresis_ratio = 0.70      # T_low = 0.7 * T_high
    min_event_ms     = 40.0      # 디바운스: 최소 지속시간
    refractory_ms    = 30.0      # 불응기간

    W = max(int(round((W_ms / 1000.0) * fps)), 3)

    def _energy(trace):
        return _moving_rms(np.abs(np.diff(trace, prepend=trace[0])), W)

    onset_series  = df[onset_col].astype(float).values  if onset_col  else total_s
    offset_series = df[offset_col].astype(float).values if offset_col else total_s

    E_on  = _energy(onset_series)
    E_off = _energy(offset_series)

    # baseline 구간 통계 (평균 + k*표준편차)
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

    def _hyst_detect(E, Th, Tl, polarity="rise"):
        above = (E >= Th).astype(int)
        low   = (E >= Tl).astype(int)
        min_frames_ev = max(1, int(round((min_event_ms/1000.0) * fps)))
        refr_frames   = max(1, int(round((refractory_ms/1000.0) * fps)))
        starts, ends = [], []
        i = 0; N = len(E); state = 0
        while i < N:
            if state == 0:
                if i + min_frames_ev <= N and np.all(above[i:i+min_frames_ev] == 1):
                    state = 1
                    starts.append(i)
                    i += min_frames_ev
                    i += refr_frames
                    continue
                i += 1
            else:
                if low[i] == 1:
                    i += 1
                else:
                    ends.append(i)
                    state = 0
                    i += refr_frames
        return np.array(starts, int), np.array(ends, int)

    on_starts, on_ends   = _hyst_detect(E_on,  Th_on,  Tl_on,  "rise")
    off_starts, off_ends = _hyst_detect(E_off, Th_off, Tl_off, "fall")

    # 움직임 시작 i_move
    i_move = int(on_starts[0]) if len(on_starts) else (cycles[0][0] if len(cycles) else None)

    VOnT = np.nan; VOffT = np.nan
    i_steady = None; i_last = None; i_end = None

    if len(cycles) >= 1 and i_move is not None:
        # 전역 진폭
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles])) if cycles else 0.0
        # 첫 steady: move 이후 사이클에서 전역의 amp_frac 이상
        for s, e in cycles:
            if s <= i_move:   # '<=' 가드
                continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp):
                i_steady = int(s); break
        # 최소 간격 4ms 보정
        MIN_VONT_GAP = int(round(0.004 * fps))
        if i_steady is not None and (i_steady - i_move) < MIN_VONT_GAP:
            for s, e in cycles:
                if s <= i_move + MIN_VONT_GAP:
                    continue
                amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
                if g_amp <= 0 or (amp >= amp_frac * g_amp):
                    i_steady = int(s)
                    break
        if i_steady is None:
            i_steady = cycles[0][0] if cycles else i_move
        # 마지막 steady
        for s, e in reversed(cycles):
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp):
                i_last = int(s); break
        if i_last is None:
            i_last = cycles[-1][0] if cycles else (len(t)-1)
        # 움직임 종료
        idxs = np.where(off_ends >= i_last)[0] if len(off_ends) else []
        if len(idxs):
            i_end = int(off_ends[idxs[-1]])
        else:
            i_end = cycles[-1][1] if cycles else (len(t)-1)

        # 시간 계산(ms)
        t_move   = float(t[i_move]) if i_move   is not None else np.nan
        t_steady = float(t[i_steady]) if i_steady is not None else np.nan
        t_last   = float(t[i_last]) if i_last   is not None else np.nan
        t_end    = float(t[min(i_end, len(t)-1)]) if i_end is not None else np.nan

        VOnT  = (t_steady - t_move) * 1000.0 if (np.isfinite(t_steady) and np.isfinite(t_move)) else np.nan
        VOffT = (t_end - t_last)   * 1000.0 if (np.isfinite(t_end) and np.isfinite(t_last)) else np.nan

    # ---- AS/PS 보완 지표 계산 ----
    # 기존 AS(평균 range) 계산 (과거 연속성 유지용)
    AS_legacy = _as_range_median(left_s, right_s, cycles)

    # 새 AS triplet (게인 정규화 포함)
    AS_triplet = _compute_as_triplet(left_s, right_s, total_s, cycles, fps, amp_frac)

    # 새 PS (circular) – sim(1=좋음)과 dist(0=정상,1=비정상) 병기
    PS_sim, PS_dist = _ps_circular(left_s, right_s, t, cycles)

    # summary
    summary = pd.DataFrame({
        "Parameter": [
            "Amplitude Periodicity (AP)",
            "Time Periodicity (TP)",
            "AS (legacy, median p2p)",
            "AS_range (robust)",
            "AS_area (energy)",
            "AS_corr (shape)",
            "PS_sim (1=good)",
            "PS_dist (0=normal)",
            "Voice Onset Time (VOnT, ms)",
            "Voice Offset Time (VOffT, ms)",
        ],
        "Value": [AP, TP, AS_legacy, AS_triplet["AS_range"], AS_triplet["AS_area"], AS_triplet["AS_corr"], PS_sim, PS_dist, VOnT, VOffT]
    })

    per_cycle = pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))

    viz = dict(
        t=t, total_s=total_s, left_s=left_s, right_s=right_s,
        E_on=E_on, E_off=E_off,
        thr_on=Th_on, thr_off=Th_off,
        Tlow_on=Tl_on, Tlow_off=Tl_off,
        i_move=i_move, i_steady=i_steady, i_last=i_last, i_end=(i_end if 'i_end' in locals() else None),
        cycles=cycles,
        AS_triplet=AS_triplet,
        PS_sim=PS_sim, PS_dist=PS_dist
    )
    extras = dict(fps=fps, n_cycles=len(cycles), viz=viz)
    return summary, per_cycle, extras

# ============== 사이드바 세팅 ==============
with st.sidebar:
    st.markdown("### ⚙ Settings")
    baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.06, step=0.01)
    k          = st.number_input("임계 배수 k",      min_value=0.50, max_value=6.00, value=1.10, step=0.10)
    M          = st.number_input("연속 프레임 M (참고용)", min_value=1, max_value=150, value=40, step=1)
    W_ms       = st.number_input("에너지 창(ms)",     min_value=2.0,  max_value=40.0, value=35.0, step=1.0)
    amp_frac   = st.slider("정상화 최소 진폭 비율", 0.10, 0.90, 0.70, 0.01)

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

# ============== 파일 업로드 ==============
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요", type=["csv", "xlsx"])
if uploaded is None:
    st.info("⬆️ 분석할 파일을 업로드하세요.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# ============== 분석 실행 ==============
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
AS_triplet= viz.get("AS_triplet", {})
PS_sim    = viz.get("PS_sim", np.nan)
PS_dist   = viz.get("PS_dist", np.nan)

# 값 추출 헬퍼
_def_get = lambda key, default=np.nan: float(summary.loc[summary["Parameter"] == key, "Value"].iloc[0]) if (key in summary["Parameter"].values) else default

AP    = _def_get("Amplitude Periodicity (AP)")
TP    = _def_get("Time Periodicity (TP)")
AS_lg = _def_get("AS (legacy, median p2p)")
AS_rg = _def_get("AS_range (robust)")
AS_ar = _def_get("AS_area (energy)")
AS_co = _def_get("AS_corr (shape)")
VOnT  = _def_get("Voice Onset Time (VOnT, ms)")
VOffT = _def_get("Voice Offset Time (VOffT, ms)")
fps   = float(extras.get("fps", np.nan))
ncyc  = int(extras.get("n_cycles", 0))

# ============== 그래프 빌더 ==============

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
                fig.add_annotation(x=xval, y=float(np.nanmax(total_s)), text=label,
                                   showarrow=False, font=dict(size=10, color=col), yshift=14)

    if zoom == "0–0.2s":   fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])

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
        fig.update_layout(template="simple_white", height=340)
        return fig

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

    if zoom == "0–0.2s":   fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Left vs Right (AS_range {AS_rg if np.isfinite(AS_rg) else np.nan:.2f} · PS_dist {PS_dist if np.isfinite(PS_dist) else np.nan:.2f})",
        xaxis_title="Time (s)",
        yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"),
        template="simple_white", height=340,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig


def make_energy_plot(mode="on", show_markers=True, zoom="전체"):
    fig = go.Figure()
    if t is None:
        fig.update_layout(template="simple_white", height=320)
        return fig
    if mode == "on":
        E, Th, Tl, color, label, event_idx = E_on, thr_on, Tlow_on, COLOR_CRIMSON, "Onset", i_move
    else:
        E, Th, Tl, color, label, event_idx = E_off, thr_off, Tlow_off, COLOR_ROYAL, "Offset", i_end

    if E is not None:
        fig.add_trace(go.Scatter(x=t, y=E, name=f"E_{label.lower()}",
                                 mode="lines", line=dict(color=color, width=2.0)))
    if Th is not None:
        fig.add_hline(y=float(Th), line=dict(color=color, width=1.5),
                      annotation_text=f"thr_{label.lower()}", annotation_position="top left")
    if Tl is not None:
        fig.add_hline(y=float(Tl), line=dict(color=color, dash="dot", width=1.2),
                      annotation_text=f"Tlow_{label.lower()}", annotation_position="bottom left")

    if show_markers and event_idx is not None and 0 <= int(event_idx) < len(t):
        xval = t[int(event_idx)]
        fig.add_vline(x=xval, line=dict(color=color, dash="dot", width=1.6))
        if E is not None:
            fig.add_annotation(x=xval, y=float(np.nanmax(E)), text=f"{label} @ {xval*1000.0:.2f} ms",
                               showarrow=False, font=dict(size=10, color=color), yshift=14)

    if zoom == "0–0.2s":   fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])

    fig.update_layout(
        title=f"Energy & Thresholds – {label}",
        xaxis_title="Time (s)", yaxis_title="Energy (a.u.)",
        template="simple_white", height=320,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig

# ============== 탭 ==============
tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Validation"])

with tab1:
    st.subheader("🩺 Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("AP", f"{AP:.4f}")
    c2.metric("TP", f"{TP:.4f}")
    c3.metric("PS_dist (0=정상)", f"{PS_dist:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("AS_range", f"{AS_rg:.4f}")
    c5.metric("AS_area", f"{AS_ar:.4f}")
    c6.metric("AS_corr", f"{AS_co:.4f}")

    st.caption(f"FPS: {fps:.1f} | 검출된 사이클 수: {ncyc}")
    st.dataframe(summary, use_container_width=True)

with tab2:
    st.subheader("📈 Visualization")
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    show_cycles   = cc1.checkbox("Cycle 밴드 표시", True)
    show_markers  = cc2.checkbox("이벤트 마커 표시", True)
    zoom_preset   = cc3.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"])
    normalize_lr  = cc4.checkbox("좌/우 정규화 시각화", False)
    energy_mode   = cc5.radio("에너지 뷰", ["Onset", "Offset"], horizontal=True)

    st.markdown("#### A) Total")
    st.plotly_chart(make_total_plot(show_cycles, show_markers, zoom_preset), use_container_width=True)

    st.markdown("#### B) Left vs Right")
    st.plotly_chart(make_lr_plot(normalize_lr, zoom_preset), use_container_width=True)

    st.markdown("#### C) Energy + Thresholds")
    st.plotly_chart(make_energy_plot("on" if energy_mode == "Onset" else "off",
                                     show_markers, zoom_preset), use_container_width=True)

with tab3:
    st.subheader("📊 Validation (참고)")
    st.info("v3.1에서 Batch Validation & RMSE 집계가 확장됩니다. (AS/PS 보완 지표 포함)")

