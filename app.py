
# ---------------------------------------------------------------
# HSV Auto Analyzer v3-alpha – Adaptive Clinical Engine (Merged)
# Isaka × Lian – app_v3alpha_overview_fix.py
# 실행: streamlit run app_v3alpha_overview_fix.py
# 요구: streamlit, plotly, pandas, numpy, (optional) scipy
# ---------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# optional savgol
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

st.set_page_config(page_title="HSV Auto Analyzer v3-alpha – Adaptive Clinical Engine",
                   layout="wide")
st.title("HSV Auto Analyzer v3-alpha – Adaptive Clinical Engine (Merged)")
st.caption("Isaka × Lian | v2.5 engine + v3 PS/AS metrics + Overview fix")

# ============== Colors ==============
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

# ============== Utils ==============
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

# ============== v3 Metrics ==============
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

def _as_legacy(left: np.ndarray, right: np.ndarray, cycles: list) -> float:
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    ratios = []
    for s, e in cycles:
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        m = max(L, R)
        ratios.append((min(L, R) / m) if m > 0 else np.nan)
    return _clamp01(_nanmean0(ratios))

def _ps_dist(left: np.ndarray, right: np.ndarray, t: np.ndarray, cycles: list) -> tuple:
    """Phase distance (0=good). Also returns PS_sim=1-PS_dist."""
    if left is None or right is None or len(cycles) < 1:
        return (np.nan, np.nan)
    dists = []
    for s, e in cycles:
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s]) if (t is not None) else 1.0
        if Ti <= 0: 
            continue
        dt = abs(float(t[li] - t[ri]))
        d = min(dt, Ti - dt) / Ti  # circular wrap
        dists.append(min(1.0, d))
    if not len(dists): 
        return (np.nan, np.nan)
    dist = _clamp01(_nanmean0(dists))
    return dist, _clamp01(1.0 - dist)

def _as_gain_normalize(left: np.ndarray, right: np.ndarray, cycles: list):
    """Return normalized L,R based on steady cycles median gain (robust)."""
    if left is None or right is None or len(cycles) < 1:
        return None, None
    # collect per-cycle p2p
    p2pL, p2pR = [], []
    for s,e in cycles:
        p2pL.append(float(np.nanmax(left[s:e]) - np.nanmin(left[s:e])))
        p2pR.append(float(np.nanmax(right[s:e]) - np.nanmin(right[s:e])))
    gL = np.nanmedian(p2pL) if len(p2pL) else np.nan
    gR = np.nanmedian(p2pR) if len(p2pR) else np.nan
    if not (np.isfinite(gL) and np.isfinite(gR)) or (gL <= 0 or gR <= 0):
        return left, right
    L = left / (gL + 1e-12)
    R = right / (gR + 1e-12)
    return L, R

def _as_range_area_corr(left: np.ndarray, right: np.ndarray, cycles: list) -> tuple:
    """AS_range (robust), AS_area (energy), AS_corr (shape)"""
    if left is None or right is None or len(cycles) < 1:
        return (np.nan, np.nan, np.nan)
    L, R = _as_gain_normalize(left, right, cycles)
    if L is None or R is None:
        return (np.nan, np.nan, np.nan)
    ranges = []
    areas  = []
    corrs  = []
    for s,e in cycles:
        l = L[s:e]; r = R[s:e]
        # range
        rL = float(np.nanmax(l) - np.nanmin(l)); rR = float(np.nanmax(r) - np.nanmin(r))
        denom = max(rL, rR, 1e-12)
        ranges.append((min(rL, rR) / denom))
        # area (L2 energy)
        aL = float(np.nansum((l - np.nanmean(l))**2))
        aR = float(np.nansum((r - np.nanmean(r))**2))
        denomA = max(aL, aR, 1e-12)
        areas.append(min(aL, aR)/denomA)
        # corr
        if np.nanstd(l) < 1e-12 or np.nanstd(r) < 1e-12:
            corrs.append(np.nan)
        else:
            lc = (l - np.nanmean(l)) / (np.nanstd(l) + 1e-12)
            rc = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)
            c = float(np.nanmean(lc * rc))
            corrs.append(max(-1.0, min(1.0, c)))
    return (_clamp01(_nanmean0(ranges)),
            _clamp01(_nanmean0(areas)),
            max(-1.0, min(1.0, _nanmean0(corrs))))

# ============== v2.5 Engine: analyze ==============
def analyze(df: pd.DataFrame, adv: dict):
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

    # fps
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0

    # smoothing
    total_s = _smooth(total, fps)
    left_s  = _smooth(left, fps)  if left  is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    # cycles
    min_frames = max(int(0.002 * fps), 5)
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    # core metrics
    AP, TP = _ap_tp(t, total_s, cycles)
    AS_legacy = _as_legacy(left_s, right_s, cycles)
    PS_dist, PS_sim = _ps_dist(left_s, right_s, t, cycles)
    AS_range, AS_area, AS_corr = _as_range_area_corr(left_s, right_s, cycles)

    # energy-based on/offset (v2.5 style)
    W_ms       = float(adv.get("W_ms", 35.0))
    baseline_s = float(adv.get("baseline_s", 0.06))
    k          = float(adv.get("k", 1.10))
    amp_frac   = float(adv.get("amp_frac", 0.70))

    hysteresis_ratio = 0.70      # T_low = 0.7 * T_high
    min_event_ms     = 40.0      # debounce
    refractory_ms    = 30.0      # refractory

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
                    state = 1; starts.append(i)
                    i += min_frames_ev; i += refr_frames; continue
                i += 1
            else:
                if low[i] == 1:
                    i += 1
                else:
                    ends.append(i); state = 0; i += refr_frames
        return np.array(starts, int), np.array(ends, int)

    on_starts, on_ends   = _hyst_detect(E_on,  Th_on,  Tl_on)
    off_starts, off_ends = _hyst_detect(E_off, Th_off, Tl_off)

    i_move = int(on_starts[0]) if len(on_starts) else (cycles[0][0] if len(cycles) else None)

    VOnT = np.nan
    VOffT = np.nan
    if len(cycles) >= 1 and i_move is not None:
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles])) if cycles else 0.0
        # first steady after move
        i_steady = None
        for s, e in cycles:
            if s <= i_move:   # ensure after move
                continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp):
                i_steady = int(s); break
        MIN_VONT_GAP = int(round(0.004 * fps))
        if i_steady is not None and (i_steady - i_move) < MIN_VONT_GAP:
            for s, e in cycles:
                if s <= i_move + MIN_VONT_GAP:
                    continue
                amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
                if g_amp <= 0 or (amp >= amp_frac * g_amp):
                    i_steady = int(s); break
        if i_steady is None:
            i_steady = cycles[0][0] if cycles else i_move

        # last steady
        i_last = None
        for s, e in reversed(cycles):
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac * g_amp):
                i_last = int(s); break
        if i_last is None:
            i_last = cycles[-1][0] if cycles else (len(t)-1)

        # end
        idxs = np.where(off_ends >= i_last)[0] if len(off_ends) else []
        if len(idxs):
            i_end = int(off_ends[idxs[-1]])
        else:
            i_end = cycles[-1][1] if cycles else (len(t)-1)

        t_move   = float(t[i_move]) if i_move   is not None else np.nan
        t_steady = float(t[i_steady]) if i_steady is not None else np.nan
        t_last   = float(t[i_last]) if i_last   is not None else np.nan
        t_end    = float(t[min(i_end, len(t)-1)]) if i_end is not None else np.nan

        VOnT  = (t_steady - t_move) * 1000.0 if (np.isfinite(t_steady) and np.isfinite(t_move)) else np.nan
        VOffT = (t_end - t_last)   * 1000.0 if (np.isfinite(t_end) and np.isfinite(t_last)) else np.nan
    else:
        i_steady = None; i_last = None; i_end = None

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
        "Value": [AP, TP, AS_legacy, AS_range, AS_area, AS_corr, PS_sim, PS_dist, VOnT, VOffT]
    })

    viz = dict(
        t=t, total_s=total_s, left_s=left_s, right_s=right_s,
        E_on=E_on, E_off=E_off,
        thr_on=Th_on, thr_off=Th_off,
        Tlow_on=Tl_on, Tlow_off=Tl_off,
        i_move=i_move, i_steady=i_steady, i_last=i_last, i_end=(locals().get("i_end", None)),
        cycles=cycles,
        AP=AP, TP=TP, AS_legacy=AS_legacy, AS_range=AS_range, AS_area=AS_area, AS_corr=AS_corr,
        PS_sim=PS_sim, PS_dist=PS_dist, VOnT=VOnT, VOffT=VOffT
    )
    extras = dict(fps=fps, n_cycles=len(cycles), viz=viz)
    return summary, pd.DataFrame(dict(cycle=[], start_time=[], end_time=[])), extras

# ============== Overview Renderer ==============
DEFAULT_KEYS = ["AP","TP","PS_dist","AS_corr","AS_range","AS_area","VOnT","VOffT"]

def _val(x, ndig=4):
    try:
        if x is None: return "N/A"
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "N/A"
        return f"{xf:.{ndig}f}"
    except Exception:
        return "N/A"

def render_overview(env: dict, keys=None):
    st.subheader("🩺 Overview")
    AP       = env.get("AP")
    TP       = env.get("TP")
    PS_dist  = env.get("PS_dist")
    AS_corr  = env.get("AS_corr")
    AS_range = env.get("AS_range")
    AS_area  = env.get("AS_area")
    VOnT     = env.get("VOnT")
    VOffT    = env.get("VOffT")
    fps      = env.get("fps", np.nan)
    ncyc     = int(env.get("ncyc", 0) or 0)

    metrics = {
        "AP":       _val(AP, 4),
        "TP":       _val(TP, 4),
        "PS_dist":  _val(PS_dist, 4),
        "AS_corr":  _val(AS_corr, 4),
        "AS_range": _val(AS_range, 4),
        "AS_area":  _val(AS_area, 4),
        "VOnT":     _val(VOnT, 2),
        "VOffT":    _val(VOffT, 2),
    }
    labels = {
        "AP":       "AP",
        "TP":       "TP",
        "PS_dist":  "PS_dist (0=정상)",
        "AS_corr":  "AS_corr",
        "AS_range": "AS_range",
        "AS_area":  "AS_area",
        "VOnT":     "VOnT (ms)",
        "VOffT":    "VOffT (ms)",
    }

    if keys is None:
        sel = st.multiselect("표시 항목", DEFAULT_KEYS,
                             default=st.session_state.get("overview_keys", DEFAULT_KEYS))
        st.session_state["overview_keys"] = sel
        keys = sel
    else:
        st.session_state["overview_keys"] = keys

    row1 = keys[:4]
    row2 = keys[4:8]

    c1 = st.columns(len(row1)) if row1 else []
    for i,k in enumerate(row1):
        with c1[i]:
            st.metric(labels[k], metrics[k])
    c2 = st.columns(len(row2)) if row2 else []
    for i,k in enumerate(row2):
        with c2[i]:
            st.metric(labels[k], metrics[k])

    qc = []
    try:
        if isinstance(PS_dist, (int,float)) and np.isfinite(PS_dist) and PS_dist > 0.08:
            qc.append("PS_dist↑ (위상 불일치 가능)")
        if isinstance(AP, (int,float)) and np.isfinite(AP) and AP < 0.70:
            qc.append("AP 낮음 (진폭 불안정)")
        if isinstance(TP, (int,float)) and np.isfinite(TP) and TP < 0.85:
            qc.append("TP 낮음 (주기 불안정)")
        if isinstance(VOnT, (int,float)) and np.isfinite(VOnT) and VOnT <= 0:
            qc.append("VOnT≤0 (가드 재확인)")
    except Exception:
        pass

    st.caption(f"FPS: {np.nan if not np.isfinite(fps) else round(float(fps),1)} | 검출된 사이클 수: {ncyc}")
    if qc:
        st.info("QC: " + " · ".join(qc))

# ============== Sidebar Settings ==============
with st.sidebar:
    st.markdown("### ⚙ Settings")
    prof = st.selectbox("분석 프로필", ["Normal", "ULP", "SD", "Custom"], index=0)
    # defaults by profile
    pmap = {
        "Normal": dict(baseline_s=0.06, k=1.10, M=40, W_ms=35.0, amp_frac=0.70),
        "ULP":    dict(baseline_s=0.06, k=1.50, M=40, W_ms=35.0, amp_frac=0.60),
        "SD":     dict(baseline_s=0.06, k=1.75, M=50, W_ms=40.0, amp_frac=0.75),
        "Custom": dict(baseline_s=0.06, k=1.10, M=40, W_ms=35.0, amp_frac=0.70),
    }
    base = pmap.get(prof, pmap["Normal"])
    baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=float(base["baseline_s"]), step=0.01)
    k          = st.number_input("임계 배수 k",      min_value=0.50, max_value=6.00,  value=float(base["k"]), step=0.10)
    M          = st.number_input("연속 프레임 M (참고용)", min_value=1, max_value=150, value=int(base["M"]), step=1)
    W_ms       = st.number_input("에너지 창(ms)",     min_value=2.0,  max_value=60.0,  value=float(base["W_ms"]), step=1.0)
    amp_frac   = st.slider("정상화 최소 진폭 비율", 0.10, 0.90, float(base["amp_frac"]), 0.01)
    st.caption("프로필은 기본값을 로드만 하며, 개별 슬라이더로 즉시 미세 조정할 수 있습니다.")

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac)

# ============== File Upload ==============
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요", type=["csv", "xlsx"])
if uploaded is None:
    st.info("⬆️ 분석할 파일을 업로드하세요.")
    st.stop()

if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# ============== Run analysis ==============
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
AP = viz.get("AP"); TP = viz.get("TP"); AS_legacy = viz.get("AS_legacy")
AS_range = viz.get("AS_range"); AS_area = viz.get("AS_area"); AS_corr = viz.get("AS_corr")
PS_sim = viz.get("PS_sim"); PS_dist = viz.get("PS_dist"); VOnT = viz.get("VOnT"); VOffT = viz.get("VOffT")
fps  = float(extras.get("fps", np.nan))
ncyc = int(extras.get("n_cycles", 0))

# ============== Plots ==============
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
    fig.update_layout(title="Total Signal with Detected Events",
                      xaxis_title="Time (s)", yaxis_title="Gray Level (a.u.)",
                      template="simple_white", height=380,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
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
    fig.update_layout(title=f"Left vs Right (AS_range {AS_range:.2f} · AS_corr {AS_corr:.2f})",
                      xaxis_title="Time (s)",
                      yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"),
                      template="simple_white", height=340,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
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
    fig.update_layout(title=f"Energy & Thresholds – {label}",
                      xaxis_title="Time (s)", yaxis_title="Energy (a.u.)",
                      template="simple_white", height=320,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig

# ============== Tabs ==============
tab1, tab2, tab3 = st.tabs(["Overview", "Visualization", "Validation"])

with tab1:
    env = dict(AP=AP, TP=TP, PS_dist=PS_dist, AS_corr=AS_corr, AS_range=AS_range,
               AS_area=AS_area, VOnT=VOnT, VOffT=VOffT, fps=fps, ncyc=ncyc)
    render_overview(env)
    st.dataframe(summary, use_container_width=True)

with tab2:
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
    st.info("자동 vs 수동 측정치 정량검증은 다음 업데이트에서 확장됩니다. (배치 집계, Bias 히스토그램 포함)")
