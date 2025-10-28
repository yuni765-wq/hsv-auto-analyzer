# ---------------------------------------------------------------
# HSV Auto Analyzer v3-alpha – Adaptive Clinical Engine (Full)
# Isaka × Lian – app_v3alpha_full_with_ParamTab.py
# 실행: streamlit run app_v3alpha_full_with_ParamTab.py
# 요구: streamlit, plotly, pandas, numpy, (optional) scipy
# ---------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# optional savgol
try:
    from scipy.signal import savgol_filter
    from scipy import stats
    _HAS_SAVGOL = True
    _HAS_SCIPY = True
except Exception:
    _HAS_SAVGOL = False
    _HAS_SCIPY = False
from insight_v32 import (
    VERSION_V32, compute_quality_from_env, render_quality_banner, inject_css
)
from metrics import (
    compute_envelope,
    detect_gat_vont_got_vofft,
    compute_oid,
    tremor_index_psd,
)
REQUIRED_FUNCS = [
    compute_envelope,
    detect_gat_vont_got_vofft,
    compute_oid,
    tremor_index_psd,
]
assert all(callable(f) for f in REQUIRED_FUNCS), "metrics functions not loaded"

# -------------------- Global fixed settings (N–D′) --------------------
AMP_FRAC_OFFSET_FIXED = 0.80      # <- Offset 전용 amp_frac 고정
K_ND_PRIME            = 1.40
M_ND_PRIME            = 6
WINDOW_ENERGY_MS      = 40.0
AS_CORR_MAX_ND        = 0.88
PS_DIST_MIN_ND        = 0.10
AP_MAX_ND             = 0.82
TP_MAX_ND             = 0.88
MAIN_SUSTAIN_FR       = 90
AUX_SUSTAIN_FR        = 45
DEBOUNCE_FR           = 20

# === v3.2 페이지 타이틀/버전 라벨 ===
VERSION_LABEL = VERSION_V32
st.set_page_config(page_title=VERSION_LABEL, layout="wide")
st.title(VERSION_LABEL)
st.caption("Isaka × Lian | Stable preset + Stats auto-load + Quality indicator + Clinical notes + Pinned banner")

# v3.2 고정배너용 CSS
inject_css(st)

# 간단 런 로그
if "run_log" not in st.session_state:
    st.session_state["run_log"] = []
st.session_state["run_log"].append(f"RUN_LOG: Preset Loaded = Stable_v3.1")

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
COLOR_AUTOON  = "#8B008B"
COLOR_AUTOOFF = "#1E90FF"

# ===============================================================
# 0) DualDetector — Onset/Offset 분리 상태기계 (간단 내장)
# ===============================================================
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class OnsetConfig:
    theta: float = 0.50
    min_amp_frac: float = 0.58
    AP_min: float = 0.85
    TP_min: float = 0.90
    AS_corr_min: float = 0.95
    PS_dist_max: float = 0.05
    sustain_frames: int = 5

@dataclass
class OffsetConfig:
    AS_corr_max: float = AS_CORR_MAX_ND
    PS_dist_min: float = PS_DIST_MIN_ND
    AP_max: float = AP_MAX_ND
    TP_max: float = TP_MAX_ND
    main_sustain_frames: int = MAIN_SUSTAIN_FR
    aux_sustain_frames: int = AUX_SUSTAIN_FR
    debounce_frames: int = DEBOUNCE_FR
    hysteresis_delta: float = 0.10

@dataclass
class DetectorConfig:
    frame_ms: float = 0.66
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    offset: OffsetConfig = field(default_factory=OffsetConfig)

class DualDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    def detect(self, feats: Dict[str, np.ndarray]) -> Dict[str, Any]:
        t = feats["t_ms"].astype(float)
        A = feats["A_norm"].astype(float)
        AP = feats["AP"].astype(float)
        TP = feats["TP"].astype(float)
        AS = feats["AS_corr"].astype(float)
        PS = feats["PS_dist"].astype(float)

        N = len(t)
        on_cnt = 0
        main_cnt = 0
        aux_cnt = 0
        on_time: Optional[float] = None
        off_time: Optional[float] = None
        off_idx_candidate: Optional[int] = None

        oc = self.cfg.onset
        fc = self.cfg.offset
        state = "PRE_ONSET"

        def main_flag(i: int) -> bool:
            return (AS[i] < fc.AS_corr_max) or (PS[i] > fc.PS_dist_min)

        def debounce_ok(k: int) -> bool:
            L = min(fc.debounce_frames, N - k)
            if L <= 0:
                return False
            for j in range(k, k+L):
                if not main_flag(j):
                    return False
            return True

        for i in range(N):
            on_flag = (
                (A[i] >= oc.theta) and
                (A[i] >= oc.min_amp_frac) and
                (AP[i] >= oc.AP_min) and
                (TP[i] >= oc.TP_min) and
                (AS[i] >= oc.AS_corr_min) and
                (PS[i] <= oc.PS_dist_max)
            )
            on_cnt = on_cnt + 1 if on_flag else 0

            f_main = main_flag(i)
            f_aux = (AP[i] < fc.AP_max) or (TP[i] < fc.TP_max)
            main_cnt = main_cnt + 1 if f_main else 0
            aux_cnt  = aux_cnt  + 1 if f_aux  else 0

            if state == "PRE_ONSET":
                if on_cnt >= oc.sustain_frames:
                    on_time = t[i]
                    state = "VOICED"

            elif state == "VOICED":
                if (main_cnt >= fc.main_sustain_frames) and (aux_cnt >= fc.aux_sustain_frames):
                    if off_idx_candidate is None:
                        off_idx_candidate = i
                    if debounce_ok(off_idx_candidate):
                        off_time = t[off_idx_candidate]
                        state = "POST_OFFSET"
                        break

        if on_time is None and N > 0:
            on_time = float(t[0])
        if off_time is None and N > 0:
            last = N - 1
            back = max(0, last - int(round(100.0 / max(1e-9, self.cfg.frame_ms))))
            k_pick = None
            for k in range(last, back, -1):
                fm = (AS[k] < fc.AS_corr_max) or (PS[k] > fc.PS_dist_min)
                fa = (AP[k] < fc.AP_max) or (TP[k] < fc.TP_max)
                if fm and fa:
                    k_pick = k
                    break
            off_time = float(t[k_pick]) if k_pick is not None else float(max(0.0, t[last] - 10.0))

        duration_ms = None
        if (on_time is not None) and (off_time is not None):
            duration_ms = float(off_time - on_time)

        return {
            "onset_time_ms": float(on_time) if on_time is not None else None,
            "offset_time_ms": float(off_time) if off_time is not None else None,
            "duration_ms": duration_ms,
        }

# -------------------- small utils --------------------
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
    base_w = int(max(7, min(21, round(fps * 0.007))))
    win = base_w if (base_w % 2 == 1) else base_w + 1
    win = min(win, n - 1) if n % 2 == 0 and win >= n else min(win, n - (1 - (n % 2)))
    if _HAS_SAVGOL:
        try:
            return savgol_filter(signal.astype(float), window_length=win, polyorder=3, mode="interp")
        except Exception:
            pass
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
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    return float(max(0.0, min(1.0, x)))

def _nanmean0(x):
    v = np.nanmean(x) if len(x) else np.nan
    return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

# -------------------- v3 Metrics --------------------
def _ap_tp(t: np.ndarray, total: np.ndarray, cycles: list) -> tuple:
    if len(cycles) < 3:
        return (np.nan, np.nan)
    amps, periods = [], []
    for s, e in cycles:
        seg = total[s:e]
        amp = float(np.nanmax(seg) - np.nanmin(seg))
        Ti = float(t[e] - t[s])
        amps.append(amp); periods.append(max(Ti, 1e-9))
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
        d = min(dt, Ti - dt) / Ti
        dists.append(min(1.0, d))
    if not len(dists): 
        return (np.nan, np.nan)
    dist = _clamp01(_nanmean0(dists))
    return dist, _clamp01(1.0 - dist)

def _as_gain_normalize(left: np.ndarray, right: np.ndarray, cycles: list):
    if left is None or right is None or len(cycles) < 1:
        return None, None
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
    if left is None or right is None or len(cycles) < 1:
        return (np.nan, np.nan, np.nan)
    L, R = _as_gain_normalize(left, right, cycles)
    if L is None or R is None:
        return (np.nan, np.nan, np.nan)
    ranges = []; areas = []; corrs = []
    for s,e in cycles:
        l = L[s:e]; r = R[s:e]
        rL = float(np.nanmax(l) - np.nanmin(l)); rR = float(np.nanmax(r) - np.nanmin(r))
        denom = max(rL, rR, 1e-12)
        ranges.append((min(rL, rR) / denom))
        aL = float(np.nansum((l - np.nanmean(l))**2))
        aR = float(np.nansum((r - np.nanmean(r))**2))
        denomA = max(aL, aR, 1e-12)
        areas.append(min(aL, aR)/denomA)
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

# -------------------- Analyzer (v2.5 energy + v3 metrics) --------------------
def analyze(df: pd.DataFrame, adv: dict):
    # 0) 입력 컬럼 정리
    cols = _norm_cols(df.columns.tolist())
    df = df.copy()
    df.columns = cols

    def pick(key):
        for c in cols:
            if key in c:
                return c
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

    # 1) 시계열 준비 (초 단위)
    t = df[time_col].astype(float).values
    if np.nanmax(t) > 10.0:
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

    # 2) fps
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0

    # 3) smoothing
    total_s = _smooth(total, fps)
    left_s  = _smooth(left, fps)  if left  is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    # 4) cycles (피크 구간)
    min_frames = max(int(0.002 * fps), 5)
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    # 5) v3 메트릭 (AP/TP/AS/PS)
    AP, TP = _ap_tp(t, total_s, cycles)
    AS_legacy = _as_legacy(left_s, right_s, cycles)
    PS_dist, PS_sim = _ps_dist(left_s, right_s, t, cycles)
    AS_range, AS_area, AS_corr = _as_range_area_corr(left_s, right_s, cycles)

    # 6) v2.5 energy 기반 onset/offset 탐지 파이프
    W_ms       = float(adv.get("W_ms", WINDOW_ENERGY_MS))
    baseline_s = float(adv.get("baseline_s", 0.06))
    k          = float(adv.get("k", K_ND_PRIME))
    amp_frac_on  = float(adv.get("amp_frac_on", 0.70))          # Onset 전용
    amp_frac_off = AMP_FRAC_OFFSET_FIXED                         # Offset 전용(고정)

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

    # 7) steady 구간 기반 VOnT/VOffT 계산
    i_move = int(on_starts[0]) if len(on_starts) else (cycles[0][0] if len(cycles) else None)
    VOnT = np.nan; VOffT = np.nan
    i_steady = None; i_last = None; i_end = None

    if len(cycles) >= 1 and i_move is not None:
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles])) if cycles else 0.0

        # onset: 움직임 이후 첫 steady (amp_frac_on)
        for s, e in cycles:
            if s <= i_move:
                continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac_on * g_amp):
                i_steady = int(s); break

        # offset: 마지막 steady (amp_frac_off)
        for s, e in reversed(cycles):
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp <= 0 or (amp >= amp_frac_off * g_amp):
                i_last = int(s); break
        if i_last is None:
            i_last = cycles[-1][0] if cycles else (len(t) - 1)

        # offset 끝점: energy 기반 종료점
        idxs = np.where(off_ends >= i_last)[0] if len(off_ends) else []
        i_end = int(off_ends[idxs[-1]]) if len(idxs) else (cycles[-1][1] if cycles else (len(t) - 1))

        t_move   = float(t[i_move])   if i_move   is not None else np.nan
        t_steady = float(t[i_steady]) if i_steady is not None else np.nan
        t_last   = float(t[i_last])   if i_last   is not None else np.nan
        t_end    = float(t[min(i_end, len(t)-1)]) if i_end is not None else np.nan

        VOnT  = (t_steady - t_move) * 1000.0 if (np.isfinite(t_steady) and np.isfinite(t_move)) else np.nan
        VOffT = (t_end - t_last)   * 1000.0 if (np.isfinite(t_end)    and np.isfinite(t_last)) else np.nan

    # 8) v3.2 envelope 기반 GAT/GOT + OID + Tremor
    err_msgs = []

        # 안전한 숫자 판별 헬퍼
    def _is_num(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

    # 8-1) envelope
    try:
        env_v32 = compute_envelope(total_s, fps)
        env_v32 = np.nan_to_num(env_v32, nan=0.0)
    except Exception as e:
        env_v32 = None
        err_msgs.append(f"[env] {type(e).__name__}: {e}")

    # 8-2) GAT/GOT/VOnT_env/VOffT_env
    try:
        if env_v32 is not None:
            gat_ms, vont_ms_env, got_ms, vofft_ms = detect_gat_vont_got_vofft(
                env_v32, fps, k=1.0, min_run_ms=12, win_cycles=3, cv_max=0.25
            )
        else:
            gat_ms = got_ms = vont_ms_env = vofft_ms = np.nan
    except Exception as e:
        gat_ms = got_ms = vont_ms_env = vofft_ms = np.nan
        err_msgs.append(f"[detect] {type(e).__name__}: {e}")
            # 8-2b) 유효성 검사 + 폴백 적용
    if not _is_num(gat_ms):
        if _is_num(vont_ms_env):
            gat_ms = float(vont_ms_env)
        elif _is_num(VOnT):
            gat_ms = float(VOnT)
        else:
            gat_ms = np.nan

    if not _is_num(got_ms):
        if _is_num(vofft_ms):
            got_ms = float(vofft_ms)
        elif _is_num(VOffT):
            got_ms = float(VOffT)
        else:
            got_ms = np.nan

    # ---- GAT fallback (when None/NaN) ------------------------------
    # 빈칸 방지: VOnT_env → VOnT 순으로 보수적 대체
    if not _is_num(gat_ms):
        if np.isfinite(vont_ms_env):
            gat_ms = float(vont_ms_env)
            err_msgs.append("[GAT] fallback → VOnT_env")
        elif np.isfinite(VOnT):
            gat_ms = float(VOnT)
            err_msgs.append("[GAT] fallback → VOnT")
        else:
            gat_ms = np.nan
            err_msgs.append("[GAT] unavailable")
    # ----------------------------------------------------------------


# ===============================================
# compute_oid : OID = VOffT_env − GOT (ms)
# ===============================================
def compute_oid(got_ms, vofft_ms):
    if not _is_num(got_ms) or not _is_num(vofft_ms):
        return np.nan
    # 필요 시 한 번 더 float 캐스팅
    got = float(got_ms)
    off = float(vofft_ms)
    return off - got

    # 8-3) OID
    try:
        oid_ms = compute_oid(got_ms, vofft_ms)   # compute_oid는 _is_num 사용 버전
    except Exception as e:
        oid_ms = np.nan
        err_msgs.append(f"[oid] {type(e).__name__}: {e}")

    # 8-4) TremorIndex (안정성 버전)
    try:
        if env_v32 is not None:
            from scipy.signal import welch
            sig = np.asarray(env_v32, float)
            sig = np.nan_to_num(sig, nan=0.0)
            L = int(sig.size)
            if L < 64:
                tremor_ratio = np.nan
                err_msgs.append("[tremor] short signal (<64 samples)")
            else:
                import math
                target_len = max(32, int(fps * 0.35))
                win_pow    = int(math.log2(max(32, min(L, target_len))))
                nperseg    = max(32, min(2 ** win_pow, L))
                noverlap   = min(nperseg // 2, nperseg - 1, max(0, L // 4))
                if noverlap >= nperseg:
                    noverlap = max(0, nperseg // 2 - 1)
                f, Pxx = welch(sig, fs=fps, nperseg=nperseg, noverlap=noverlap)
                tgt = ((f >= 4.0) & (f <= 5.0))
                tot = ((f >= 1.0) & (f <= 20.0))
                num = np.trapz(Pxx[tgt], f[tgt]) if np.any(tgt) else 0.0
                den = np.trapz(Pxx[tot], f[tot]) if np.any(tot) else 0.0
                tremor_ratio = (num / den) if (den > 0 and np.isfinite(num)) else np.nan
        else:
            tremor_ratio = np.nan
    except Exception as e:
        tremor_ratio = np.nan
        err_msgs.append(f"[tremor] {type(e).__name__}: {e}")

    # 9) 결과표 구성 ------------------------------------------------------------
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
            "GAT (ms)", "GOT (ms)",
            "VOnT_env (ms)", "VOffT_env (ms)",
            "OID = VOffT_env − GOT (ms)",
            "Tremor Index (4–5 Hz, env)"
        ],
        "Value": [
            AP, TP, AS_legacy, AS_range, AS_area, AS_corr,
            PS_sim, PS_dist,
            VOnT, VOffT,
            gat_ms, got_ms,
            vont_ms_env, vofft_ms,
            oid_ms, tremor_ratio
        ]
    })

    # 표시 포맷: NaN/None → "N/A" (표시 전용)
    def _fmt(v):
        return "N/A" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v
    summary["Value"] = [_fmt(v) for v in summary["Value"]]

    # 10) viz 패킷 -------------------------------------------------------------
    viz = dict(
        t=t, total_s=total_s, left_s=left_s, right_s=right_s,
        E_on=E_on, E_off=E_off,
        thr_on=Th_on, thr_off=Th_off, Tlow_on=Tl_on, Tlow_off=Tl_off,
        i_move=i_move, i_steady=i_steady, i_last=i_last, i_end=i_end,
        cycles=cycles,
        AP=AP, TP=TP,
        AS_legacy=AS_legacy, AS_range=AS_range, AS_area=AS_area, AS_corr=AS_corr,
        PS_sim=PS_sim, PS_dist=PS_dist,
        VOnT=VOnT, VOffT=VOffT,
        # v3.2
        env_v32=locals().get("env_v32", None),
        GAT_ms=gat_ms, GOT_ms=got_ms,
        VOnT_env_ms=vont_ms_env, VOffT_env_ms=vofft_ms,
        OID_ms=oid_ms, TremorIndex=tremor_ratio,
    )
    extras = dict(fps=fps, n_cycles=len(cycles), viz=viz)

    # 11) 함수 반환 -------------------------------------------------------------
    return summary, pd.DataFrame(dict(cycle=[], start_time=[], end_time=[])), extras
    # =====================================================
    # v3.2 반환 패킷 구성 (✅ analyze 함수 내부, 동일 인덴트)
    # =====================================================
    viz = dict(
        t=t, total_s=total_s, left_s=left_s, right_s=right_s,
        E_on=E_on, E_off=E_off,
        thr_on=Th_on, thr_off=Th_off, Tlow_on=Tl_on, Tlow_off=Tl_off,
        i_move=i_move, i_steady=i_steady, i_last=i_last, i_end=i_end,
        cycles=cycles,
        AP=AP, TP=TP,
        AS_legacy=AS_legacy, AS_range=AS_range, AS_area=AS_area, AS_corr=AS_corr,
        PS_sim=PS_sim, PS_dist=PS_dist,
        VOnT=VOnT, VOffT=VOffT,
        env_v32=locals().get("env_v32", None),
        GAT_ms=gat_ms, GOT_ms=got_ms,
        VOnT_env_ms=vont_ms_env, VOffT_env_ms=vofft_ms,
        OID_ms=oid_ms, TremorIndex=tremor_ratio,
    )

    extras = dict(fps=fps, n_cycles=len(cycles), viz=viz)

    return summary, pd.DataFrame(dict(cycle=[], start_time=[], end_time=[])), extras
    # -------------------- Overview renderer --------------------
DEFAULT_KEYS = [
    "AP", "TP", "PS_dist", "AS_corr", "AS_range", "AS_area",
    "VOnT", "VOffT", "Auto_On_ms", "Auto_Off_ms", "Auto_Dur_ms",
    # 신규 6종 표시 필수
    "GAT_ms", "GOT_ms", "VOnT_env_ms", "VOffT_env_ms", "OID_ms", "TremorIndex"
]

def _val(x, ndig=4):
    try:
        if x is None: return "N/A"
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "N/A"
        return f"{xf:.{ndig}f}"
    except Exception:
        return "N/A"

# 스칼라 변환 헬퍼
def to_scalar(x):
    """넘어온 값이 배열/시리즈여도 안전하게 float으로 변환"""
    try:
        if x is None:
            return np.nan
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            try:
                x = x.item()
            except Exception:
                x = x[0]
        return float(x)
    except Exception:
        return np.nan
        
def render_overview(env: dict, keys=None):
    st.subheader("🩺 Overview")
    metrics = {k: _val(env.get(k), 4 if "ms" not in k else 2) for k in env.keys()}
    labels = {
        "AP":"AP","TP":"TP","PS_dist":"PS_dist (0=정상)","AS_corr":"AS_corr",
        "AS_range":"AS_range","AS_area":"AS_area",
        "VOnT":"VOnT (ms)","VOffT":"VOffT (ms)",
        "Auto_On_ms":"Auto On (ms)","Auto_Off_ms":"Auto Off (ms)","Auto_Dur_ms":"Auto Duration (ms)",
    }

    default = st.session_state.get("overview_keys", DEFAULT_KEYS)
    sel = st.multiselect(
        "표시 항목",
        DEFAULT_KEYS,
        default=default,
        key="ov_keys_ms"
    )

    st.session_state["overview_keys"] = sel
    keys = sel

    # ✅ 여기 붙여넣기
    rows = [keys[:4], keys[4:8], keys[8:12]]
    for row in rows:
        cols = st.columns(len(row)) if row else []
        for i, k in enumerate(row):
            with cols[i]:
                st.metric(labels.get(k, k), metrics.get(k, "N/A"))


    # ---- QC(선택) 메시지 계산 (기존 유지) ----
    fps  = env.get("fps", np.nan)
    ncyc = int(env.get("ncyc", 0) or 0)

    qc = []
    try:
        if isinstance(env.get("PS_dist"), (int, float)) and np.isfinite(env.get("PS_dist")) and env.get("PS_dist") > 0.08:
            qc.append("PS_dist↑ (위상 불일치 가능)")
        if isinstance(env.get("AP"), (int, float)) and np.isfinite(env.get("AP")) and env.get("AP") < 0.70:
            qc.append("AP 낮음 (진폭 불안정)")
        if isinstance(env.get("TP"), (int, float)) and np.isfinite(env.get("TP")) and env.get("TP") < 0.85:
            qc.append("TP 낮음 (주기 불안정)")
    except Exception:
        pass

    # ✅ 여기서 QI 계산 + 세션에 저장 + (비고정) 배지 렌더
    qi = compute_quality_from_env(env)
    st.session_state['__qi_latest__'] = qi

    render_quality_banner(
        st, qi,
        show_debug=st.session_state.get('debug_view', False),
        pinned=False
    )

    # ✅ 마지막에 FPS/사이클 수 & QC 표기
    st.caption(f"FPS: {np.nan if not np.isfinite(fps) else round(float(fps), 1)} | 검출된 사이클 수: {ncyc}")
    if qc:
        st.info("QC: " + " · ".join(qc))

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🧩 Preset")
    advanced = st.toggle("Advanced (연구자 모드 열기)", value=False, help="일반 임상 사용자는 끄고 사용하세요. Stable v3.1 프리셋이 자동 적용됩니다.")    
    st.markdown("---")
    st.markdown("### 🔬 Debug / Research")
    debug_view = st.toggle("Show debug info (연구자 전용)", value=False, key="debug_view")


    # Stable v3.1 프리셋(임상 기본): 슬라이더 숨김
    STABLE_PRESET = dict(baseline_s=0.06, k=1.40, M=40, W_ms=40.0, amp_frac_on=0.70)  # v3.1 권장값
    if not advanced:
        st.success("Preset: Stable v3.1 (임상용) · 매개변수는 숨김 처리")
        baseline_s = float(STABLE_PRESET["baseline_s"])
        k          = float(STABLE_PRESET["k"])
        M          = int(STABLE_PRESET["M"])
        W_ms       = float(STABLE_PRESET["W_ms"])
        amp_frac_on= float(STABLE_PRESET["amp_frac_on"])
    else:
        st.markdown("### ⚙ Energy & Profile (연구자)")
        prof = st.selectbox("분석 프로필", ["Normal", "ULP", "SD", "Custom"], index=0)
        pmap = {
            "Normal": dict(baseline_s=0.06, k=1.10, M=40, W_ms=35.0, amp_frac_on=0.70),
            "ULP":    dict(baseline_s=0.06, k=1.50, M=40, W_ms=35.0, amp_frac_on=0.60),
            "SD":     dict(baseline_s=0.06, k=1.75, M=50, W_ms=40.0, amp_frac_on=0.75),
            "Custom": dict(baseline_s=0.06, k=1.10, M=40, W_ms=35.0, amp_frac_on=0.70),
        }
        base = pmap.get(prof, pmap["Normal"])
        baseline_s = st.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=float(base["baseline_s"]), step=0.01)
        k          = st.number_input("임계 배수 k",      min_value=0.50, max_value=6.00,  value=float(base["k"]), step=0.10)
        M          = st.number_input("연속 프레임 M (참고용)", min_value=1, max_value=150, value=int(base["M"]), step=1)
        W_ms       = st.number_input("에너지 창(ms)",     min_value=2.0,  max_value=60.0,  value=float(base["W_ms"]), step=1.0)
        amp_frac_on= st.slider("정상화 최소 진폭 비율 (Onset 전용)", 0.10, 0.90, float(base["amp_frac_on"]), 0.01)
        st.caption("※ Offset은 N–D′ 기준으로 **0.80 고정**되어 DualDetector 내부에 적용됩니다.")

    st.markdown("---")
    st.markdown("### 🧲 DualDetector 설정 (Onset/Offset)")
    frame_ms = st.number_input("프레임 간격(ms)", min_value=0.10, max_value=5.0, value=0.66, step=0.01)

    st.markdown("**Onset 설정**")
    onset_theta = st.slider("θ_on (A_norm)", 0.10, 0.90, 0.50, 0.01, disabled=not advanced)
    onset_min_amp = st.slider("min_amp_frac", 0.10, 0.90, 0.58, 0.01, disabled=not advanced)
    onset_AP_min = st.slider("AP_min", 0.50, 1.00, 0.85, 0.01, disabled=not advanced)
    onset_TP_min = st.slider("TP_min", 0.50, 1.00, 0.90, 0.01, disabled=not advanced)
    onset_AS_min = st.slider("AS_corr_min", 0.50, 1.00, 0.95, 0.01, disabled=not advanced)
    onset_PS_max = st.slider("PS_dist_max", 0.00, 0.20, 0.05, 0.01, disabled=not advanced)
    onset_sustain = st.number_input("onset_sustain (frames)", min_value=1, max_value=60, value=5, step=1, disabled=not advanced)

adv = dict(baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac_on=amp_frac_on)

# -------------------- File upload (단일 케이스) --------------------
uploaded = st.file_uploader("CSV 또는 XLSX 파일을 업로드하세요 (단일 케이스 분석)", type=["csv", "xlsx"])
if uploaded is not None:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
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

    # DualDetector용 특징
    if total_s is not None and len(total_s):
        mn, mx = float(np.nanmin(total_s)), float(np.nanmax(total_s))
        denom = (mx - mn) if (mx - mn) > 1e-12 else 1.0
        A_norm = (total_s - mn) / denom
    else:
        A_norm = np.zeros_like(total_s) if total_s is not None else np.array([])

    on_cfg = OnsetConfig(theta=onset_theta, min_amp_frac=onset_min_amp,
                         AP_min=onset_AP_min, TP_min=onset_TP_min,
                         AS_corr_min=onset_AS_min, PS_dist_max=onset_PS_max,
                         sustain_frames=int(onset_sustain))
    off_cfg = OffsetConfig()  # 고정 N–D′ 파라미터 사용
    det_cfg = DetectorConfig(frame_ms=frame_ms, onset=on_cfg, offset=off_cfg)
    det = DualDetector(det_cfg)

    feats = {
        "t_ms": (t * 1000.0) if (t is not None) else np.array([]),
        "A_norm": A_norm if A_norm is not None else np.array([]),
        "AP": np.repeat(AP if np.isfinite(AP) else 0.0, len(A_norm)) if len(A_norm) else np.array([]),
        "TP": np.repeat(TP if np.isfinite(TP) else 0.0, len(A_norm)) if len(A_norm) else np.array([]),
        "AS_corr": np.repeat(AS_corr if np.isfinite(AS_corr) else 1.0, len(A_norm)) if len(A_norm) else np.array([]),
        "PS_dist": np.repeat(PS_dist if np.isfinite(PS_dist) else 0.0, len(A_norm)) if len(A_norm) else np.array([]),
    }
    det_res = det.detect(feats) if len(feats["t_ms"]) else {"onset_time_ms": None, "offset_time_ms": None, "duration_ms": None}
    Auto_On_ms  = det_res.get("onset_time_ms")
    Auto_Off_ms = det_res.get("offset_time_ms")
    Auto_Dur_ms = det_res.get("duration_ms")

# ---------------- Tabs 생성 및 Overview 실행 ----------------
if uploaded is None:
    st.info("📌 CSV/Excel 형식의 데이터를 업로드하면 분석/시각화 기능이 활성화됩니다.")
    st.markdown("---")
    tab_names = ["Parameter Comparison", "Batch Offset"]  # 업로드가 없어도 일부 탭 사용 가능
else:
    tab_names = ["Overview", "Visualization", "Batch Offset", "Parameter Comparison"]

# ✅ 1) pinned 배지를 넣을 상단 영역 확보
top_banner = st.container()

# ✅ 2) 탭 생성
tabs = st.tabs(tab_names)

# ✅ 3) Overview 탭이 존재한다면 → 먼저 실행
if "Overview" in tab_names and uploaded is not None:
    with tabs[tab_names.index("Overview")]:
        env = dict(
            AP=AP, TP=TP, PS_dist=PS_dist, AS_corr=AS_corr, AS_range=AS_range,
            AS_area=AS_area, VOnT=VOnT, VOffT=VOffT, fps=float(fps), ncyc=ncyc,
            Auto_On_ms=Auto_On_ms, Auto_Off_ms=Auto_Off_ms, Auto_Dur_ms=Auto_Dur_ms
        )
        render_overview(env)  # ✅ 여기서 QI 계산됨 & 세션에 저장됨
        st.dataframe(summary, use_container_width=True)

# ✅ 4) pinned 배지를 "탭 위"에 1번만 렌더
qi_latest = st.session_state.get("__qi_latest__")
with top_banner:
    if qi_latest is not None:
        render_quality_banner(
            st,
            qi_latest,
            show_debug=st.session_state.get("debug_view", False),
            pinned=True
        )

# 이후 기존 나머지 탭 콘텐트 유지

# ---- Onset/Offset 안내 ----
st.markdown(
    """
    > ⚠️ **Onset/Offset 자동검출 주의**  
    > 잔류소음, 비대칭 진동, 보간 오차 등에 따라 자동 검출값은 수동치와 ±10–30 ms 범위의 차이가 발생할 수 있습니다.  
    > 임상 해석 시, 병리군(특히 ULP/SD)은 신뢰도 저하가 자연스럽게 나타날 수 있으니 함께 제공되는 **Quality Indicator**를 참고하세요.
    """.strip()
)

# ---- Visualization ----
def render_v32(viz: dict):
    import numpy as np
    import matplotlib.pyplot as plt
    t = viz.get("t")
    env = viz.get("env_v32")
    if env is None or t is None:
        st.info("v3.2 envelope가 없어 마커를 표시하지 않습니다.")
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, env, label="Envelope (v3.2)")

    def ms_to_s(ms):
        return (ms / 1000.0) if (ms is not None and np.isfinite(ms)) else None

    v_vont_env  = ms_to_s(viz.get("VOnT_env_ms"))
    v_vofft_env = ms_to_s(viz.get("VOffT_env_ms"))
    t0 = float(t[0]) if hasattr(t, "__len__") and len(t) else 0.0

    if v_vont_env is not None:
        ax.axvline(t0 + v_vont_env, linestyle="--", label="VOnT_env", alpha=0.8)
    if v_vofft_env is not None:
        ax.axvline(t0 + v_vofft_env, linestyle="--", label="VOffT_env", alpha=0.8)

    oid_ms = viz.get("OID_ms")
    if oid_ms is not None and np.isfinite(oid_ms):
        ax.text(0.01, 0.95, f"OID = {oid_ms:.2f} ms",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", alpha=0.2))

    tri = viz.get("TremorIndex")
    if tri is not None and np.isfinite(tri):
        ax.text(0.99, 0.95, f"TremorIndex(4–5 Hz) = {tri:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", alpha=0.2))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")

    # ✅ Tremor 배지와 겹치지 않도록 개선됨!
    ax.legend(loc="lower right", framealpha=0.85)

    st.pyplot(fig)

def make_total_plot(t, total_s, cycles, i_move, i_steady, i_last, i_end, Auto_On_ms, Auto_Off_ms, zoom="전체"):
    fig = go.Figure()
    if t is None or total_s is None:
        fig.update_layout(template="simple_white", height=360)
        return fig
    fig.add_trace(go.Scatter(x=t, y=total_s, mode="lines",
                             line=dict(color=COLOR_TOTAL, width=2.2),
                             name="Total (smoothed)"))
    if cycles:
        for s, e in cycles[:120]:
            fig.add_vrect(x0=t[s], x1=t[e], fillcolor=COLOR_BAND, opacity=0.08, line_width=0)
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

    if isinstance(Auto_On_ms, (int,float)) and np.isfinite(Auto_On_ms):
        xon = Auto_On_ms / 1000.0
        fig.add_vline(x=xon, line=dict(color=COLOR_AUTOON, dash="dash", width=1.8))
        fig.add_annotation(x=xon, y=float(np.nanmax(total_s)), text=f"Auto On {Auto_On_ms:.1f} ms",
                           showarrow=False, font=dict(size=10, color=COLOR_AUTOON), yshift=28)
    if isinstance(Auto_Off_ms, (int,float)) and np.isfinite(Auto_Off_ms):
        xoff = Auto_Off_ms / 1000.0
        fig.add_vline(x=xoff, line=dict(color=COLOR_AUTOOFF, dash="dash", width=1.8))
        fig.add_annotation(x=xoff, y=float(np.nanmax(total_s)), text=f"Auto Off {Auto_Off_ms:.1f} ms",
                           showarrow=False, font=dict(size=10, color=COLOR_AUTOOFF), yshift=42)

    if zoom == "0–0.2s":   fig.update_xaxes(range=[0, 0.2])
    elif zoom == "0–0.5s": fig.update_xaxes(range=[0, 0.5])
    fig.update_layout(title="Total Signal with Detected Events",
                      xaxis_title="Time (s)", yaxis_title="Gray Level (a.u.)",
                      template="simple_white", height=420,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig

# ---- Left vs Right plot (예전 스타일) ----
def make_lr_plot2(t, left_s, right_s, AS_range, AS_corr, normalize=False, zoom="전체"):
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

    title = f"Left vs Right (AS_range {AS_range:.2f} · AS_corr {AS_corr:.2f})" if \
            (AS_range is not None and AS_corr is not None and
             isinstance(AS_range,(int,float)) and isinstance(AS_corr,(int,float))) else "Left vs Right"
    fig.update_layout(title=title,
                      xaxis_title="Time (s)",
                      yaxis_title=("Normalized" if normalize else "Gray Level (a.u.)"),
                      template="simple_white", height=340,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


# ---- Energy plot (Onset / Offset 전환) ----
def make_energy_plot2(mode, t, E_on, thr_on, Tlow_on, E_off, thr_off, Tlow_off, i_move, i_end, zoom="전체"):
    """
    mode: 'on' or 'off'
    """
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
    if event_idx is not None and t is not None and 0 <= int(event_idx) < len(t):
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


# ---- Visualization 탭 ----
if "Visualization" in tab_names and uploaded is not None:
    with tabs[tab_names.index("Visualization")]:
        st.markdown("#### Visualization")

        c1, c2, c3 = st.columns(3)
        view = c1.selectbox("표시 프리셋", ["전체", "좌/우", "Onset 에너지", "Offset 에너지"], index=0)
        zoom_preset = c2.selectbox("줌 프리셋", ["전체", "0–0.2s", "0–0.5s"], index=0)
        energy_mode = c3.radio("에너지 뷰", ["Onset", "Offset"], horizontal=True, index=0)

        # v3.2 viz 패킷 (공용)
        viz = extras.get("viz", {}) if isinstance(extras, dict) else {}

        # A) Total
        if view == "전체":
            st.markdown("#### A) Total")
            st.plotly_chart(
                make_total_plot(t, total_s, cycles, i_move, i_steady, i_last, i_end,
                                Auto_On_ms, Auto_Off_ms, zoom_preset),
                use_container_width=True
            )

            # ✅ v3.2 Envelope + OID + Tremor 표시 (Total 뷰 아래에 겹쳐 보여주기)
            if viz.get("env_v32") is not None:
                render_v32(viz)

            # B) Left vs Right
            st.markdown("#### B) Left vs Right")
            normalize_lr = st.checkbox("좌/우 정규화", value=False)
            st.plotly_chart(
                make_lr_plot2(t, left_s, right_s, AS_range, AS_corr, normalize_lr, zoom_preset),
                use_container_width=True
            )

            # C) Energy + Thresholds
            st.markdown("#### C) Energy + Thresholds")
            st.plotly_chart(
                make_energy_plot2("on" if energy_mode == "Onset" else "off",
                                  t, E_on, thr_on, Tlow_on, E_off, thr_off, Tlow_off,
                                  i_move, i_end, zoom_preset),
                use_container_width=True
            )

        elif view == "좌/우":
            st.markdown("#### Left vs Right")
            normalize_lr = st.checkbox("좌/우 정규화", value=False)
            st.plotly_chart(
                make_lr_plot2(t, left_s, right_s, AS_range, AS_corr, normalize_lr, zoom_preset),
                use_container_width=True
            )

        elif view == "Onset 에너지":
            st.markdown("#### Energy (Onset)")
            st.plotly_chart(
                make_energy_plot2("on", t, E_on, thr_on, Tlow_on, E_off, thr_off, Tlow_off,
                                  i_move, i_end, zoom_preset),
                use_container_width=True
            )

        elif view == "Offset 에너지":
            st.markdown("#### Energy (Offset)")
            st.plotly_chart(
                make_energy_plot2("off", t, E_on, thr_on, Tlow_on, E_off, thr_off, Tlow_off,
                                  i_move, i_end, zoom_preset),
                use_container_width=True
            )

# ---- Stats ----
if "stats" in tab_names and uploaded is not None:
    with tabs[tab_names.index("Validation")]:
        st.subheader("📊 Validation (RMSE / MAE / Bias)")
        st.info("자동 vs 수동 측정치 정량검증은 이 탭에서 확장됩니다. (배치 집계는 Batch Offset 탭)")

# ---- Batch Offset (수동 vs 자동) ----
def _load_offsets(file) -> pd.DataFrame:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    cols = _norm_cols(df.columns)
    df = df.copy(); df.columns = cols
    needed = {"case_id","offset_manual_ms","offset_auto_ms","group"}
    miss = needed - set(cols)
    if miss:
        raise ValueError(f"필수 컬럼 누락: {sorted(list(miss))}")
    return df[list(needed)]

def _offset_metrics(df: pd.DataFrame):
    d = df["offset_auto_ms"] - df["offset_manual_ms"]
    bias = float(np.nanmean(d))
    rmse = float(np.sqrt(np.nanmean(d**2)))
    within12 = float(np.nanmean(np.abs(d) <= 12.0))
    return bias, rmse, within12

if "Batch Offset" in tab_names:
    with tabs[tab_names.index("Batch Offset")]:
        st.subheader("🧪 Batch Offset — 수동 vs 자동 RMSE 검증")
        st.caption("파일 형식: CSV/XLSX (case_id, group[Normal/ULP/SD], offset_manual_ms, offset_auto_ms)")
        files = st.file_uploader("여러 파일 업로드 가능", type=["csv","xlsx"], accept_multiple_files=True)
        if files:
            frames = []
            for f in files:
                try:
                    frames.append(_load_offsets(f))
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                g = df_all.groupby("group", dropna=False)
                rows = []
                for k, gdf in g:
                    b, r, p12 = _offset_metrics(gdf)
                    rows.append(dict(group=k, bias_ms=b, rmse_ms=r, within_12ms=p12))
                tbl = pd.DataFrame(rows).sort_values("group")
                st.dataframe(tbl, use_container_width=True)

                # bar plot
                fig = px.bar(tbl.melt(id_vars="group", value_vars=["bias_ms","rmse_ms","within_12ms"]),
                             x="group", y="value", color="variable", barmode="group",
                             title="Offset: bias / RMSE / ±12ms 비율")
                st.plotly_chart(fig, use_container_width=True)

                # trend (산점 + y=x)
                df_all["diff"] = df_all["offset_auto_ms"] - df_all["offset_manual_ms"]
                fig2 = px.scatter(df_all, x="offset_manual_ms", y="offset_auto_ms", color="group",
                                  trendline="ols", title="수동 vs 자동 Offset 산점도 (y=x 참조)")
                fig2.add_trace(go.Scatter(x=df_all["offset_manual_ms"], y=df_all["offset_manual_ms"],
                                          mode="lines", line=dict(color="gray", dash="dot"),
                                          name="y=x"))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("정량 검증을 위해 파일을 업로드하세요.")

# ---- Parameter Comparison (누적) ----
def _load_params(file) -> pd.DataFrame:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    cols = _norm_cols(df.columns)
    df = df.copy(); df.columns = cols
    needed = {"case_id", "group", "ap", "tp", "as_corr", "ps_dist"}
    miss = needed - set(cols)
    if miss:
        raise ValueError(f"필수 컬럼 누락: {sorted(list(miss))}")
    return df[list(needed)]

if "Parameter Comparison" in tab_names:
    with tabs[tab_names.index("Parameter Comparison")]:
        st.subheader("🧬 Parameter Comparison (Normal vs ULP vs SD)")
        st.caption("파일 형식: case_id, group, AP, TP, AS_corr, PS_dist (CSV/XLSX) — 여러 파일 업로드 가능")

        if "param_hist" not in st.session_state:
            st.session_state["param_hist"] = pd.DataFrame(columns=["case_id","group","ap","tp","as_corr","ps_dist"])
        if st.button("📦 히스토리 초기화"):
            st.session_state["param_hist"] = st.session_state["param_hist"].iloc[0:0]

        pfiles = st.file_uploader("Drag & drop (다중 업로드)", type=["csv","xlsx"], accept_multiple_files=True, key="param_upl")
        if pfiles:
            add_frames = []
            for f in pfiles:
                try:
                    add_frames.append(_load_params(f))
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if add_frames:
                new_df = pd.concat(add_frames, ignore_index=True)
                # 중복 제거: case_id, group 기준 최신 업로드 우선
                hist = st.session_state["param_hist"]
                key_cols = ["case_id","group"]
                if not hist.empty:
                    key = pd.MultiIndex.from_frame(hist[key_cols])
                    m = pd.MultiIndex.from_frame(new_df[key_cols])
                    keep = ~m.isin(key)
                    merged = pd.concat([hist, new_df[keep]], ignore_index=True)
                    st.session_state["param_hist"] = merged
                else:
                    st.session_state["param_hist"] = new_df

        hist = st.session_state["param_hist"]
        if hist.empty:
            st.info("아직 저장된 히스토리가 없습니다.")
        else:
            st.markdown("#### 누적 히스토리")
            st.dataframe(hist.sort_values(["group","case_id"]), use_container_width=True, height=280)

            metric = st.selectbox("지표 선택", ["ap","tp","as_corr","ps_dist"], index=0)
            figv = px.violin(hist, x="group", y=metric, box=True, points="all",
                             title=f"{metric.upper()} 분포 (그룹별)", color="group")
            st.plotly_chart(figv, use_container_width=True)

            # 기본 통계
            st.markdown("#### 그룹별 기술통계")
            st.dataframe(hist.groupby("group")[["ap","tp","as_corr","ps_dist"]].describe().round(4), use_container_width=True)

            # 간단 효과크기/검정(옵션)
            if _HAS_SCIPY:
                st.markdown("#### 간단 통계검정 (정상 vs 병리군)")
                base = hist[hist["group"].str.lower().eq("normal")]
                for gname in [x for x in hist["group"].unique() if str(x).lower() != "normal"]:
                    other = hist[hist["group"]==gname]
                    cols = ["ap","tp","as_corr","ps_dist"]
                    rows=[]
                    for c in cols:
                        x = base[c].astype(float).dropna(); y = other[c].astype(float).dropna()
                        if len(x)>2 and len(y)>2:
                            t,p = stats.ttest_ind(x,y,equal_var=False)
                            d = (x.mean()-y.mean())/np.sqrt(((x.std(ddof=1)**2)+(y.std(ddof=1)**2))/2)
                            rows.append(dict(group_vs=f"Normal vs {gname}", metric=c, t=t, p=p, cohen_d=d))
                    if rows:
                        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed collaboratively by Isaka & Lian · 2025 © HSV Auto Analyzer v3.1 Stable")


























































