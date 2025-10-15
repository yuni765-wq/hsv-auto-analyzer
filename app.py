# app_v2.3.3.py
# HSV Auto Analyzer – onset-stabilized (hysteresis + debounce + steady gating + robust E_on)
# - 변경점:
#   * E_on 품질 자동판단 → 부적합 시 total 에너지로 대체 (패치 A)
#   * onset 상태기계 보강 + 완화 재탐색(fallback) (패치 B)
#   * i_move == i_steady일 때 0 ms 방지 (패치 C)
#   * 기본값 튜닝: baseline 0.18s, k 2.5, hysteresis_ratio 0.78, amp_frac 0.70,
#                 min_duration_ms 35, refractory_ms 35

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    _HAS_SAVITZKY = True
except Exception:
    _HAS_SAVITZKY = False

import streamlit as st

# --------------------------- UI / PAGE ---------------------------------
st.set_page_config(page_title="HSV Auto Analyzer v2.3.3", layout="wide")
st.title("HSV Auto Analyzer v2.3.3")
st.caption("AP/TP/AS/PS + Voice On/Off (히스테리시스/디바운스/steady 게이팅 + onset 안정화)")

# --------------------------- Utils -------------------------------------
def _norm_cols(cols):
    return [c.lower().strip().replace(" ", "_") for c in cols]

def _moving_rms(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return np.sqrt(np.maximum(x * x, 0.0))
    w = int(w)
    pad = w // 2
    xx = np.pad(x, (pad, pad), mode="edge")
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
    if _HAS_SAVITZKY:
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
    cu = (s[:-1] > 0) & (s[1:] <= 0)
    idx = np.where(cu)[0] + 1
    return idx.astype(int)

def _build_cycles(t: np.ndarray, signal: np.ndarray, min_frames: int = 3) -> list:
    peaks = _detect_peaks(signal)
    cycles = []
    if len(peaks) < 2:
        return cycles
    for i in range(len(peaks) - 1):
        s = int(peaks[i]); e = int(peaks[i + 1])
        if (e - s) >= max(2, min_frames):
            cycles.append((s, e))
    return cycles

def _nanmean0(x):
    v = np.nanmean(x) if len(x) else np.nan
    return 0.0 if (v is None or np.isnan(v)) else float(v)

def _clamp01(x):
    if x is None or np.isnan(x):
        return np.nan
    return float(max(0.0, min(1.0, x)))

# -------------------------- Metrics -------------------------------------
def _ap_tp(t: np.ndarray, total: np.ndarray, cycles: list) -> tuple:
    if len(cycles) < 3:
        return (np.nan, np.nan)
    amps, periods = [], []
    for s, e in cycles:
        if e <= s: continue
        seg = total[s:e]
        amp = float(np.nanmax(seg) - np.nanmin(seg))
        Ti  = float(t[e] - t[s])
        amps.append(amp)
        periods.append(max(Ti, 1e-9))
    amps, periods = np.array(amps, float), np.array(periods, float)

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
        if e <= s: continue
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
        if e <= s: continue
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s]) if (t is not None) else 1.0
        if Ti <= 0: continue
        d = abs(float(t[li] - t[ri])) / Ti
        diffs.append(min(1.0, d))
    if not len(diffs): return np.nan
    return _clamp01(1.0 - _nanmean0(diffs))

# ------------------------ Main analyzer ---------------------------------
def analyze(df: pd.DataFrame, adv: dict):
    # ---- column mapping ----
    cols = _norm_cols(df.columns.tolist()); df.columns = cols
    def pick(name):
        for c in cols:
            if name in c: return c
        return None
    time_col  = pick("time")
    left_col  = pick("left")
    right_col = pick("right")
    total_col = pick("total")
    onset_col = pick("onset")
    offset_col= pick("offset")

    if time_col is None:
        empty = pd.DataFrame()
        return (pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0))

    # ---- signals ----
    t = df[time_col].astype(float).values
    if np.nanmax(t) > 10.0:  # ms -> s
        t = t / 1000.0
    if total_col is not None:
        total = df[total_col].astype(float).values
    elif left_col is not None and right_col is not None:
        total = (df[left_col].astype(float).values + df[right_col].astype(float).values) / 2.0
    else:
        empty = pd.DataFrame()
        return (pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0))
    left  = df[left_col].astype(float).values if left_col else None
    right = df[right_col].astype(float).values if right_col else None

    # ---- fps / smoothing ----
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0
    total_s = _smooth(total, fps)
    left_s  = _smooth(left, fps) if left is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    # ---- cycles ----
    min_frames = max(int(0.002 * fps), 5)  # ≥2ms
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    # ---- AP/TP/AS/PS ----
    AP, TP = _ap_tp(t, total_s, cycles)
    AS     = _as_range(left_s, right_s, cycles)
    PS     = _ps(left_s, right_s, t, cycles)

    # ---------- VOnT / VOffT (with hysteresis/ debounce) ----------
    diff_total = np.abs(np.diff(total_s, prepend=total_s[0]))
    W = max(int(round((adv.get("W_ms", 10.0) / 1000.0) * fps)), 3)
    E_total = _moving_rms(diff_total, W)
    st.line_chart(E_total)
    onset_series  = df[onset_col].astype(float).values if onset_col else None
    offset_series = df[offset_col].astype(float).values if offset_col else None

    # ---- 패치 A: E_on 품질 자동판단 후 선택 ----
    if onset_series is not None:
        E_on_candidate = _moving_rms(np.abs(np.diff(onset_series, prepend=onset_series[0])), W)
        # 평탄/희소 여부 간단 판단(표준편차 + 유효 포인트 수)
        good = (np.nanstd(E_on_candidate) > 1e-6) and \
               (np.count_nonzero(E_on_candidate > np.nanmean(E_on_candidate)) > 5)
        E_on = E_on_candidate if good else E_total
    else:
        E_on = E_total

    E_off = _moving_rms(np.abs(np.diff(offset_series, prepend=offset_series[0])), W) if offset_series is not None else E_total

    # ---- 임계값 (baseline 기반) ----
    baseline_s = adv.get("baseline_s", 0.18)  # 보정 기본값
    nB = max(int(round(baseline_s * fps)), 5)

    def _thr(E):
        base = E[:min(nB, len(E))]
        mu0 = float(np.mean(base)) if len(base) else 0.0
        s0  = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        return mu0 + adv.get("k", 2.5) * s0

    thr_on, thr_off = _thr(E_on), _thr(E_off)

    # === Hysteresis + Debounce ===
    hysteresis_ratio = float(adv.get("hysteresis_ratio", 0.78))
    min_duration_ms  = float(adv.get("min_duration_ms", 35))   # 기본값 35 (완화)
    refractory_ms    = float(adv.get("refractory_ms", 35))

    Tlow_on, Tlow_off = hysteresis_ratio * thr_on, hysteresis_ratio * thr_off
    min_dur_n = max(1, int(round(min_duration_ms * fps / 1000.0)))
    refrac_n  = max(1, int(round(refractory_ms   * fps / 1000.0)))

    # ---- onset: 상태기계 (패치 B: baseline 이후 + 완화 재탐색 포함) ----
    i_move = None
    in_voiced = False
    last_onset_idx = -10**9
    onset_idx = None
    start_allowed = nB  # baseline 이후만 onset 후보 허용

    for i in range(len(E_on)):
        x = E_on[i]
        if not in_voiced:
            if i >= start_allowed and x > thr_on and (i - last_onset_idx) >= refrac_n:
                in_voiced = True
                onset_idx = i
                last_onset_idx = i
        else:
            if x < Tlow_on and (i - onset_idx) >= min_dur_n:
                i_move = onset_idx
                in_voiced = False
                break

    # 1차 실패 시 완화 재탐색
    if i_move is None:
        in_voiced = False
        onset_idx = None
        thr_on_relaxed = thr_on * 0.95
        min_dur_relax  = max(1, int(0.7 * min_dur_n))
        for i in range(start_allowed, len(E_on)):
            x = E_on[i]
            if not in_voiced:
                if x > thr_on_relaxed:
                    in_voiced = True
                    onset_idx = i
            else:
                if x < Tlow_on and (i - onset_idx) >= min_dur_relax:
                    i_move = onset_idx
                    break

    # ---- steady / last steady (AP/TP 힌트 반영) ----
    VOnT = np.nan; VOffT = np.nan
    if len(cycles) >= 3 and i_move is not None:
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles]))
        t_move = float(t[i_move])

        amp_frac = adv.get("amp_frac", 0.70)
        ap_thr = adv.get("ap_thr", 0.95)
        tp_thr = adv.get("tp_thr", 0.98)

        def _local_periodicity(s, e):
            amps, periods, cnt = [], [], 0
            for (cs, ce) in cycles:
                if ce <= s: continue
                if cs >= e: break
                seg = total_s[cs:ce]
                amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
                periods.append(float(t[ce] - t[cs]))
                cnt += 1
                if cnt >= 3: break
            if len(amps) < 2 or len(periods) < 2: return (0.0, 0.0)
            def _p(v):
                m = np.nanmean(v); sd = np.nanstd(v, ddof=1)
                return max(0.0, min(1.0, 1.0 - (sd / m))) if m > 0 else 0.0
            return (_p(amps), _p(periods))

        i_steady = None
        for (s, e) in cycles:
            if s < i_move: 
                continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp > 0 and amp >= amp_frac * g_amp:
                AP_loc, TP_loc = _local_periodicity(s, e)
                if AP_loc >= ap_thr and TP_loc >= tp_thr:
                    i_steady = int(s)
                    break
        if i_steady is None:
            i_steady = cycles[0][0]

        # ---- 패치 C: i_move와 i_steady가 같거나 역전되면 steady를 다음 사이클로 미룸 ----
        if i_steady <= i_move:
            for (s, e) in cycles:
                if s > i_move:
                    i_steady = s
                    break

        t_steady = float(t[i_steady])

        i_last = None
        for (s, e) in reversed(cycles):
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp > 0 and amp >= amp_frac * g_amp:
                i_last = int(s); break
        if i_last is None:
            i_last = cycles[-1][1]
        t_last = float(t[i_last])

        # ---- offset: 상태기계 (마지막 종료지점 선택) ----
        i_end = None
        in_voiced = False
        seg_start = None
        start_i = max(0, int(i_last))
        for i in range(start_i, len(E_off)):
            x = E_off[i]
            if not in_voiced:
                if x > thr_off:
                    in_voiced = True
                    seg_start = i
            else:
                if x < Tlow_off and (i - seg_start) >= min_dur_n:
                    i_end = i
                    in_voiced = False  # 계속 탐색해 가장 마지막 종료 지점 선택
        if i_end is None:
            M = int(adv.get("M", 60))
            above_off = (E_off > thr_off).astype(int)
            run_off = np.convolve(above_off, np.ones(M, dtype=int), mode="same")
            off_edges = np.diff(np.r_[0, (run_off >= M).astype(int), 0])
            off_starts = np.where(off_edges == 1)[0]
            off_ends   = np.where(off_edges == -1)[0] - 1
            m = np.where(off_starts >= i_last)[0]
            i_end = int(off_ends[m[-1]]) if len(m) else int(cycles[-1][1])

        t_end = float(t[min(i_end, len(t) - 1)])
        VOnT  = float(t_steady - t_move)
        VOffT = float(t_end - t_last)

    # 너무 작은 값은 0으로 스냅
    if VOnT  is not None and not np.isnan(VOnT)  and VOnT  < 1e-4: VOnT  = 0.0
    if VOffT is not None and not np.isnan(VOffT) and VOffT < 1e-4: VOffT = 0.0

    per_cycle = pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))
    summary = pd.DataFrame({
        "Parameter": [
            "Amplitude Periodicity (AP)",
            "Time Periodicity (TP)",
            "Amplitude Symmetry (AS)",
            "Phase Symmetry (PS)",
            "Voice Onset Time (VOnT, ms)",
            "Voice Offset Time (VOffT, ms)",
        ],
        "Value": [AP, TP, AS, PS,
                  (VOnT * 1000.0 if VOnT is not None and not np.isnan(VOnT) else np.nan),
                  (VOffT * 1000.0 if VOffT is not None and not np.isnan(VOffT) else np.nan)]
    })
    extras = dict(fps=fps, n_cycles=len(cycles))
    # ---- NaN 방어: 값이 None이거나 NaN이면 0으로 스냅 ----
if VOnT is None or np.isnan(VOnT):
    VOnT = 0.0
if VOffT is None or np.isnan(VOffT):
VOffT = 0.0
return summary, per_cycle, extras
st.write(f"DEBUG ▶ FPS: {fps:.2f}, Cycles: {len(cycles)}, VOnT_raw: {VOnT}, VOffT_raw: {VOffT}")

# ---------------------------- UI ---------------------------------------
uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

with st.expander("⚙ 고급 설정 (필요 시만 조정)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", 0.05, 0.50, 0.18, 0.01)
    k          = c2.number_input("임계 배수 k", 1.0, 6.0, 2.5, 0.1)
    M          = c3.number_input("연속 프레임 M", 1, 150, 60, 1)
    W_ms       = c4.number_input("에너지 창(ms)", 2.0, 40.0, 40.0, 1.0)
    amp_frac   = c5.slider("정상화 최소 진폭 (max 비율)", 0.10, 0.90, 0.70, 0.01)

    c6, c7 = st.columns(2)
    ap_thr = c6.slider("AP 임계값(steady 힌트)", 0.70, 1.00, 0.95, 0.01)
    tp_thr = c7.slider("TP 임계값(steady 힌트)", 0.70, 1.00, 0.98, 0.01)

with st.expander("🔬 고급(읽기전용: Hysteresis/Debounce)", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Hysteresis Ratio", "0.78")
    c2.metric("Min Duration (ms)", "35")
    c3.metric("Refractory (ms)", "35")

adv = dict(
    baseline_s=baseline_s,
    k=k,
    M=M,
    W_ms=W_ms,
    amp_frac=amp_frac,
    ap_thr=ap_thr,
    tp_thr=tp_thr,
    # NEW: advanced fixed params
    hysteresis_ratio=0.78,
    min_duration_ms=35,  # 완화
    refractory_ms=35,
)

st.markdown("---")

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    with st.spinner("분석 중..."):
        summary, per_cycle, extras = analyze(df, adv)

    st.subheader("✅ 결과 요약")
    st.dataframe(summary, use_container_width=True)
    st.write(f"FPS: {extras.get('fps', np.nan):.1f}, 검출된 사이클 수: {extras.get('n_cycles', 0)}")
else:
    st.info("샘플 파일(시간 + 좌/우 또는 total, 선택적으로 onset/offset 컬럼)을 업로드해 주세요.")


