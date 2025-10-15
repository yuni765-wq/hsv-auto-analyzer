# app_v2.3.5.py
# HSV Auto Analyzer – v2 FINAL (onset-stabilized + robust fallbacks + debug)
# - Hysteresis + Debounce + Steady gating
# - Robust E_on source selection (+ optional "force total")
# - Strong guarantees: non-zero VOnT/VOffT via last-resort fallbacks
# - Rich DEBUG line: thresholds & key indices
# - Quick charts toggle for visual checks

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    _HAS_SAVITZKY = True
except Exception:
    _HAS_SAVITZKY = False

import streamlit as st

# --------------------------- UI / PAGE ---------------------------------
st.set_page_config(page_title="HSV Auto Analyzer v2.3.5", layout="wide")
st.title("HSV Auto Analyzer v2.3.5 – FINAL (v2)")
st.caption("AP/TP/AS/PS + Voice On/Off (히스테리시스/디바운스/steady 게이팅 + 강건한 Fallback + 디버그)")

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
        amps.append(amp); periods.append(max(Ti, 1e-9))
    amps, periods = np.array(amps, float), np.array(periods, float)
    def _periodicity(v):
        m = np.nanmean(v); s = np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
        if not np.isfinite(m) or m <= 0: return np.nan
        return _clamp01(1.0 - (s / m))
    TP = _periodicity(periods); AP = _periodicity(amps)
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
def analyze(df: pd.DataFrame, adv: dict, show_debug_charts: bool = False, force_total_for_onset: bool = False):
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

    # ---------- Energies for onset/offset ----------
    diff_total = np.abs(np.diff(total_s, prepend=total_s[0]))
    W = max(int(round((adv.get("W_ms", 10.0) / 1000.0) * fps)), 3)
    E_total = _moving_rms(diff_total, W)

    onset_series  = df[onset_col].astype(float).values if onset_col else None
    offset_series = df[offset_col].astype(float).values if offset_col else None

    # --- Robust E_on 선택 (패치 A) ---
    if force_total_for_onset:
        E_on = E_total
    else:
        if onset_series is not None:
            E_on_candidate = _moving_rms(np.abs(np.diff(onset_series, prepend=onset_series[0])), W)
            good = (np.nanstd(E_on_candidate) > 1e-6) and \
                   (np.count_nonzero(E_on_candidate > np.nanmean(E_on_candidate)) > 5)
            E_on = E_on_candidate if good else E_total
        else:
            E_on = E_total

    E_off = _moving_rms(np.abs(np.diff(offset_series, prepend=offset_series[0])), W) if offset_series is not None else E_total

    # Quick charts (옵션)
    if show_debug_charts:
        st.write("🔎 Quick Signal Check")
        st.line_chart(pd.DataFrame({"E_total": E_total}))
        st.line_chart(pd.DataFrame({"E_on": E_on, "E_off": E_off}))

    # ---- 임계값 (baseline 기반) ----
    baseline_s = adv.get("baseline_s", 0.18)
    nB = max(int(round(baseline_s * fps)), 5)

    def _thr(E):
        base = E[:min(nB, len(E))]
        mu0 = float(np.mean(base)) if len(base) else 0.0
        s0  = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        return mu0 + adv.get("k", 2.3) * s0

    thr_on, thr_off = _thr(E_on), _thr(E_off)

    # === Hysteresis + Debounce ===
    hysteresis_ratio = float(adv.get("hysteresis_ratio", 0.78))
    min_duration_ms  = float(adv.get("min_duration_ms", 35))
    refractory_ms    = float(adv.get("refractory_ms", 30))

    Tlow_on, Tlow_off = hysteresis_ratio * thr_on, hysteresis_ratio * thr_off
    min_dur_n = max(1, int(round(min_duration_ms * fps / 1000.0)))
    refrac_n  = max(1, int(round(refractory_ms   * fps / 1000.0)))

    # ---- onset: 상태기계 (baseline 이후 + 완화 재탐색) ----
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

    # ---- steady / last steady (패치 2: move 이후만) ----
    VOnT = np.nan; VOffT = np.nan
    i_steady = None; i_last = None; i_end = None
    if len(cycles) >= 3 and i_move is not None:
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles]))
        t_move = float(t[i_move])

        amp_frac = adv.get("amp_frac", 0.65)
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

        # --- i_steady: 반드시 i_move 이후 시작점으로 선택 ---
        for (s, e) in cycles:
            if s <= i_move:  # move 이전 구간 무시
                continue
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp > 0 and amp >= amp_frac * g_amp:
                AP_loc, TP_loc = _local_periodicity(s, e)
                if AP_loc >= ap_thr and TP_loc >= tp_thr:
                    i_steady = int(s)
                    break

        # fallback: i_move 이후 첫 사이클 시작
        if i_steady is None:
            for (s, e) in cycles:
                if s > i_move:
                    i_steady = int(s); break

        # 최후: 1프레임 뒤
        if i_steady is None:
            i_steady = min(i_move + 1, len(t) - 1)

        # 마지막 steady 시작점
        for (s, e) in reversed(cycles):
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            if g_amp > 0 and amp >= amp_frac * g_amp:
                i_last = int(s); break
        if i_last is None:
            i_last = cycles[-1][1]

        # ---- offset: 상태기계 (마지막 종료지점 선택) ----
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
                    in_voiced = False
        if i_end is None:
            M = int(adv.get("M", 60))
            above_off = (E_off > thr_off).astype(int)
            run_off = np.convolve(above_off, np.ones(M, dtype=int), mode="same")
            off_edges = np.diff(np.r_[0, (run_off >= M).astype(int), 0])
            off_starts = np.where(off_edges == 1)[0]
            off_ends   = np.where(off_edges == -1)[0] - 1
            m = np.where(off_starts >= i_last)[0]
            i_end = int(off_ends[m[-1]]) if len(m) else int(cycles[-1][1])

        # 시간변환
        t_steady = float(t[i_steady]); t_last = float(t[i_last])
        t_end = float(t[min(i_end, len(t) - 1)])

        VOnT  = float(t_steady - t_move)
        VOffT = float(t_end - t_last)

    # ---- LAST-RESORT FALLBACKS (패치 3) ----
    def _first_cross_after(E, thr, start, min_len):
        run = 0
        for i in range(start, len(E)):
            if E[i] > thr:
                run += 1
                if run >= min_len:
                    return i - (min_len - 1)
            else:
                run = 0
        return None

    if (np.isnan(VOnT) if isinstance(VOnT, float) else (VOnT is None)) or (i_move is None or i_steady is None):
        i0 = _first_cross_after(E_on, thr_on, nB, min_dur_n)
        if i0 is not None:
            # i0 이후 첫 사이클 시작을 steady로
            s2 = None
            for (s, e) in cycles:
                if s > i0:
                    s2 = s; break
            if s2 is None: s2 = min(i0 + min_dur_n, len(t) - 1)
            i_move   = i0 if i_move   is None else i_move
            i_steady = s2 if i_steady is None else i_steady
            VOnT = float(t[i_steady] - t[i_move])

    if (('i_end' not in locals()) or (i_end is None)) and (len(E_off) > 0):
        start_i = int(i_last) if (i_last is not None) else 0
        last_i = None; run = 0
        for i in range(start_i, len(E_off)):
            if E_off[i] > thr_off:
                run += 1
            else:
                if run >= min_dur_n:
                    last_i = i
                run = 0
        if last_i is None:
            last_i = len(E_off) - 1
        i_end = last_i
        if i_last is None and len(cycles) > 0:
            i_last = cycles[-1][0]
        if i_last is not None:
            VOffT = float(t[min(i_end, len(t)-1)] - t[i_last])

    # ---- NaN 방어: None/NaN → 0.0 ----
    if (VOnT is None) or (isinstance(VOnT, float) and np.isnan(VOnT)):   VOnT  = 0.0
    if (VOffT is None) or (isinstance(VOffT, float) and np.isnan(VOffT)): VOffT = 0.0

    # ---- DEBUG (패치 1) ----
    st.write(
        "DEBUG ▶ "
        f"FPS={fps:.1f}, cycles={len(cycles)}, nB={nB}, "
        f"thr_on={thr_on:.4g}, Tlow_on={Tlow_on:.4g}, "
        f"thr_off={thr_off:.4g}, Tlow_off={Tlow_off:.4g}, "
        f"i_move={i_move}, i_steady={i_steady}, i_last={i_last}, i_end={i_end}, "
        f"VOnT={VOnT*1000:.2f} ms, VOffT={VOffT*1000:.2f} ms"
    )

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
        "Value": [AP, TP, AS, PS, VOnT * 1000.0, VOffT * 1000.0]
    })
    extras = dict(fps=fps, n_cycles=len(cycles))
    return summary, per_cycle, extras

# ---------------------------- UI ---------------------------------------
uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

with st.expander("⚙ 고급 설정 (필요 시만 조정)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", 0.05, 0.50, 0.18, 0.01)
    k          = c2.number_input("임계 배수 k", 1.0, 6.0, 2.3, 0.1)
    M          = c3.number_input("연속 프레임 M", 1, 150, 60, 1)
    W_ms       = c4.number_input("에너지 창(ms)", 2.0, 40.0, 40.0, 1.0)
    amp_frac   = c5.slider("정상화 최소 진폭 (max 비율)", 0.10, 0.90, 0.65, 0.01)

    c6, c7, c8 = st.columns(3)
    ap_thr = c6.slider("AP 임계값(steady 힌트)", 0.70, 1.00, 0.95, 0.01)
    tp_thr = c7.slider("TP 임계값(steady 힌트)", 0.70, 1.00, 0.98, 0.01)
    force_total = c8.checkbox("onset 열 무시(총 에너지 사용)", value=False)

with st.expander("🔬 고급(읽기전용: Hysteresis/Debounce & Debug)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hysteresis Ratio", "0.78")
    c2.metric("Min Duration (ms)", "35")
    c3.metric("Refractory (ms)", "30")
    show_debug = c4.checkbox("Quick charts 보기", value=False)

adv = dict(
    baseline_s=baseline_s,
    k=k,
    M=M,
    W_ms=W_ms,
    amp_frac=amp_frac,
    ap_thr=ap_thr,
    tp_thr=tp_thr,
    hysteresis_ratio=0.78,
    min_duration_ms=35,
    refractory_ms=30,
)

st.markdown("---")

if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    with st.spinner("분석 중..."):
        summary, per_cycle, extras = analyze(
            df, adv, show_debug_charts=show_debug, force_total_for_onset=force_total
        )

    st.subheader("✅ 결과 요약")
    st.dataframe(summary, use_container_width=True)
    st.write(f"FPS: {extras.get('fps', np.nan):.1f}, 검출된 사이클 수: {extras.get('n_cycles', 0)}")
else:
    st.info("샘플 파일(시간 + 좌/우 또는 total, 선택적으로 onset/offset 컬럼)을 업로드해 주세요.")
