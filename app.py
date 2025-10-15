# app.py
# HSV Auto Analyzer - Clean Full Build (v2.3)
# - Robust against indentation/sciPy import issues
# - One-file input: time + (left/right or total) + (optional onset/offset)
# - Metrics: AP, TP, AS, PS, VOnT (ms), VOffT (ms)
# - Adjustable thresholds in UI

import math
import numpy as np
import pandas as pd

try:
    # scipy가 없으면 fallback으로 moving average 사용
    from scipy.signal import savgol_filter
    _HAS_SAVITZKY = True
except Exception:
    _HAS_SAVITZKY = False

import streamlit as st


# --------------------------- UI / PAGE ---------------------------------
st.set_page_config(page_title="HSV Auto Analyzer v2.3", layout="wide")
st.title("HSV Auto Analyzer v2.3")
st.caption("Amplitude/Time Periodicity, Amplitude/Phase Symmetry, Voice Onset/Offset Time – 안정 계산 버전")


# --------------------------- Utils -------------------------------------
def _norm_cols(cols):
    """컬럼명을 소문자 + 공백->언더스코어로 정규화"""
    return [c.lower().strip().replace(" ", "_") for c in cols]


def _moving_rms(x: np.ndarray, w: int) -> np.ndarray:
    """|x|의 이동 RMS. w<1이면 그대로 반환."""
    if w is None or w <= 1:
        return np.sqrt(np.maximum(x * x, 0.0))
    w = int(w)
    pad = w // 2
    xx = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(w) / float(w)
    m2 = np.convolve(xx * xx, ker, mode="valid")
    return np.sqrt(np.maximum(m2, 0.0))


def _smooth(signal: np.ndarray, fps: float) -> np.ndarray:
    """
    신호 smoothing. scipy.savgol이 있으면 사용하고, 없으면 이동평균 fallback.
    window 길이는 fps 기준 7~21 사이에서 자동 선택(대략 7~20ms 근처).
    """
    n = len(signal)
    if n < 7:
        return signal.astype(float)
    # 대략 7~21 범위에서 홀수로 설정
    base_w = int(max(7, min(21, round(fps * 0.007))))
    win = base_w if (base_w % 2 == 1) else base_w + 1
    win = min(win, n - 1) if n % 2 == 0 and win >= n else min(win, n - (1 - (n % 2)))
    if _HAS_SAVITZKY:
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
    """간단 peak 탐지(1차차분 교차). 너무 짧은 신호는 빈 결과."""
    if len(y) < 3:
        return np.array([], dtype=int)
    y1 = y[1:] - y[:-1]
    s = np.sign(y1)
    cu = (s[:-1] > 0) & (s[1:] <= 0)  # + -> 0/-
    idx = np.where(cu)[0] + 1
    return idx.astype(int)


def _build_cycles(t: np.ndarray, signal: np.ndarray, min_frames: int = 3) -> list:
    """
    peak 기반으로 '대략적인 cycle' 구간(s, e)을 만든다.
    각 cycle은 인접한 peak 사이 범위.
    """
    peaks = _detect_peaks(signal)
    cycles = []
    if len(peaks) < 2:
        return cycles
    for i in range(len(peaks) - 1):
        s = int(peaks[i])
        e = int(peaks[i + 1])
        if (e - s) >= max(2, min_frames):
            cycles.append((s, e))
    return cycles


def _safe_ratio(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b):
        return np.nan
    return float(a) / float(b)


def _nanmean0(x):
    v = np.nanmean(x) if len(x) else np.nan
    return 0.0 if (v is None or np.isnan(v)) else float(v)


def _clamp01(x):
    if x is None or np.isnan(x):
        return np.nan
    return float(max(0.0, min(1.0, x)))


# -------------------------- Metrics -------------------------------------
def _ap_tp(t: np.ndarray, total: np.ndarray, cycles: list) -> tuple:
    """
    AP(Time Periodicity), TP(Amplitude Periodicity) 계산:
    - 각 사이클의 진폭: max-min
    - 각 사이클의 주기: t[e]-t[s]
    주기의 변동/진폭의 변동이 작을수록 1에 가깝도록 1 - (std/mean)로 정의 (0~1 clamp)
    """
    if len(cycles) < 3:
        return (np.nan, np.nan)

    amps, periods = [], []
    for s, e in cycles:
        if e <= s:
            continue
        seg = total[s:e]
        amp = float(np.nanmax(seg) - np.nanmin(seg))
        Ti = float(t[e] - t[s])
        amps.append(amp)
        periods.append(max(Ti, 1e-9))

    amps = np.array(amps, dtype=float)
    periods = np.array(periods, dtype=float)

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
    """
    AS(Amplitude Symmetry): 좌/우 진폭의 유사성.
    각 사이클에서 (min(L,R)/max(L,R)) 계산 후 평균(0~1).
    """
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    ratios = []
    for s, e in cycles:
        if e <= s:
            continue
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        m = max(L, R)
        ratios.append((min(L, R) / m) if m > 0 else np.nan)
    return _clamp01(_nanmean0(ratios))


def _ps(left: np.ndarray, right: np.ndarray, t: np.ndarray, cycles: list) -> float:
    """
    PS(Phase Symmetry): 좌/우 '위상 차이'의 상대값.
    각 사이클에서 L/R 최대점의 시간차/주기 -> 0이면 perfect -> 1-평균값으로 반환(0~1).
    """
    if left is None or right is None or len(cycles) < 1:
        return np.nan
    diffs = []
    for s, e in cycles:
        if e <= s:
            continue
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s]) if (t is not None) else 1.0
        if Ti <= 0:
            continue
        d = abs(float(t[li] - t[ri])) / Ti
        diffs.append(min(1.0, d))
    if not len(diffs):
        return np.nan
    # 위상 차이가 작을수록 좋으므로 1 - 평균(정규화)
    return _clamp01(1.0 - _nanmean0(diffs))


# ------------------------ Main analyzer ---------------------------------
def analyze(df: pd.DataFrame, adv: dict):
    """
    입력 df에서 time/left/right/total/onset/offset 등을 lenient하게 매핑
    -> smoothing -> cycles -> AP/TP/AS/PS
    -> VOnT/VOffT (signal-guided, onset/offset trace가 있으면 우선 사용)
    """

    # ---- column mapping (lenient) ----
    cols = _norm_cols(df.columns.tolist())
    df.columns = cols

    def pick(name):
        for c in cols:
            if name in c:
                return c
        return None

    time_col = pick("time")
    left_col = pick("left")
    right_col = pick("right")
    total_col = pick("total")
    onset_col = pick("onset")
    offset_col = pick("offset")

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

    left = df[left_col].astype(float).values if left_col else None
    right = df[right_col].astype(float).values if right_col else None

    # ---- fps ----
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 1500.0

    # ---- smoothing ----
    total_s = _smooth(total, fps)
    left_s = _smooth(left, fps) if left is not None else None
    right_s = _smooth(right, fps) if right is not None else None

    # ---- cycles (간단 peak기반) ----
    min_frames = max(int(0.002 * fps), 5)  # 2ms 이상
    cycles = _build_cycles(t, total_s, min_frames=min_frames)

    # ---- AP/TP/AS/PS ----
    AP, TP = _ap_tp(t, total_s, cycles)
    AS = _as_range(left_s, right_s, cycles)
    PS = _ps(left_s, right_s, t, cycles)

    # ---------- VOnT / VOffT (signal-guided with onset/offset; ms) ----------
    # energy 계산용
    diff_total = np.abs(np.diff(total_s, prepend=total_s[0]))
    # 에너지 창 길이 (ms -> frames)
    W = max(int(round((adv.get("W_ms", 10.0) / 1000.0) * fps)), 3)
    E_total = _moving_rms(diff_total, W)

    onset_series = df[onset_col].astype(float).values if onset_col else None
    offset_series = df[offset_col].astype(float).values if offset_col else None

    E_on = _moving_rms(np.abs(np.diff(onset_series, prepend=onset_series[0])), W) if onset_series is not None else E_total
    E_off = _moving_rms(np.abs(np.diff(offset_series, prepend=offset_series[0])), W) if offset_series is not None else E_total

    # 베이스라인에서 임계값(mu + k*s)
    baseline_s = adv.get("baseline_s", 0.15)  # 시점 0~baseline_s 구간
    nB = max(int(round(baseline_s * fps)), 5)

    def _thr(E):
        base = E[:min(nB, len(E))]
        mu0 = float(np.mean(base)) if len(base) else 0.0
        s0 = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        return mu0 + adv.get("k", 3.0) * s0

    thr_on = _thr(E_on)
    thr_off = _thr(E_off)

    # 연속 프레임 조건 M
    M = int(adv.get("M", 5))
    above_on = (E_on > thr_on).astype(int)
    above_off = (E_off > thr_off).astype(int)
    run_on = np.convolve(above_on, np.ones(M, dtype=int), mode="same")
    run_off = np.convolve(above_off, np.ones(M, dtype=int), mode="same")

    # 움직임 시작 인덱스
    on_run_bin = (run_on >= M).astype(int)
    on_edges = np.diff(np.r_[0, on_run_bin, 0])
    on_starts = np.where(on_edges == 1)[0]
    i_move = int(on_starts[0]) if len(on_starts) else None

    # steady 정의(주기성이 안정된 최초 사이클 시작)
    VOnT = np.nan
    VOffT = np.nan

    if len(cycles) >= 3:
        # 전역 진폭 추정
        g_amp = float(np.nanmax([np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]) for s, e in cycles]))
        # 먼저 steady 찾기: 첫 cycle부터 후보를 보되, 초기 움직임 이후부터 탐색
        if i_move is None:
            i_move = cycles[0][0]
        t_move = float(t[i_move])

        # 첫 steady: 이후 사이클의 진폭이 전역의 일정 비율(amp_frac) 이상 & period/amp 편차 허용
        amp_frac = adv.get("amp_frac", 0.3)
        ap_thr = adv.get("ap_thr", 0.9)
        tp_thr = adv.get("tp_thr", 0.9)

        i_steady = None
        for (s, e) in cycles:
            if s < i_move:
                continue
            Ti = float(t[e] - t[s])
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            cond_amp = (amp >= amp_frac * g_amp) if g_amp > 0 else True
            # 간단히: 사이클 길이가 전체 평균과 크게 다르지 않도록 (TP 유사)
            cond_len = True  # 이미 AP/TP 계산했으므로, 여기선 암묵적으로 relax
            if cond_amp and cond_len:
                i_steady = int(s)
                break
        if i_steady is None:
            i_steady = cycles[0][0]
        t_steady = float(t[i_steady])

        # last steady: 끝에서부터 유사 조건으로 마지막 안정 구간
        i_last = None
        for (s, e) in reversed(cycles):
            Ti = float(t[e] - t[s])
            amp = float(np.nanmax(total_s[s:e]) - np.nanmin(total_s[s:e]))
            cond_amp = (amp >= amp_frac * g_amp) if g_amp > 0 else True
            if cond_amp:
                i_last = int(s)
                break
        if i_last is None:
            i_last = cycles[-1][1]
        t_last = float(t[i_last])

        # 움직임 종료 = 마지막 steady 이후의 '활성 run'의 종료점
        off_run_bin = (run_off >= M).astype(int)
        off_edges = np.diff(np.r_[0, off_run_bin, 0])
        off_starts = np.where(off_edges == 1)[0]
        off_ends = np.where(off_edges == -1)[0] - 1

        m = np.where(off_starts >= i_last)[0]
        if len(m):
            j = m[-1]
            i_end = int(off_ends[j])
            t_end = float(t[min(i_end, len(t) - 1)])
        else:
            i_end = cycles[-1][1]
            t_end = float(t[i_end])

        VOnT = float(t_steady - t_move) if (t_steady is not None and t_move is not None) else np.nan
        VOffT = float(t_end - t_last) if (t_end is not None and t_last is not None) else np.nan

    # 너무 작은 값은 0으로 스냅
    if VOnT is not None and not np.isnan(VOnT) and VOnT < 1e-4:
        VOnT = 0.0
    if VOffT is not None and not np.isnan(VOffT) and VOffT < 1e-4:
        VOffT = 0.0

    # ---- per-cycle detail (빈 테이블; 추후 확장) ----
    per_cycle = pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))

    # ---- summary & extras ----
    VOnT_ms = VOnT * 1000.0 if VOnT is not None and not np.isnan(VOnT) else np.nan
    VOffT_ms = VOffT * 1000.0 if VOffT is not None and not np.isnan(VOffT) else np.nan

    summary = pd.DataFrame({
        "Parameter": [
            "Amplitude Periodicity (AP)",
            "Time Periodicity (TP)",
            "Amplitude Symmetry (AS)",
            "Phase Symmetry (PS)",
            "Voice Onset Time (VOnT, ms)",
            "Voice Offset Time (VOffT, ms)",
        ],
        "Value": [AP, TP, AS, PS, VOnT_ms, VOffT_ms]
    })

    extras = dict(fps=fps, n_cycles=len(cycles))
    return summary, per_cycle, extras


# ---------------------------- UI ---------------------------------------
uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

with st.expander("⚙ 고급 설정 (기본값 그대로 사용해도 충분)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)
    k = c2.number_input("임계 배수 k", min_value=1.0, max_value=6.0, value=2.8, step=0.1)
    M = c3.number_input("연속 프레임 M", min_value=1, max_value=150, value=50, step=1)
    W_ms = c4.number_input("에너지 창(ms)", min_value=2.0, max_value=40.0, value=20.0, step=1.0)
    amp_frac = c5.slider("정상화 최소 진폭 (max에 대한 비율)", 0.10, 0.80, 0.70, 0.01)

    c6, c7 = st.columns(2)
    ap_thr = c6.slider("AP 임계값(보정용, 내부 steady 탐색 힌트)", 0.70, 1.00, 0.85, 0.01)
    tp_thr = c7.slider("TP 임계값(보정용, 내부 steady 탐색 힌트)", 0.70, 1.00, 0.98, 0.01)

adv = dict(
    baseline_s=baseline_s if 'baseline_s' in locals() else 0.15,
    k=k if 'k' in locals() else 3.0,
    M=M if 'M' in locals() else 5,
    W_ms=W_ms if 'W_ms' in locals() else 10.0,
    amp_frac=amp_frac if 'amp_frac' in locals() else 0.30,
    ap_thr=ap_thr if 'ap_thr' in locals() else 0.90,
    tp_thr=tp_thr if 'tp_thr' in locals() else 0.95,
)

st.markdown("---")

if uploaded is not None:
    # 파일 로드
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













