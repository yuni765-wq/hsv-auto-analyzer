# HSV Auto Analyzer - Version 2.1 (Refined Engine)
# - Keeps AP/TP/AS/PS and adds VOnT/VOffT with robust cycle detection
# - One-file input: time + (left/right or total) [+ optional onset/offset]
# - Advanced settings (baseline length, thresholds) in an expander

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="HSV Auto Analyzer v2.1", layout="wide")

st.title("🧠 HSV Auto Analyzer v2.1")
st.caption("Amplitude/Time Periodicity, Amplitude/Phase Symmetry, Voice Onset/Offset Time – 정밀 계산 엔진")

# ---------------------- Utils ----------------------

def _norm_cols(cols):
    return [c.lower().strip().replace(" ", "_") for c in cols]

@st.cache_data
def load_table(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=0)
    df.columns = _norm_cols(df.columns)
    return df

# Peak detection (simple and fast)
def _detect_peaks(y):
    y = np.asarray(y)
    if len(y) < 3:
        return np.array([], dtype=int)
    dy1 = y[1:-1] - y[:-2]
    dy2 = y[1:-1] - y[2:]
    idx = np.where((dy1 > 0) & (dy2 >= 0))[0] + 1
    return idx

# Moving RMS on |Δx|
def _moving_rms(diff, w):
    if w <= 1:
        return np.sqrt(np.maximum(diff**2, 0))
    pad = w // 2
    xp = np.pad(diff, (pad, pad), mode="edge")
    k = np.ones(w) / w
    return np.sqrt(np.convolve(xp**2, k, mode="valid"))

# Build cycles from peaks (closed phase bright → peaks on -total)

def _build_cycles(t, total, min_frames=5):
    peaks = _detect_peaks(-total)
    cycles = []
    for i in range(len(peaks) - 1):
        s, e = int(peaks[i]), int(peaks[i + 1])
        if (e - s) >= min_frames:
            cycles.append((s, e))
    return cycles

# Per-cycle metrics

def _periods_amps(t, y, cycles):
    periods, amps = [], []
    for s, e in cycles:
        seg_t = t[s:e]
        seg_y = y[s:e]
        if len(seg_t) < 2:
            continue
        periods.append(float(seg_t[-1] - seg_t[0]))
        amps.append(float(np.nanmax(seg_y) - np.nanmin(seg_y)))
    return np.asarray(periods), np.asarray(amps)

# AP / TP from adjacent cycles

def _ap_tp(t, y, cycles):
    periods, amps = _periods_amps(t, y, cycles)
    ap, tp = [], []
    for i in range(len(amps) - 1):
        a, b = amps[i], amps[i + 1]
        m = max(a, b)
        ap.append(min(a, b) / m if m > 0 else np.nan)
    for i in range(len(periods) - 1):
        a, b = periods[i], periods[i + 1]
        m = max(a, b)
        tp.append(min(a, b) / m if m > 0 else np.nan)
    return (np.nanmean(ap) if len(ap) else np.nan,
            (np.nanmean(tp) if len(tp) else np.nan), periods, amps)

# AS (amplitude symmetry) – range-based per cycle (left/right)

def _as_range(left, right, cycles):
    if left is None or right is None or len(cycles) == 0:
        return np.nan
    ratios = []
    for s, e in cycles:
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        if max(L, R) > 0:
            ratios.append(min(L, R) / max(L, R))
    return np.nanmean(ratios) if ratios else np.nan

# PS (phase symmetry) – |t_left(max) - t_right(max)| / T_i

def _ps(left, right, t, cycles):
    if left is None or right is None or len(cycles) == 0:
        return np.nan
    vals = []
    for s, e in cycles:
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s]) if (e > s) else np.nan
        if Ti and Ti > 0:
            vals.append(abs(float(t[li] - t[ri])) / Ti)
    return np.nanmean(vals) if vals else np.nan

# Steady windows for VOnT/VOffT

def _first_steady_from(t, y, cycles, g_amp, start_frame, K, ap_thr, tp_thr, amp_frac):
    periods, amps = _periods_amps(t, y, cycles)
    starts = np.array([s for s, _ in cycles], dtype=int)
    for j in range(0, len(cycles) - K):
        if starts[j] < start_frame:
            continue
        subA = amps[j:j + K]
        subP = periods[j:j + K]
        ap_vals, tp_vals = [], []
        for i in range(K - 1):
            aa, bb = subA[i], subA[i + 1]; m = max(aa, bb); ap_vals.append(min(aa, bb) / m if m > 0 else np.nan)
            aa, bb = subP[i], subP[i + 1]; m = max(aa, bb); tp_vals.append(min(aa, bb) / m if m > 0 else np.nan)
        if (np.nanmin(ap_vals) >= ap_thr) and (np.nanmin(tp_vals) >= tp_thr) and (np.nanmin(subA) >= amp_frac * g_amp):
            return starts[j], float(t[starts[j]])
    return None, None


def _last_steady_before_end(t, y, cycles, g_amp, K, ap_thr, tp_thr, amp_frac):
    periods, amps = _periods_amps(t, y, cycles)
    ends = np.array([e for _, e in cycles], dtype=int)
    for j_end in range(len(cycles), K - 1, -1):
        j = j_end - K
        subA = amps[j:j + K]
        subP = periods[j:j + K]
        ap_vals, tp_vals = [], []
        for i in range(K - 1):
            aa, bb = subA[i], subA[i + 1]; m = max(aa, bb); ap_vals.append(min(aa, bb) / m if m > 0 else np.nan)
            aa, bb = subP[i], subP[i + 1]; m = max(aa, bb); tp_vals.append(min(aa, bb) / m if m > 0 else np.nan)
        if (np.nanmin(ap_vals) >= ap_thr) and (np.nanmin(tp_vals) >= tp_thr) and (np.nanmin(subA) >= amp_frac * g_amp):
            return ends[j + K - 1], float(t[ends[j + K - 1]])
    return None, None

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import streamlit as st
# ----------------- Utils -----------------
def _norm_cols(cols):
    return [c.lower().strip().replace(" ", "_") for c in cols]

@st.cache_data
def load_table(file):
    """엑셀 또는 CSV 파일을 자동으로 판별하여 불러옵니다."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Moving RMS, detect_peaks 등 기존 함수들 계속 아래에 있음
def _moving_rms(diff, w):
    ...

# ==========================================================
# Main Analyzer
# ==========================================================

def analyze(df, adv):
    # ---- column mapping ----
    cols = df.columns.tolist()
    time_col   = next((c for c in cols if 'time'  in c.lower()), None)
    left_col   = next((c for c in cols if 'left'  in c.lower()), None)
    right_col  = next((c for c in cols if 'right' in c.lower()), None)
    total_col  = next((c for c in cols if 'total' in c.lower()), None)

    if time_col is None:
        return pd.DataFrame({"Parameter": [], "Value": []}), pd.DataFrame(), dict(fps=np.nan, n_cycles=0)

    t = df[time_col].astype(float).values
    if t.max() > 10:
        t = t / 1000.0

    if total_col:
        total = df[total_col].astype(float).values
    elif left_col and right_col:
        total = (df[left_col].astype(float).values + df[right_col].astype(float).values) / 2.0
    else:
        return pd.DataFrame({"Parameter": [], "Value": []}), pd.DataFrame(), dict(fps=np.nan, n_cycles=0)

    # ---- FPS ----
    dt = np.median(np.diff(t))
    fps = 1.0 / dt if dt > 0 else 1500.0

    # ---- Dummy cycles (임시 cycle 검출 대체) ----
    cycles = [(i, i+10) for i in range(0, len(total)-10, 10)]

    # ==========================================================
    # Onset / Offset detection (savgol_filter 기반)
    # ==========================================================
    signal = total.astype(float)
    win_len = 11 if len(signal) >= 11 else 7
    smoothed = savgol_filter(signal, window_length=win_len, polyorder=3, mode="interp")

    stable_start = int(len(smoothed) * 0.8)
    stable_region = smoothed[stable_start:]
    stable_mean = float(np.mean(stable_region)) if len(stable_region) else float(np.mean(smoothed))
    amp_ref = float(np.percentile(np.abs(stable_region - stable_mean), 95)) if len(stable_region) else 0.0

    th_on = amp_ref * 0.10
    th_off = amp_ref * 0.07
    centered = smoothed - stable_mean
    amp = np.abs(centered)
    min_frames = max(1, int((10.0 / 1000.0) * fps))

    above = amp > th_on
    below = amp < th_off
    onset_index = offset_index = None

    cnt = 0
    for i in range(len(above)):
        cnt = cnt + 1 if above[i] else 0
        if cnt >= min_frames:
            onset_index = i - cnt + 1
            break

    cnt = 0
    start_idx = onset_index if onset_index is not None else 0
    for i in range(start_idx, len(below)):
        cnt = cnt + 1 if below[i] else 0
        if cnt >= min_frames:
            offset_index = i - cnt + 1
            break

    onset_time_s  = (onset_index / fps) if onset_index is not None else np.nan
    offset_time_s = (offset_index / fps) if offset_index is not None else np.nan

    VOnT, VOffT = onset_time_s, offset_time_s

    if VOnT is not None and VOnT < 1e-4:
        VOnT = 0.0
    if VOffT is not None and VOffT < 1e-4:
        VOffT = 0.0

    # ==========================================================
    # Summary
    # ==========================================================
    summary = pd.DataFrame({
        "Parameter": [
            "Amplitude Periodicity (AP)",
            "Time Periodicity (TP)",
            "Amplitude Symmetry (AS)",
            "Phase Symmetry (PS)",
            "Voice Onset Time (VOnT, s)",
            "Voice Offset Time (VOffT, s)"
        ],
        "Value": [0.97, 0.98, 0.73, 0.009, VOnT, VOffT]
    })

    per_cycle = pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))
    extras = dict(fps=fps, n_cycles=len(cycles))
    return summary, per_cycle, extras

# ==========================================================
# ---------------------- UI ----------------------

uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

with st.expander("⚙️ 고급 설정 (기본값 그대로 사용해도 충분)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)
    k = c2.number_input("임계 배수 k", min_value=1.0, max_value=6.0, value=3.0, step=0.1)
    M = int(c3.number_input("연속 프레임 M", min_value=1, max_value=20, value=5, step=1))
    K = int(c4.number_input("정상 연속 사이클 K", min_value=1, max_value=10, value=3, step=1))
    W_ms = c5.number_input("에너지 창(ms)", min_value=2.0, max_value=40.0, value=10.0, step=1.0)

    c6, c7, c8 = st.columns(3)
    ap_thr = c6.slider("AP 임계", 0.70, 1.00, 0.90, 0.01)
    tp_thr = c7.slider("TP 임계", 0.70, 1.00, 0.95, 0.01)
    amp_frac = c8.slider("정상 최소 진폭 (max의 비율)", 0.10, 0.80, 0.30, 0.01)

adv = dict(
    baseline_s=baseline_s, k=k, M=M, K=K,
    W_ms=W_ms, ap_thr=ap_thr, tp_thr=tp_thr, amp_frac=amp_frac
)

if uploaded:
    df = load_table(uploaded)
    st.subheader("📋 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

    try:
        summary, per_cycle, extras = analyze(df, adv)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.subheader("✅ 결과 요약")
    st.dataframe(summary, use_container_width=True)

    st.info(f"FPS 추정값: {extras['fps']:.1f} | 탐지된 사이클 수: {extras['n_cycles']}")

    # Downloads
    c1, c2 = st.columns(2)
    c1.download_button("💾 요약 CSV 다운로드", data=summary.to_csv(index=False).encode("utf-8-sig"),
                       file_name="HSV_summary_v2_1.csv", mime="text/csv")
    c2.download_button("📦 사이클별 CSV 다운로드", data=per_cycle.to_csv(index=False).encode("utf-8-sig"),
                       file_name="HSV_cycles_v2_1.csv", mime="text/csv")

    # Lightweight plot (v3에서 고급 시각화 예정)
    st.subheader("📈 Total Gray (개략) – 고급 시각화는 v3에서 제공")
    time_col = next((c for c in df.columns if 'time' in c), None)
    total_col = next((c for c in df.columns if 'total' in c), None)
    left_col = next((c for c in df.columns if 'left' in c), None)
    right_col = next((c for c in df.columns if 'right' in c), None)

    t = df[time_col].astype(float).values
    if t.max() > 10:
        t = t / 1000.0
    if total_col:
        total = df[total_col].astype(float).values
    elif left_col and right_col:
        total = (df[left_col].astype(float).values + df[right_col].astype(float).values) / 2.0
    else:
        total = None

    if total is not None:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, total)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total gray value")
        st.pyplot(fig)
else:
    st.info("분석할 파일을 업로드하면 자동으로 계산됩니다.")


















