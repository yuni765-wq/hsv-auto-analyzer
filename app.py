# HSV Auto Analyzer v2.2 (stable)
# - Keeps AP/TP/AS/PS and robust VOnT/VOffT with smoothing + simple peak/cycle detection
# - One-file input: time + (left/right or total)  [+optional onset/offset columns]
# - Streamlit UI w/ safe defaults

import numpy as np
import pandas as pd
import streamlit as st

# --------- Optional Savitzky–Golay (fallback to moving average if scipy not available)
try:
    from scipy.signal import savgol_filter
except Exception:
    def savgol_filter(x, window_length=11, polyorder=3, mode="interp"):
        x = np.asarray(x, dtype=float)
        w = int(window_length)
        if w < 3:
            return x.copy()
        if w % 2 == 0:
            w += 1
        max_allowed = len(x) if len(x) % 2 == 1 else len(x) - 1
        w = min(max(3, w), max_allowed)
        if w < 3:
            return x.copy()
        pad = w // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(w, dtype=float) / w
        return np.convolve(xp, ker, mode="valid")

# ---------------- UI chrome ----------------
st.set_page_config(page_title="HSV Auto Analyzer v2.2", layout="wide")
st.title("🧠 HSV Auto Analyzer v2.2")
st.caption("Amplitude/Time Periodicity, Amplitude/Phase Symmetry, Voice Onset/Offset Time – 안정 계산 엔진")

# ---------------- Utils ----------------
def _norm_cols(cols):
    return [c.lower().strip().replace(" ", "_") for c in cols]

@st.cache_data
def load_table(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = _norm_cols(df.columns)
    return df

def _moving_rms(diff, w):
    y = np.asarray(diff, dtype=float)
    if w <= 1:
        return np.sqrt(np.maximum(y**2, 0.0))
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(w, dtype=float)
    num = np.convolve(yp**2, ker, mode="valid")
    return np.sqrt(num / float(w))

def _detect_peaks(y, min_dist=3):
    """simple local maxima detection."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)
    dy = np.diff(y)
    peaks = np.where((np.r_[dy, 0] <= 0) & (np.r_[0, dy] > 0))[0]
    if len(peaks) <= 1:
        return peaks
    # enforce minimum distance
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] >= min_dist:
            keep.append(p)
        elif y[p] > y[keep[-1]]:
            keep[-1] = p
    return np.array(keep, dtype=int)

def _build_cycles(t, total, min_frames=5):
    """cycles as [(start_idx, end_idx)] using peaks on smoothed total"""
    if len(total) < (min_frames * 3):
        return []
    # rough peaks -> cycle spans between successive peaks
    peaks = _detect_peaks(total, min_dist=max(3, min_frames))
    if len(peaks) < 3:
        # fallback: fixed windows
        step = max(min_frames * 2, 10)
        spans = []
        i = 0
        while i + step < len(total):
            spans.append((i, i + step))
            i += step
        return spans
    spans = []
    for i in range(len(peaks) - 1):
        s = peaks[i]
        e = peaks[i + 1]
        if e - s >= min_frames:
            spans.append((s, e))
    return spans

def _ap_tp(t, total, cycles):
    """Return AP, TP and arrays"""
    if len(cycles) < 2:
        return np.nan, np.nan, [], []
    periods = []
    amps = []
    for s, e in cycles:
        if e <= s or e > len(t):
            continue
        period = float(t[e] - t[s])
        amp = float(np.nanmax(total[s:e]) - np.nanmin(total[s:e]))
        if period > 0 and amp > 0:
            periods.append(period)
            amps.append(amp)
    if len(periods) < 2 or len(amps) < 2:
        return np.nan, np.nan, periods, amps
    # periodicity (1 - CoV) clipped to [0,1]
    cv_p = np.std(periods, ddof=1) / (np.mean(periods) + 1e-12)
    cv_a = np.std(amps,    ddof=1) / (np.mean(amps)    + 1e-12)
    TP = float(np.clip(1.0 - cv_p, 0.0, 1.0))
    AP = float(np.clip(1.0 - cv_a, 0.0, 1.0))
    return AP, TP, periods, amps

def _as_range(left, right, cycles):
    if left is None or right is None or len(cycles) == 0:
        return np.nan
    ratios = []
    for s, e in cycles:
        if e <= s: 
            continue
        L = float(np.nanmax(left[s:e])  - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        mx = max(L, R)
        if mx > 0:
            ratios.append(min(L, R) / mx)
    return float(np.nanmean(ratios)) if len(ratios) else np.nan

def _ps(left, right, t, cycles):
    if left is None or right is None or len(cycles) == 0:
        return np.nan
    phs = []
    for s, e in cycles:
        if e <= s:
            continue
        li = s + int(np.nanargmax(left[s:e]))
        ri = s + int(np.nanargmax(right[s:e]))
        Ti = float(t[e] - t[s])
        if Ti > 0:
            lag = abs(float(t[li] - t[ri])) / Ti   # 0..1
            phs.append(max(0.0, 1.0 - lag))       # 1 == in-phase
    return float(np.nanmean(phs)) if len(phs) else np.nan

def _first_steady_from(t, total, cycles, g_amp, i_from, K=10, ap_thr=0.9, tp_thr=0.9, amp_frac=0.3):
    """very simple steady finder: pick first cycle after i_from whose amp >= amp_frac*g_amp"""
    for s, e in cycles:
        if s < i_from:
            continue
        amp = float(np.nanmax(total[s:e]) - np.nanmin(total[s:e]))
        if amp >= amp_frac * g_amp:
            return s, float(t[s])
    return None, None

def _last_steady_before_end(t, total, cycles, g_amp, K=10, ap_thr=0.9, tp_thr=0.9, amp_frac=0.3):
    """last cycle whose amp >= amp_frac*g_amp"""
    for s, e in reversed(cycles):
        amp = float(np.nanmax(total[s:e]) - np.nanmin(total[s:e]))
        if amp >= amp_frac * g_amp:
            return e, float(t[e])
    return None, None

# ================================
# Main analyzer
# ================================
def analyze(df, adv):
    # ---- column mapping ----
    cols = df.columns.tolist()
    time_col   = next((c for c in cols if 'time'  in c), None)
    left_col   = next((c for c in cols if 'left'  in c), None)
    right_col  = next((c for c in cols if 'right' in c), None)
    total_col  = next((c for c in cols if 'total' in c), None)
    onset_col  = next((c for c in cols if 'onset' in c), None)
    offset_col = next((c for c in cols if 'offset'in c), None)

    if time_col is None:
        empty = pd.DataFrame()
        return (pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0))

    # ---- signals ----
    t = df[time_col].astype(float).values
    if t.max() > 10:  # ms → s
        t = t / 1000.0

    if total_col:
        total = df[total_col].astype(float).values
    elif left_col and right_col:
        total = (df[left_col].astype(float).values + df[right_col].astype(float).values) / 2.0
    else:
        empty = pd.DataFrame()
        return (pd.DataFrame({"Parameter": [], "Value": []}), empty, dict(fps=np.nan, n_cycles=0))

    left  = df[left_col].astype(float).values  if left_col  else None
    right = df[right_col].astype(float).values if right_col else None

    # ---- FPS ----
    dt  = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fps = 1.0 / dt if dt > 0 else 1500.0

    # ---- smoothing for robust cycles ----
    signal = np.asarray(total, dtype=float)
    base_w  = int(max(7, min(21, round(fps * 0.007))))  # ≈7ms
    win_len = base_w if base_w % 2 == 1 else base_w + 1
    if len(signal) <= 3:
        win_len = 3
    else:
        max_allowed = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        win_len = min(max(3, win_len), max_allowed)
    smoothed = savgol_filter(signal, window_length=win_len, polyorder=3, mode="interp")

    # ---- cycles ----
    min_frames = max(int(0.002 * fps), 5)
    cycles = _build_cycles(t, smoothed, min_frames=min_frames)

    # ---- AP/TP/AS/PS ----
    AP, TP, periods, amps = _ap_tp(t, smoothed, cycles)
    AS = _as_range(left, right, cycles)
    PS = _ps(left, right, t, cycles)

    # ---------- VOnT / VOffT (개선판: 조용구간 + 런 길이 조건) ----------
    # 1) 에너지 계산 (이미 smoothing 된 smoothed 기반)
    diff_total = np.abs(np.diff(smoothed, prepend=smoothed[0]))
    W = max(int(round((adv['W_ms'] / 1000.0) * fps)), 3)        # 에너지 창
    E_total = _moving_rms(diff_total, W)

    onset_series  = df[onset_col].astype(float).values  if onset_col  else None
    offset_series = df[offset_col].astype(float).values if offset_col else None

    E_on  = _moving_rms(np.abs(np.diff(onset_series,  prepend=onset_series[0])),  W) if onset_series  is not None else E_total
    E_off = _moving_rms(np.abs(np.diff(offset_series, prepend=offset_series[0])), W) if offset_series is not None else E_total

    # 2) 베이스라인에서 임계값(μ + k·σ)
    nB = max(int(round(adv['baseline_s'] * fps)), 5)
    def _thr(E):
        base = E[:min(nB, len(E))]
        mu0  = float(np.mean(base)) if len(base) else 0.0
        s0   = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        return mu0 + adv['k'] * s0

    thr_on, thr_off = _thr(E_on), _thr(E_off)

    # 3) 일정 길이(min_len=M) 이상 지속되는 run(start~end) 추출
    def _runs(mask_bool, min_len):
        m = mask_bool.astype(int)
        edges = np.diff(np.r_[0, m, 0])
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0] - 1
        keep   = np.where((ends - starts + 1) >= min_len)[0]
        return starts[keep], ends[keep]

    runs_on_st, runs_on_en   = _runs(E_on  > thr_on,  adv['M'])
    runs_off_st, runs_off_en = _runs(E_off > thr_off, adv['M'])

    # 4) '조용구간' 조건: run 전/후로 조용한 기간(quiet_ms) 확보
    quiet_ms = float(adv.get('quiet_ms', 30.0))                  # UI가 없어도 기본 30ms
    quiet_n  = max(int(round(quiet_ms / 1000.0 * fps)), 1)

    def _has_quiet_before(arr, idx, qn):
        s = max(0, idx - qn)
        return bool(np.all(arr[s:idx] <= thr_on)) if idx > 0 else True

    def _has_quiet_after(arr, idx, qn):
        e = min(len(arr), idx + qn)
        return bool(np.all(arr[idx:e] <= thr_off)) if idx < len(arr) else True

 # === 온셋: 램프-업 시작 기반 ===
rise_pre_ms      = float(adv.get('rise_pre_ms', 100.0))
rise_post_ms     = float(adv.get('rise_post_ms',  60.0))
rise_frac        = float(adv.get('rise_frac',      0.15))
rise_sustain_ms  = float(adv.get('rise_sustain_ms', 8.0))

pre_n      = max(int(round(rise_pre_ms     / 1000.0 * fps)), 1)
post_n     = max(int(round(rise_post_ms    / 1000.0 * fps)), 1)
sustain_n  = max(int(round(rise_sustain_ms / 1000.0 * fps)), 1)

win_lo = max(0, i_steady - pre_n)
win_hi = min(len(smoothed), i_steady + post_n)
seg    = smoothed[win_lo:win_hi]

i_move = None
if len(seg) >= sustain_n + 2:
    b_local = float(np.min(seg))
    p_local = float(np.max(seg))
    thr_rise = b_local + rise_frac * (p_local - b_local)

    slope = np.convolve(
        np.diff(smoothed, prepend=smoothed[0]),
        np.ones(sustain_n, dtype=float) / sustain_n,
        mode="same"
    )
    for j in range(win_lo, i_steady):
        if (smoothed[j] >= thr_rise) and np.all(slope[j:j + sustain_n] > 0):
            i_move = j
            break

# 실패 시 on-run fallback
if i_move is None:
    i_move = int(runs_on_st[0]) if len(runs_on_st) else (cycles[0][0] if len(cycles) else 0)
t_move = float(t[i_move]) if len(t) else np.nan

# === 오프셋: 감쇠 시작 기반 ===
look_ms    = float(adv.get('look_ms', 40.0))
decay_frac = float(adv.get('decay_frac', 0.15))
decay_sustain_ms = float(adv.get('decay_sustain_ms', 8.0))

look_n    = max(int(round(look_ms          / 1000.0 * fps)), 1)
sustain_n = max(int(round(decay_sustain_ms / 1000.0 * fps)), 1)

seg_off = smoothed[i_last:i_last + look_n] if i_last < len(smoothed) else np.array([])
if len(seg_off) > 0:
    pmax_local = float(np.max(seg_off))
else:
    pmax_local = float(np.max(smoothed[i_last:])) if i_last < len(smoothed) else float(np.max(smoothed))

decay_threshold = pmax_local * (1.0 - decay_frac)

slope_off = np.convolve(
    np.diff(smoothed, prepend=smoothed[0]),
    np.ones(sustain_n, dtype=float) / sustain_n,
    mode="same"
)

i_end = None
for i in range(i_last, len(smoothed) - sustain_n):
    if (smoothed[i] <= decay_threshold) and np.all(slope_off[i:i + sustain_n] < 0):
        i_end = i
        break

if i_end is None:
    i_end = cycles[-1][1]
t_end = float(t[min(i_end, len(t) - 1)])

# === 최종 VOnT/VOffT (초) ===
VOnT  = float(t_steady - t_move) if (not np.isnan(t_steady) and not np.isnan(t_move)) else np.nan
VOffT = float(t_end    - t_last) if (not np.isnan(t_end)    and not np.isnan(t_last)) else np.nan

if VOnT  is not None and VOnT  < 1e-4: VOnT  = 0.0
if VOffT is not None and VOffT < 1e-4: VOffT = 0.0

# ---- per-cycle detail (optional stub, empty for now) ----
per_cycle = pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))

# ---- summary & extras ----
VOnT_ms  = VOnT  * 1000.0 if VOnT  is not None and not np.isnan(VOnT)  else np.nan
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

  
# ================================
# Streamlit UI
# ================================
uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

with st.expander("⚙️ 고급 설정 (기본값 그대로 사용해도 충분)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)
    k          = c2.number_input("임계 배수 k",     min_value=1.0,  max_value=6.0,  value=3.0,  step=0.1)
    M          = c3.number_input("연속 프레임 M",   min_value=1,    max_value=20,   value=5,    step=1)
    ap_thr     = c4.slider("AP 임계값", 0.70, 1.00, 0.90, 0.01)
    tp_thr     = c5.slider("TP 임계값", 0.70, 1.00, 0.95, 0.01)
    c6, c7, c8 = st.columns(3)
    amp_frac   = c6.slider("정상 최소 진폭(백분율)", 0.10, 0.80, 0.30, 0.01)
    W_ms       = c7.slider("에너지 창(ms)", 2.0, 40.0, 10.0, 1.0)

adv = dict(
    baseline_s=baseline_s, k=k, M=M, K=10,
    W_ms=W_ms, ap_thr=ap_thr, tp_thr=tp_thr, amp_frac=amp_frac
)

if uploaded is None:
    st.info("분석할 파일을 업로드해 주세요.")
    st.stop()

df = load_table(uploaded)
summary, per_cycle, extras = analyze(df, adv)

st.subheader("✅ 결과 요약")
st.dataframe(summary, use_container_width=True)
st.write(f"FPS: {extras['fps']:.1f}, 검출된 사이클 수: {extras['n_cycles']}")
















