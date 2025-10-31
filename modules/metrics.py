# modules/metrics.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, welch, hilbert

# --- Adaptive 엔진 사용/모드 설정 ---
USE_ADAPTIVE = True
ADAPTIVE_MODE = "full"

# Adaptive Threshold Engine (절대경로 임포트 - 들여쓰기 금지)
from modules.adaptive_threshold import detect_gat_got_with_adaptive

# ---------- Envelope ----------
def compute_envelope(gray, fs, sg_window=21, sg_poly=3, norm=True):
    """
    gray: ROI gray-value 1D array
    fs: sampling rate in Hz (frame rate)
    return: amplitude envelope (Hilbert + Savitzky–Golay)
    """
    x = np.asarray(gray, dtype=float)
    if norm:
        x = (x - np.median(x)) / (np.std(x) + 1e-9)
    env = np.abs(hilbert(x))                    # amplitude envelope
    env = savgol_filter(env, sg_window, sg_poly, mode="interp")
    return env


# ---------- Threshold (baseline/hysteresis) ----------
def estimate_baseline_threshold(env, k=1.0):
    """
    env: amplitude envelope
    k:  scale for robust spread
    returns:
      base:   robust baseline (P20)
      thr_up: upper threshold   (base + k*sigma)
      thr_dn: lower threshold   (base + 0.5*k*sigma)
    """
    base = np.percentile(env, 20)                           # robust baseline
    sigma = np.median(np.abs(env - base)) * 1.4826          # robust spread (MAD→σ)
    thr_up = base + k * sigma
    thr_dn = base + 0.5 * k * sigma                         # hysteresis lower
    return base, thr_up, thr_dn


# ---------- Periodicity quality ----------
def periodicity_is_stable(peaks, fs, win_cycles=4, cv_max=0.12):
    """
    peaks: indices of local maxima above threshold
    return: boolean mask per peak where stability criterion satisfied
    """
    if len(peaks) < win_cycles + 1:
        return np.array([], dtype=bool)
    ipi = np.diff(peaks) / fs  # inter-peak intervals (s)
    stable = np.zeros_like(ipi, dtype=bool)
    w = win_cycles
    for i in range(w, len(ipi)):
        seg = ipi[i - w:i]
        if np.mean(seg) > 0:
            cv = np.std(seg) / np.mean(seg)
            stable[i] = cv <= cv_max
    # align back to peaks (mark peaks after stability found)
    stable_peaks = np.r_[np.zeros(1, dtype=bool), stable]
    return stable_peaks


# ---------- Onset/Offset markers (클래식 + Adaptive 분기) ----------
def detect_gat_vont_got_vofft(env, fs, k=1.0, min_run_ms=12, win_cycles=4, cv_max=0.12):
    """
    Returns (ms): GAT, VOnT, GOT, VOffT

    Definitions (classic path):
      GAT  = first time env crosses thr_up with persistence >= min_run_ms
      VOnT = first peak time where periodicity becomes stable
      GOT  = first time after stable phonation where env falls below thr_dn
             and loses stability for >= min_run_ms
      VOffT= last peak time before stable periodicity disappears completely
    """
    # --- Adaptive 엔진 분기 (성공하면 즉시 반환, 실패 시 아래 클래식 폴백) ---
    if USE_ADAPTIVE:
        try:
            res = detect_gat_got_with_adaptive(
                env=np.asarray(env, dtype=float),
                fs=float(fs),
                k=k,
                min_run_ms=min_run_ms,
                win_cycles=win_cycles,
                cv_max=cv_max,
            )
            # 이 함수의 반환 순서를 유지: (GAT, VOnT, GOT, VOffT)
            return res["gat_ms"], res["vont_ms"], res["got_ms"], res["vofft_ms"]
        except Exception:
            # 폴백: 아래 클래식 경로 계속 수행
            pass

    # ---------- 클래식 경로 ----------
    base, thr_up, thr_dn = estimate_baseline_threshold(env, k=k)
    n = len(env)

    # Crossings above upper threshold → GAT
    above = env >= thr_up
    run = 0
    gat_idx = None
    min_run = int((min_run_ms / 1000.0) * fs)
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            gat_idx = i - run + 1
            break

    # Peaks and stability → VOnT
    pk_idx, _ = find_peaks(env, height=thr_up)
    stable_peaks_mask = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
    vont_idx = None
    if stable_peaks_mask.size:
        stable_peaks = pk_idx[stable_peaks_mask]
        if len(stable_peaks):
            vont_idx = stable_peaks[0]

    # Tail side: last stable peak → VOffT, then persistent drop below thr_dn → GOT
    vofft_idx = None
    got_idx = None
    if len(pk_idx) >= win_cycles + 1:
        stable_peaks_mask_fwd = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
        pk_rev = (n - 1) - pk_idx[::-1]
        stable_peaks_mask_bwd = periodicity_is_stable(pk_rev, fs, win_cycles=win_cycles, cv_max=cv_max)[::-1]
        both = stable_peaks_mask_fwd & stable_peaks_mask_bwd
        stable_peaks = pk_idx[both]
        if len(stable_peaks):
            vofft_idx = stable_peaks[-1]
            start = vofft_idx + 1
            below = env < thr_dn
            run = 0
            for i in range(start, n):
                run = run + 1 if below[i] else 0
                if run >= min_run:
                    got_idx = i - run + 1
                    break

    def ms(idx):
        return None if idx is None else 1000.0 * (idx / fs)

    return ms(gat_idx), ms(vont_idx), ms(got_idx), ms(vofft_idx)


# ---------- OID and Tremor ----------
def compute_oid(got_ms, vofft_ms):
    if got_ms is None or vofft_ms is None:
        return np.nan
    return max(0.0, vofft_ms - got_ms)


def tremor_index_psd(env, fs, band=(4.0, 5.0), total=(1.0, 20.0)):
    """
    Welch PSD on the amplitude envelope.
    Returns power ratio in band over total.
    """
    f, pxx = welch(env, fs=fs, nperseg=min(len(env), 1024), noverlap=512, detrend='constant')
    def bandpower(lo, hi):
        m = (f >= lo) & (f <= hi)
        return np.trapz(pxx[m], f[m]) if np.any(m) else 0.0
    p_band = bandpower(*band)
    p_total = bandpower(*total) + 1e-12
    return p_band / p_total






