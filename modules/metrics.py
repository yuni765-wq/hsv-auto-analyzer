# modules/metrics.py (UTF-8 no BOM clean version)
import numpy as np
from scipy.signal import savgol_filter, find_peaks, welch, hilbert

# --- Adaptive 엔진 토글 ---
USE_ADAPTIVE = True
ADAPTIVE_MODE = "full"  # "full" | "lite"

# Adaptive Threshold Engine 연결
from modules.adaptive_threshold import detect_gat_got_with_adaptive


# ---------- Envelope ----------
def compute_envelope(gray, fs, sg_window=21, sg_poly=3, norm=True):
    """
    gray: ROI gray-value 1D array
    fs:   sampling rate in Hz (frame rate)
    return: amplitude envelope (Hilbert + Savitzky–Golay)
    """
    x = np.asarray(gray, dtype=float)
    if norm:
        x = (x - np.median(x)) / (np.std(x) + 1e-9)
    env = np.abs(hilbert(x))  # amplitude envelope
    env = savgol_filter(env, sg_window, sg_poly, mode="interp")
    return env


# ---------- Threshold ----------
def estimate_baseline_threshold(env, k=1.0):
    base = np.percentile(env, 20)
    sigma = np.median(np.abs(env - base)) * 1.4826
    thr_up = base + k * sigma
    thr_dn = base + 0.5 * k * sigma
    return base, thr_up, thr_dn


# ---------- Periodicity ----------
def periodicity_is_stable(peaks, fs, win_cycles=4, cv_max=0.12):
    if len(peaks) < win_cycles + 1:
        return np.array([], dtype=bool)
    ipi = np.diff(peaks) / fs
    stable = np.zeros_like(ipi, dtype=bool)
    w = win_cycles
    for i in range(w, len(ipi)):
        seg = ipi[i - w:i]
        if np.mean(seg) > 0:
            cv = np.std(seg) / np.mean(seg)
            stable[i] = cv <= cv_max
    stable_peaks = np.r_[np.zeros(1, dtype=bool), stable]
    return stable_peaks


# ---------- GAT/VOnT/GOT/VOffT ----------
def detect_gat_vont_got_vofft(env, fs, k=1.0, min_run_ms=12, win_cycles=4, cv_max=0.12):
    if USE_ADAPTIVE:
        try:
            res = detect_gat_got_with_adaptive(
                env=np.asarray(env, dtype=float),
                fs=float(fs),
                k=k,
                min_run_ms=float(min_run_ms),
                win_cycles=int(win_cycles),
                cv_max=float(cv_max),
            )
            return res["gat_ms"], res["vont_ms"], res["got_ms"], res["vofft_ms"]
        except Exception:
            pass

    base, thr_up, thr_dn = estimate_baseline_threshold(env, k=k)
    n = len(env)
    above = env >= thr_up
    run = 0
    gat_idx = None
    min_run = int((min_run_ms / 1000.0) * fs)
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            gat_idx = i - run + 1
            break

    pk_idx, _ = find_peaks(env, height=thr_up)
    stable_mask = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
    vont_idx = None
    if stable_mask.size:
        stable_peaks = pk_idx[stable_mask]
        if len(stable_peaks):
            vont_idx = stable_peaks[0]

    vofft_idx = None
    got_idx = None
    if len(pk_idx) >= win_cycles + 1:
        stable_fwd = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
        pk_rev = (n - 1) - pk_idx[::-1]
        stable_bwd = periodicity_is_stable(pk_rev, fs, win_cycles=win_cycles, cv_max=cv_max)[::-1]
        both = stable_fwd & stable_bwd
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


# ---------- OID / Tremor ----------
def compute_oid(got_ms, vofft_ms):
    if got_ms is None or vofft_ms is None:
        return np.nan
    return max(0.0, vofft_ms - got_ms)


def tremor_index_psd(env, fs, band=(4.0, 5.0), total=(1.0, 20.0)):
    f, pxx = welch(env, fs=fs, nperseg=min(len(env), 1024), noverlap=512, detrend="constant")
    def bandpower(lo, hi):
        m = (f >= lo) & (f <= hi)
        return np.trapz(pxx[m], f[m]) if np.any(m) else 0.0
    p_band = bandpower(*band)
    p_total = bandpower(*total) + 1e-12
    return p_band / p_total

