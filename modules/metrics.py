# modules/metrics.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, welch, hilbert

# --- Adaptive 엔진 사용/모드 설정 ---
USE_ADAPTIVE = True              # 필요 시 False로 레거시만 사용
ADAPTIVE_MODE = "full"           # "full" | "lite"

# Adaptive Threshold Engine 연동 (모듈 경로 유연화)
try:
    # 정식 모듈 경로
    from modules.adaptive_threshold import detect_gat_got_with_adaptive
except Exception:
    # 로컬/백업 경로
    from adaptive_threshold import detect_gat_got_with_adaptive

# ---------- Envelope ----------
def compute_envelope(gray, fs, sg_window=21, sg_poly=3, norm=True):
    """
    gray: ROI gray-value 1D array
    fs: sampling rate in Hz (frame rate)
    return: amplitude envelope (Hilbert + Savitzky–Golay)
    """
    x = np.asarray(gray).astype(float)
    if norm:
        x = (x - np.median(x)) / (np.std(x) + 1e-9)
    env = np.abs(hilbert(x))  # amplitude envelope
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
    base = np.percentile(env, 20)  # robust baseline
    sigma = np.median(np.abs(env - base)) * 1.4826  # robust spread
    thr_up = base + k * sigma
    thr_dn = base + 0.5 * k * sigma  # hysteresis lower
    return base, thr_up, thr_dn


# ---------- Periodicity quality ----------
def periodicity_is_stable(peaks, fs, win_cycles=4, cv_max=0.12):
    """
    peaks: indices of local maxima above threshold
    return: boolean mask per peak where stability criterion satisfied
    """
    if len(peaks) < win_cycles + 1:
        return np.array([], dtype=bool)
    ipi = np.diff(peaks) / fs  # inter-peak intervals in seconds
    stable = np.zeros_like(ipi, dtype=bool)
    w = win_cycles
    for i in range(w, len(ipi)):
        seg = ipi[i-w:i]
        if np.mean(seg) > 0:
            cv = np.std(seg) / np.mean(seg)
            stable[i] = cv <= cv_max
    # align back to peaks (mark peaks after stability found)
    stable_peaks = np.r_[np.zeros(1, dtype=bool), stable]
    return stable_peaks


# ---------- Onset/Offset markers (클래식) ----------
def detect_gat_vont_got_vofft(env, fs, k=1.0, min_run_ms=12, win_cycles=4, cv_max=0.12):
    """
    Returns times in milliseconds: GAT, VOnT, GOT, VOffT
    Definitions:
      GAT  = first time env crosses thr_up with persistence >= min_run_ms
      VOnT = first peak time where periodicity becomes stable
      GOT  = first time after stable phonation where env falls below thr_dn
             and loses stability for >= min_run_ms
      VOffT= last peak time before stable periodicity disappears completely
    """
        # --- [ADD] Adaptive 엔진 분기 (성공하면 바로 반환, 실패시 폴백) ---
    if USE_ADAPTIVE:
        try:
            res = detect_gat_got_with_adaptive(
                env=np.asarray(env, dtype=float),
                fs=float(fs),
                k=k,
                min_run_ms=min_run_ms,
                win_cycles=win_cycles,
                cv_max=cv_max,
                # adaptive_params / reference_marks 필요 시 전달 가능
            )
            # 이 함수의 반환 순서를 유지: (GAT, VOnT, GOT, VOffT)
            return res["gat_ms"], res["vont_ms"], res["got_ms"], res["vofft_ms"]
        except Exception:
            # 폴백: 아래 클래식 경로 계속 수행
            pass
    base, thr_up, thr_dn = estimate_baseline_threshold(env, k=k)
    n = len(env)
    t = np.arange(n) / fs

    # Crossings above threshold
    above = env >= thr_up
    run = 0
    gat_idx = None
    min_run = int((min_run_ms / 1000.0) * fs)
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            gat_idx = i - run + 1
            break

    # Peaks and stability
    pk_idx, _ = find_peaks(env, height=thr_up)
    stable_peaks_mask = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
    vont_idx = None
    if stable_peaks_mask.size:
        stable_peaks = pk_idx[stable_peaks_mask]
        if len(stable_peaks):
            vont_idx = stable_peaks[0]

    # For offset we operate from the tail
    # Find last stretch of stable periodicity
    vofft_idx = None
    got_idx = None
    if len(pk_idx) >= win_cycles + 1:
        # mark stability forward
        stable_peaks_mask_fwd = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
        # mark stability backward by reversing the signal
        pk_rev = (n - 1) - pk_idx[::-1]
        stable_peaks_mask_bwd = periodicity_is_stable(pk_rev, fs, win_cycles=win_cycles, cv_max=cv_max)[::-1]

        # a peak is inside stable phonation if both fwd and bwd consider it stable neighborhood
        both = stable_peaks_mask_fwd & stable_peaks_mask_bwd
        stable_peaks = pk_idx[both]
        if len(stable_peaks):
            vofft_idx = stable_peaks[-1]  # last stable peak time
            # search forward from last stable peak to detect persistent drop below thr_dn
            start = vofft_idx + 1
            below = env < thr_dn
            run = 0
            for i in range(start, n):
                run = run + 1 if below[i] else 0
                if run >= min_run:
                    got_idx = i - run + 1  # onset of instability below lower hysteresis
                    break

    # Convert to milliseconds with safety
    def ms(idx):
        return None if idx is None else 1000.0 * (idx / fs)

    return ms(gat_idx), ms(vont_idx), ms(got_idx), ms(vofft_idx)


# ---------- Onset/Offset markers (Adaptive v3.3) ----------
from typing import Optional

def detect_gat_got_with_adaptive(env: np.ndarray,
                                 fs: float,
                                 k: float = 1.0,
                                 min_run_ms: float = 12.0,
                                 win_cycles: int = 4,
                                 cv_max: float = 0.12,
                                 adaptive_params: Optional[AdaptiveParams] = None,
                                 reference_marks=None):
    # adaptive_params가 안 들어오면 새로 생성
    if adaptive_params is None:
        adaptive_params = AdaptiveParams()

    """
    Adaptive Threshold Engine 기반으로 GAT/GOT 계산.
    1) baseline에서 upper-threshold(thr_up)를 base_threshold로 사용
    2) adaptive_optimize로 지역 임계값 생성(thr_local)
    3) 교차 기반으로 GAT/GOT 산출
    4) QC 메트릭 반환
    """
    # 1) baseline/hysteresis 계산
    base, thr_up, thr_dn = estimate_baseline_threshold(env, k=k)

    # 2) Adaptive 엔진 실행 (base_threshold는 thr_up 사용)
    ares = adaptive_optimize(env, sr_hz=fs, base_threshold=float(thr_up),
                             params=adaptive_params, reference_marks=reference_marks)
    thr_local = ares.thr_local

    # 3) Adaptive 교차 기반 GAT/GOT
    cross_up = np.where((env[:-1] < thr_local[:-1]) & (env[1:] >= thr_local[1:]))[0]
    cross_dn = np.where((env[:-1] >= thr_local[:-1]) & (env[1:] <  thr_local[1:]))[0]

    # GAT: thr_local 상향 교차가 일정 시간 이상 유지되는 최초 시점
    min_run = int((min_run_ms / 1000.0) * fs)
    gat_idx = None
    run = 0
    above_local = env >= thr_local
    for i, a in enumerate(above_local):
        run = run + 1 if a else 0
        if run >= min_run:
            gat_idx = i - run + 1
            break

            
    # VOnT: 안정적 주기성의 첫 peak
    pk_idx, _ = find_peaks(env)  # height 인자 제거
    # thr_local보다 위에 있는 피크만 남김
    pk_idx = pk_idx[env[pk_idx] >= thr_local[pk_idx]]
    
    vont_idx = None
    if len(pk_idx):
        stable_mask = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
        if stable_mask.size and np.any(stable_mask):
            vont_idx = pk_idx[stable_mask][0]
            

    # VOffT/GOT: 안정적 구간의 마지막 peak와 thr_dn(down) 기준 이탈
    vofft_idx = None
    got_idx = None
    if len(pk_idx) >= win_cycles + 1:
        # fwd/bwd 안정성 교집합으로 stable peaks 추림
        stable_fwd = periodicity_is_stable(pk_idx, fs, win_cycles=win_cycles, cv_max=cv_max)
        pk_rev = (len(env) - 1) - pk_idx[::-1]
        stable_bwd = periodicity_is_stable(pk_rev, fs, win_cycles=win_cycles, cv_max=cv_max)[::-1]
        stable_both = stable_fwd & stable_bwd
        stable_peaks = pk_idx[stable_both]
        if len(stable_peaks):
            vofft_idx = stable_peaks[-1]
            # thr_dn은 baseline 기반 하강 히스테리시스 유지
            start = vofft_idx + 1
            below_dn = env < thr_dn
            run = 0
            for i in range(start, len(env)):
                run = run + 1 if below_dn[i] else 0
                if run >= min_run:
                    got_idx = i - run + 1
                    break

    def ms(idx):
        return None if idx is None else 1000.0 * (idx / fs)

    return {
        "gat_ms": ms(gat_idx),
        "got_ms": ms(got_idx),
        "vont_ms": ms(vont_idx),
        "vofft_ms": ms(vofft_idx),
        "thr_local": thr_local,
        "adaptive_qc": {
            "noise_ratio": ares.noise_ratio,
            "global_gain": ares.global_gain,
            "iters": ares.iters,
            "est_rmse": ares.est_rmse,
            "qc_label": ares.qc_label,
        },
        "preset": "Adaptive v3.3"
    }


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





