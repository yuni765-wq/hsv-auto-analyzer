# ---------------------------------------------------------------
# adaptive_threshold.py – HSV Auto Analyzer v3.3
# Isaka × Lian – Adaptive Threshold Engine (Residual Noise + QC)
# deps: numpy
# ---------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

@dataclass
class AdaptiveParams:
    # Residual noise detection
    win_ms: float = 12.0          # sliding RMS window (ms)
    hop_ms: float = 6.0           # hop (ms)
    min_seg_ms: float = 18.0      # 최소 지속 길이
    snr_db_cut: float = 6.0       # SNR dB 컷 (낮으면 노이즈 구간)
    slope_z_cut: float = -0.5     # 기울기 z-score 컷 (하강변/플랫 구간)
    merge_gap_ms: float = 10.0    # 인접 노이즈 구간 병합 허용 간격

    # Adaptive correction
    max_gain: float = 1.8         # 지역 임계값 상향 배율 상한
    min_gain: float = 0.85        # 지역 임계값 하향 배율 하한
    var_ref_eps: float = 1e-6     # 분산 안정화 상수
    gain_smooth_ms: float = 20.0  # 보정 게인 스무딩 윈도 길이

    # Feedback loop
    target_rmse: float = 0.12     # 품질 목표 (ex: 수동 표식과의 오차)
    max_iters: int = 4            # 반복 최적화 횟수 상한
    global_gain_step: float = 0.12# 전역 임계값 조정 스텝

@dataclass
class AdaptiveResult:
    thr_base: float                   # 입력 기본 임계값
    thr_local: np.ndarray             # 샘플별 지역 임계값
    noise_mask: np.ndarray            # 잔류 노이즈 마스크 (bool, len=N)
    noise_ratio: float                # 전체 길이 대비 노이즈 구간 비율
    global_gain: float                # 전역 임계값 보정 배율
    iters: int                        # 수행된 반복 수
    est_rmse: Optional[float]         # 추정 RMSE(참값 없을 시 내부 프록시)
    qc_label: str                     # 🟢/🟡/🔴 라벨 텍스트

def _frame_signal(x: np.ndarray, sr_hz: float, win_ms: float, hop_ms: float) -> Tuple[np.ndarray, int]:
    N = len(x)
    win = int(round(sr_hz * win_ms / 1000.0))
    hop = int(round(sr_hz * hop_ms / 1000.0))
    win = max(win, 1); hop = max(hop, 1)
    n_frames = 1 + max(0, (N - win) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    if len(frames) > n_frames:
        frames = frames[:n_frames]
    return frames, hop

def _rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

def _snr_db(envelope: np.ndarray, frames: np.ndarray) -> np.ndarray:
    # frame RMS / 전체 신호 RMS 비
    sig_rms = np.sqrt(np.mean(envelope**2) + 1e-12)
    fr_rms = _rms(frames)
    snr = 20.0 * np.log10(np.maximum(fr_rms, 1e-12) / max(sig_rms, 1e-12))
    return snr

def _slope_z(envelope: np.ndarray, frames: np.ndarray) -> np.ndarray:
    # 프레임 평균의 일차차분 → z-score
    mu = frames.mean(axis=1)
    d = np.diff(mu, prepend=mu[0])
    if np.std(d) < 1e-12:
        return np.zeros_like(d)
    return (d - np.mean(d)) / (np.std(d) + 1e-12)

def _expand_to_samples(frame_mask: np.ndarray, N: int, hop: int, win: int) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    idx = 0
    for f, flag in enumerate(frame_mask):
        if flag:
            s = idx
            e = min(idx + win, N)
            mask[s:e] = True
        idx += hop
        if idx >= N: break
    return mask

def _merge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    # True 구간 사이의 짧은 0 구간을 1로 메움
    N = len(mask)
    i = 0
    out = mask.copy()
    while i < N:
        while i < N and out[i]:
            i += 1
        gap_start = i
        while i < N and not out[i]:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        if 0 < gap_len <= max_gap:
            out[gap_start:gap_end] = True
    return out

def _min_len_filter(mask: np.ndarray, min_len: int) -> np.ndarray:
    # True 구간의 최소 길이 미만은 0으로 제거
    N = len(mask)
    out = np.zeros(N, dtype=bool)
    i = 0
    while i < N:
        while i < N and not mask[i]:
            i += 1
        s = i
        while i < N and mask[i]:
            i += 1
        e = i
        if e - s >= min_len:
            out[s:e] = True
    return out

def _smooth_gain(gain: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return gain
    pad = win // 2
    x = np.pad(gain, (pad, pad), mode='edge')
    kernel = np.ones(win) / win
    sm = np.convolve(x, kernel, mode='same')[pad:-pad]
    return sm

def detect_residual_noise(envelope: np.ndarray, sr_hz: float, params: AdaptiveParams) -> np.ndarray:
    """프레임 기반 SNR + slope z를 활용한 잔류 노이즈 구간 검출(mask)."""
    frames, hop = _frame_signal(envelope, sr_hz, params.win_ms, params.hop_ms)
    win = frames.shape[1]
    snr = _snr_db(envelope, frames)
    sz = _slope_z(envelope, frames)
    # 노이즈 후보: SNR 낮고(snr < cut) & 기울기 하강/플랫(sz < cut)
    cand = (snr < params.snr_db_cut) & (sz < params.slope_z_cut)
    mask = _expand_to_samples(cand, len(envelope), hop, win)
    # 병합 & 최소 길이 유지
    merge_gap = int(round(sr_hz * params.merge_gap_ms / 1000.0))
    min_len = int(round(sr_hz * params.min_seg_ms / 1000.0))
    mask = _merge_short_gaps(mask, merge_gap)
    mask = _min_len_filter(mask, min_len)
    return mask

def build_local_threshold(envelope: np.ndarray,
                          base_threshold: float,
                          noise_mask: np.ndarray,
                          params: AdaptiveParams) -> np.ndarray:
    """노이즈 마스크 구간에서 지역 분산 기반 가변 임계값 생성."""
    N = len(envelope)
    thr = np.full(N, float(base_threshold), dtype=float)
    # 지역 분산 → 상대 게인
    # 분산이 낮으면 상향 보정(=더 보수적), 분산이 높으면 완화
    local_var = np.ones(N) * params.var_ref_eps
    # 간단한 이동분산(여기선 9샘플 윈도우): 실데이터에 맞춰 조정 가능
    win = 9
    pad = win // 2
    x = np.pad(envelope, (pad, pad), mode="edge")
    for i in range(N):
        seg = x[i:i+win]
        local_var[i] = np.var(seg) + params.var_ref_eps

    var_med = np.median(local_var)
    raw_gain = np.clip(np.sqrt(var_med / local_var), params.min_gain, params.max_gain)
    # 노이즈 구간만 적용
    thr *= np.where(noise_mask, raw_gain, 1.0)

    # 게인 스무딩
    smooth_win = max(1, int(round((params.gain_smooth_ms/1000.0) * 200.0)))  # 샘플링 200Hz 가정 → 후속에서 실제 sr로 보정 가능
    thr = _smooth_gain(thr, smooth_win)
    return thr

def proxy_rmse(envelope: np.ndarray, thr_local: np.ndarray) -> float:
    """
    참 표식(수동 GAT/GOT)이 없을 때 사용할 내부 프록시:
    임계값 교차 지점 주변의 에너지/기울기 불일치도를 RMSE처럼 근사.
    """
    cross = (envelope[:-1] < thr_local[:-1]) & (envelope[1:] >= thr_local[1:])
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return 0.5  # 교차 없음: 보수적으로 중간값 고정
    diffs = []
    for i in idx:
        # 교차점 ±3 구간에서 임계값 대비 여유(margin)과 기울기 부조화 측정
        s = max(0, i-3); e = min(len(envelope)-1, i+3)
        margin = np.abs(envelope[s:e+1] - thr_local[s:e+1]).mean()
        slope = np.abs(np.diff(envelope[s:e+1])).mean()
        diffs.append(margin * 0.6 + slope * 0.4)
    return float(np.sqrt(np.mean(np.square(diffs)) + 1e-12))

def quality_label(noise_ratio: float, rmse: float) -> str:
    """QC 라벨 규칙: 노이즈 비율과 RMSE 모두 고려."""
    # 예시 기준: 가볍게 시작, 이후 데이터 보며 튜닝
    if (noise_ratio <= 0.25 and rmse <= 0.12):
        return "🟢 High"
    if (noise_ratio <= 0.45 and rmse <= 0.22):
        return "🟡 Medium"
    return "🔴 Low"

def adaptive_optimize(envelope: np.ndarray,
                      sr_hz: float,
                      base_threshold: float,
                      params: Optional[AdaptiveParams] = None,
                      reference_marks: Optional[Dict[str, float]] = None) -> AdaptiveResult:
                      thr_lo_ratio_default = 0.85    
    """
    Adaptive 파이프라인:
    1) 잔류 노이즈 감지
    2) 지역 임계값 구축
    3) 피드백 루프 (참값 있으면 RMSE 실제 계산, 없으면 proxy_rmse)
    """
    if params is None:
        params = AdaptiveParams()

    # 1) 초기 노이즈
    noise_mask = detect_residual_noise(envelope, sr_hz, params)
    thr_local = build_local_threshold(envelope, base_threshold, noise_mask, params)

    # 2) RMSE 평가: 참값(GAT/GOT 등)이 있으면 활용 (여기선 placeholder)
    def eval_rmse(thr):
        if reference_marks and all(k in reference_marks for k in ("GAT", "GOT")):
            # TODO: envelope-임계값 교차로 산출한 GAT/GOT vs 참값 비교 RMSE
            # 여기선 간단히 proxy로 대체
            return proxy_rmse(envelope, thr)
        else:
            return proxy_rmse(envelope, thr)

    rmse = eval_rmse(thr_local)
    global_gain = 1.0
    it = 0

    # 3) Feedback loop: 목표치까지 전역 배율로 미세 조정
    while (rmse > params.target_rmse) and (it < params.max_iters):
        it += 1
        # 전역적으로 약간 상향 → 더 보수적(덜 민감)으로 만들어 노이즈 교차 감소
        global_gain *= (1.0 + params.global_gain_step)
        thr_local = thr_local * (1.0 + params.global_gain_step)
        rmse = eval_rmse(thr_local)

    noise_ratio = float(np.mean(noise_mask.astype(float)))
    label = quality_label(noise_ratio, rmse)

    return AdaptiveResult(
        thr_base=float(base_threshold),
        thr_local=thr_local,
        noise_mask=noise_mask,
        noise_ratio=noise_ratio,
        global_gain=global_gain,
        iters=it,
        est_rmse=float(rmse),
        qc_label=label
    )

# ===============================================================
# Wrapper: detect_gat_got_with_adaptive
# - app.py에서 바로 호출하는 엔트리 포인트
#   res = detect_gat_got_with_adaptive(env, fs, k=1.0, min_run_ms=12, ...)
#   -> { "gat_ms": ..., "got_ms": ..., "vont_ms": ..., "vofft_ms": ...,
#        "preset": "Adaptive v3.3",
#        "adaptive_qc": { "qc_label": ..., "noise_ratio": ..., "est_rmse": ...,
#                         "global_gain": ..., "iters": ... } }
# ===============================================================
def _estimate_baseline_threshold(env: np.ndarray, k: float = 1.0):
    """robust base + hysteresis (상단/하단 임계값)"""
    base = np.percentile(env, 20)
    mad  = np.median(np.abs(env - base)) * 1.4826
    thr_up = base + k * mad
    thr_dn = base + 0.5 * k * mad
    return float(base), float(thr_up), float(thr_dn)

def _first_persistent_index(mask: np.ndarray, min_run: int) -> int | None:
    """mask가 True인 구간이 min_run 연속되는 첫 시작 index 반환"""
    if min_run <= 1:
        idx = int(np.argmax(mask))
        return idx if mask[idx] else None
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= min_run:
            return int(i - run + 1)
    return None

def detect_gat_got_with_adaptive(
    env: np.ndarray,
    fs: float,
    k: float = 1.0,
    min_run_ms: float = 12.0,
    win_cycles: int = 3,   # 자리만 유지 (호출 시그니처 호환)
    cv_max: float = 0.25,  # 자리만 유지
    mode: str = "full",
):
    """
    1) 기본 임계값 추정(base, thr_up/thr_dn)
    2) Adaptive 최적화로 thr_local/QC 추출 (full)  |  Lite θ_on/θ_off 경로
    3) 히스테리시스로 GAT/GOT/VOnT/VOffT 결정
    4) app.py가 기대하는 dict 반환
    """
    env = np.asarray(env, dtype=float)
    N = env.size
    if N == 0 or fs <= 0:
        return {
            "gat_ms": np.nan, "got_ms": np.nan,
            "vont_ms": np.nan, "vofft_ms": np.nan,
            "preset": "Adaptive v3.3",
            "adaptive_qc": {
                "qc_label": "Unknown", "noise_ratio": np.nan,
                "est_rmse": np.nan, "global_gain": np.nan, "iters": 0,
            },
        }

    # 1) baseline + 기본 히스테리시스
    base, thr_up, thr_dn = _estimate_baseline_threshold(env, k=k)

    # 2) Adaptive 최적화 (QC 포함)
    res = adaptive_optimize(env, sr_hz=float(fs), base_threshold=thr_up, params=None, reference_marks=None)


    thr_hi = res.thr_local
    thr_lo_ratio = getattr(res, "thr_lo_ratio", thr_lo_ratio_default)  # ✅ 변경: 기본값(0.85) or AdaptiveParams에서 주입
    thr_lo = res.thr_local * float(thr_lo_ratio)                       # ✅ 비율 적용


    # 3) 히스테리시스 기반 On/Off 마스크 생성
    min_run = max(1, int(round((min_run_ms / 1000.0) * fs)))

    above = env >= thr_hi
    gat_idx = _first_persistent_index(above, min_run)

    # 전체 True 구간(발성) 마스크를 만들기 위해 한 번 올라간 후엔 thr_lo로 유지
    state = False
    voiced = np.zeros(N, dtype=bool)
    for i in range(N):
        if not state:
            # 시작: thr_hi 이상이 min_run 지속되면 True로 전환
            if i == gat_idx:
                state = True
        else:
            # 유지: thr_lo 이상이면 계속 True
            if env[i] < thr_lo[i]:
                state = False
        voiced[i] = state

    # VOnT/VOffT 계산: 첫 True 시작 이후 첫 안정 peak/혹은 단순 상한 구간의 중앙값 근사
    # (여기서는 간단히 경계점으로 정의)
    if np.any(voiced):
        starts = np.flatnonzero((voiced.astype(int)[1:] - voiced.astype(int)[:-1]) == 1) + 1
        ends   = np.flatnonzero((voiced.astype(int)[1:] - voiced.astype(int)[:-1]) == -1)
        if voiced[0]:
            starts = np.r_[0, starts]
        if voiced[-1]:
            ends = np.r_[ends, N - 1]

        # 첫 발성 구간과 마지막 발성 구간을 사용
        if len(starts) > 0:
            i_on  = int(starts[0])
            i_off = int(ends[-1])
        else:
            i_on = i_off = None
    else:
        i_on = i_off = None

    # 지표 산출
    gat_ms   = (1000.0 * gat_idx / fs) if gat_idx is not None else np.nan
    # 간단화: VOnT/VOffT는 경계점으로 정의 (필요시 정교화 가능)
    vont_ms  = (1000.0 * i_on  / fs) if i_on  is not None else np.nan
    vofft_ms = (1000.0 * i_off / fs) if i_off is not None else np.nan

    # GOT: 발성 종료 직전, thr_lo를 아래로 떨어져 안정적으로 유지되기 시작한 지점 근사
    if i_off is not None:
        below = env < thr_lo
        got_idx = _first_persistent_index(below[i_off:], min_run)
        if got_idx is not None:
            got_idx = int(i_off + got_idx)
        else:
            # 종료점 근사
            got_idx = i_off
    else:
        got_idx = None
    got_ms = (1000.0 * got_idx / fs) if got_idx is not None else np.nan

    # 4) 반환 패킷
    adaptive_qc = {
        "qc_label":   res.qc_label,        # "🟢 High" / "🟡 Medium" / "🔴 Low"
        "noise_ratio": float(res.noise_ratio),
        "est_rmse":    float(res.est_rmse) if res.est_rmse is not None else np.nan,
        "global_gain": float(res.global_gain),
        "iters":       int(res.iters),
    }

    return {
        "gat_ms": float(gat_ms),
        "got_ms": float(got_ms),
        "vont_ms": float(vont_ms),
        "vofft_ms": float(vofft_ms),
        "preset": "Adaptive v3.3",
        "adaptive_qc": adaptive_qc,
    }

