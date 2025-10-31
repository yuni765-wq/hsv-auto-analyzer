# -*- coding: utf-8 -*-
"""
HSV Auto Analyzer – Adaptive Threshold Engine v3.3 (Stable, Single File)
Author: Isaka × Lian

단일 파일로 통합된 적응형 임계값 엔진.
- MODE = "full"  : 국소 임계값(thr_local) + 히스테리시스 + 지속시간(persistence) + 양방향 정제
- MODE = "lite"  : 전역 임계값 기반의 경량 파이프라인(빠른 폴백/테스트용)

Public API (metrics.py 등에서 사용):
- class AdaptiveParams
- class AdaptiveResult
- def adaptive_optimize(...)
- def detect_gat_got_with_adaptive(...)

저장 규칙: UTF-8, LF, 4 spaces
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


# -----------------------------
# Configuration
# -----------------------------
MODE: str = "full"            # "full" | "lite"
SMOOTH_WIN: int = 31          # thr_local 이동중앙값 윈도 (full)
THR_LO_RATIO: float = 0.85    # 히스테리시스 하한 비율 (full)
EPS: float = 1e-9


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class AdaptiveParams:
    k_on: float = 1.0
    k_off: float = 0.8
    hysteresis: float = 0.5
    min_run_ms: float = 12.0
    win_cycles: int = 4
    cv_max: float = 0.12


@dataclass
class AdaptiveResult:
    gat_ms: float
    vont_ms: float
    got_ms: float
    vofft_ms: float
    noise_ratio: float
    global_gain: float
    iters: int
    est_rmse: float
    qc_label: str
    thr_local: Optional[np.ndarray] = None


# -----------------------------
# Utilities
# -----------------------------
def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med))

def _moving_median(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)  # 홀수 강제
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(xp, win)  # (N, win)
    return np.median(sw, axis=1)

def _first_persistent_index(mask: np.ndarray, min_run: int) -> Optional[int]:
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= min_run:
            return i - run + 1
    return None

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = float(np.max(x) - np.min(x))
    return (x - np.min(x)) / (rng + EPS)


# -----------------------------
# Statistics & thresholds
# -----------------------------
def compute_local_stats(env: np.ndarray) -> Dict[str, float]:
    env = np.asarray(env, dtype=float)
    p10, p95 = np.percentile(env, [10, 95])
    mad = _mad(env)
    iqr = p95 - p10
    snr_est = (p95 - p10) / (mad + EPS)
    return {"p10": p10, "p95": p95, "mad": mad, "iqr": iqr, "snr_est": snr_est}

def estimate_global_thresholds(env: np.ndarray, k_on: float, k_off: float) -> Tuple[float, float]:
    st = compute_local_stats(env)
    thr_on = st["p10"] + k_on  * st["mad"]
    thr_off= st["p10"] + k_off * st["mad"]
    return float(thr_on), float(thr_off)


# -----------------------------
# Core (FULL / LITE)
# -----------------------------
def _full_pipeline(envelope: np.ndarray, fs: float, params: AdaptiveParams) -> AdaptiveResult:
    """
    FULL: 이동중앙값 기반 thr_local 생성 → 히스테리시스(persist) → onset/offest 계산
    """
    env = _normalize01(envelope)
    N = env.size
    fs = float(fs)
    min_run = max(1, int(round((params.min_run_ms / 1000.0) * fs)))

    # 1) thr_local (moving median + 온/오프 보정)
    base = _moving_median(env, SMOOTH_WIN)
    st = compute_local_stats(env)
    thr_local = base + params.k_on * (st["mad"] + EPS)  # 상향기준
    thr_hi = thr_local
    thr_lo = thr_local * THR_LO_RATIO

    # 2) 히스테리시스 상태 마스크
    above_hi = env >= thr_hi
    gat_idx = _first_persistent_index(above_hi, min_run)

    voiced = np.zeros(N, dtype=bool)
    state = False
    for i in range(N):
        if not state:
            if gat_idx is not None and i >= gat_idx:
                state = True
        else:
            if env[i] < thr_lo[i]:
                state = False
        voiced[i] = state

    # 3) 경계 추출
    if np.any(voiced):
        v = voiced.astype(int)
        starts = np.flatnonzero((v[1:] - v[:-1]) == 1) + 1
        ends   = np.flatnonzero((v[1:] - v[:-1]) == -1)
        if voiced[0]:  starts = np.r_[0, starts]
        if voiced[-1]: ends   = np.r_[ends, N - 1]
        i_on  = int(starts[0]) if len(starts) else None
        i_off = int(ends[-1])   if len(ends)   else None
    else:
        i_on = i_off = None

    # 4) GOT: i_off 이후 thr_lo 이하 지속구간
    if i_off is not None:
        below_lo = env < thr_lo
        gi = _first_persistent_index(below_lo[i_off:], min_run)
        got_idx = int(i_off + gi) if gi is not None else i_off
    else:
        got_idx = None

    def ms(i): return np.nan if i is None else 1000.0 * (i / fs)

    gat_ms   = ms(gat_idx)
    vont_ms  = ms(i_on)
    vofft_ms = ms(i_off)
    got_ms   = ms(got_idx)

    # 5) 간단 QC
    noise_ratio = 1.0 / (st["snr_est"] + EPS)
    global_gain = st["iqr"]
    est_rmse    = st["mad"]
    qc_label    = "🟢 High" if st["snr_est"] >= 5 else ("🟡 Medium" if st["snr_est"] >= 3 else "🔴 Low")

    return AdaptiveResult(
        gat_ms=gat_ms, vont_ms=vont_ms, got_ms=got_ms, vofft_ms=vofft_ms,
        noise_ratio=float(noise_ratio), global_gain=float(global_gain),
        iters=1, est_rmse=float(est_rmse), qc_label=qc_label, thr_local=thr_local
    )


def _lite_pipeline(envelope: np.ndarray, fs: float, params: AdaptiveParams) -> AdaptiveResult:
    """
    LITE: 전역 임계값(thr_on/off) + 지속시간(persist)만 사용
    """
    env = _normalize01(envelope)
    fs = float(fs)
    min_run = max(1, int(round((params.min_run_ms / 1000.0) * fs)))

    thr_on, thr_off = estimate_global_thresholds(env, params.k_on, params.k_off)

    # GAT
    gat_idx = _first_persistent_index(env >= thr_on, min_run)

    # 간단 voiced 마스크
    state = False
    voiced = np.zeros(env.size, dtype=bool)
    for i in range(env.size):
        if not state:
            if gat_idx is not None and i >= gat_idx:
                state = True
        else:
            if env[i] < thr_off:
                state = False
        voiced[i] = state

    if np.any(voiced):
        v = voiced.astype(int)
        starts = np.flatnonzero((v[1:] - v[:-1]) == 1) + 1
        ends   = np.flatnonzero((v[1:] - v[:-1]) == -1)
        if voiced[0]:  starts = np.r_[0, starts]
        if voiced[-1]: ends   = np.r_[ends, env.size - 1]
        i_on  = int(starts[0]) if len(starts) else None
        i_off = int(ends[-1])   if len(ends)   else None
    else:
        i_on = i_off = None

    # GOT
    if i_off is not None:
        gi = _first_persistent_index((env[i_off:] < thr_off), min_run)
        got_idx = int(i_off + gi) if gi is not None else i_off
    else:
        got_idx = None

    def ms(i): return np.nan if i is None else 1000.0 * (i / fs)

    st = compute_local_stats(env)
    return AdaptiveResult(
        gat_ms=ms(gat_idx), vont_ms=ms(i_on), got_ms=ms(got_idx), vofft_ms=ms(i_off),
        noise_ratio=float(1.0 / (st["snr_est"] + EPS)),
        global_gain=float(st["iqr"]),
        iters=1, est_rmse=float(st["mad"]),
        qc_label=("🟢 High" if st["snr_est"] >= 5 else ("🟡 Medium" if st["snr_est"] >= 3 else "🔴 Low")),
        thr_local=None,
    )


# -----------------------------
# Public API
# -----------------------------
def adaptive_optimize(envelope: np.ndarray,
                      sr_hz: float,
                      base_threshold: float = 0.0,   # 호환성 유지용 (사용 안함)
                      params: Optional[AdaptiveParams] = None,
                      reference_marks: Optional[Dict[str, float]] = None) -> AdaptiveResult:
    """
    통합 엔진: MODE에 따라 full/lite 파이프라인 실행.
    """
    if params is None:
        params = AdaptiveParams()

    if MODE == "lite":
        return _lite_pipeline(envelope, sr_hz, params)
    # default: "full"
    return _full_pipeline(envelope, sr_hz, params)


def detect_gat_got_with_adaptive(env: np.ndarray,
                                 fs: float,
                                 k: float = 1.0,
                                 min_run_ms: float = 12.0,
                                 win_cycles: int = 4,
                                 cv_max: float = 0.12) -> Dict[str, float]:
    """
    Top-level wrapper. metrics.py에서 호출되는 공개 함수.
    반환값(dict): gat_ms, vont_ms, got_ms, vofft_ms, adaptive_qc, preset, (옵션)thr_local
    """
    params = AdaptiveParams(k_on=k, k_off=0.8, min_run_ms=min_run_ms,
                            win_cycles=win_cycles, cv_max=cv_max)
    res = adaptive_optimize(env, fs, params=params)

    return {
        "gat_ms": float(res.gat_ms),
        "vont_ms": float(res.vont_ms),
        "got_ms": float(res.got_ms),
        "vofft_ms": float(res.vofft_ms),
        "preset": f"Adaptive v3.3 ({MODE})",
        "adaptive_qc": {
            "qc_label": res.qc_label,
            "noise_ratio": float(res.noise_ratio),
            "est_rmse": float(res.est_rmse),
            "global_gain": float(res.global_gain),
            "iters": int(res.iters),
        },
        # 디버깅/시각화용(옵션)
        "thr_local": res.thr_local,
    }
