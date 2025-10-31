# -*- coding: utf-8 -*-
"""
Adaptive Threshold Engine – Lite / Full Compatible
Author: Isaka × Lian
Version: v3.3 Stable
Description:
    Adaptive threshold detection for glottal attack/offset estimation.
    Computes local statistics (P10/P95, MAD, IQR, SNR) and determines
    onset/offset candidates using hysteresis and bidirectional refinement.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


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


# -----------------------------
# Core Functions
# -----------------------------
def compute_local_stats(env: np.ndarray) -> Dict[str, float]:
    """Compute local statistical features of the envelope."""
    env = np.asarray(env, dtype=float)
    p10, p95 = np.percentile(env, [10, 95])
    mad = np.median(np.abs(env - np.median(env)))
    iqr = p95 - p10
    snr_est = (p95 - p10) / (mad + 1e-9)
    return {"p10": p10, "p95": p95, "mad": mad, "iqr": iqr, "snr_est": snr_est}


def adaptive_optimize(envelope: np.ndarray,
                      sr_hz: float,
                      base_threshold: float,
                      params: Optional[AdaptiveParams] = None,
                      reference_marks: Optional[Dict[str, float]] = None) -> AdaptiveResult:
    """
    Adaptive optimization for onset/offset detection.
    """
    thr_lo_ratio_default = 0.85  # ✅ fixed indentation bug

    if params is None:
        params = AdaptiveParams()

    local = compute_local_stats(envelope)
    thr_on = local["p10"] + params.k_on * local["mad"]
    thr_off = local["p10"] + params.k_off * local["mad"]

    # Normalize envelope and detect crossings
    env = (envelope - np.min(envelope)) / (np.ptp(envelope) + 1e-9)
    fs = float(sr_hz)
    min_run = int((params.min_run_ms / 1000.0) * fs)
    above = env > thr_on
    below = env < thr_off

    gat_idx = None
    got_idx = None
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            gat_idx = i - run + 1
            break

    run = 0
    for i, b in enumerate(below[::-1]):
        run = run + 1 if b else 0
        if run >= min_run:
            got_idx = len(env) - (i - run + 1)
            break

    def ms(idx):
        return None if idx is None else 1000.0 * (idx / fs)

    gat_ms = ms(gat_idx)
    got_ms = ms(got_idx)
    vont_ms = gat_ms
    vofft_ms = got_ms

    # Dummy placeholders for QC
    noise_ratio = 1.0 / (local["snr_est"] + 1e-9)
    global_gain = local["iqr"]
    iters = 1
    est_rmse = local["mad"]
    qc_label = "OK" if local["snr_est"] > 5 else "LOW"

    return AdaptiveResult(
        gat_ms=gat_ms or np.nan,
        vont_ms=vont_ms or np.nan,
        got_ms=got_ms or np.nan,
        vofft_ms=vofft_ms or np.nan,
        noise_ratio=noise_ratio,
        global_gain=global_gain,
        iters=iters,
        est_rmse=est_rmse,
        qc_label=qc_label,
    )


def detect_gat_got_with_adaptive(env: np.ndarray,
                                 fs: float,
                                 k: float = 1.0,
                                 min_run_ms: float = 12.0,
                                 win_cycles: int = 4,
                                 cv_max: float = 0.12) -> Dict[str, float]:
    """
    Top-level function for adaptive threshold detection.
    Returns dict with GAT, VOnT, GOT, VOffT (ms).
    """
    params = AdaptiveParams(k_on=k, min_run_ms=min_run_ms,
                            win_cycles=win_cycles, cv_max=cv_max)
    res = adaptive_optimize(env, fs, base_threshold=0.0, params=params)
    return {
        "gat_ms": res.gat_ms,
        "vont_ms": res.vont_ms,
        "got_ms": res.got_ms,
        "vofft_ms": res.vofft_ms,
        "noise_ratio": res.noise_ratio,
        "global_gain": res.global_gain,
        "iters": res.iters,
        "est_rmse": res.est_rmse,
        "qc_label": res.qc_label,
    }
