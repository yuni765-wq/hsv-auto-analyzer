# -*- coding: utf-8 -*-
"""
Adaptive Threshold Engine – Minimal v0.2 (Step 2: 최소 동작)
Branch: feat/ate-qc-v33
File: adaptive_threshold.py

Purpose
-------
Provide a working, numpy‑only minimal implementation of the adaptive path so that
`metrics.detect_gat_vont_got_vofft()` can toggle it on for smoke tests. This is
still conservative and computation‑safe (no SciPy).

What’s implemented now
----------------------
1) `AdaptiveParams` dataclass (unchanged surface).
2) `compute_local_stats`: rolling P10/P95, MAD, IQR, and a rough SNR_est.
3) `adaptive_thresholds`: θ_on/θ_off traces from local spread (+ optional smoothing)
   and per‑sample hysteresis (samples).
4) `detect_edges_basic`: simple rising/falling crossing detector with hysteresis
   guard — returns coarse on/off in milliseconds.
5) `ensemble_vote`: still minimal but supports weights.
6) `bidirectional_refine`: no‑op (placeholder for Step 2.5).

Assumptions
-----------
- Inputs are 1‑D numpy arrays of equal length sampled at `fs` (Hz).
- `total` is the primary envelope cue; `left/right` optional for future use.

Author: Isaka × Lian (2025-10-31, KST)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-9

# ----------------------------------------------------------------------------
# 0) Hyper-parameter container (public, stable surface for the app UI)
# ----------------------------------------------------------------------------
@dataclass
class AdaptiveParams:
    """Configuration for adaptive onset/offset estimation.

    Attributes
    ----------
    win_ms : float
        Sliding window length for local-stat estimation.
    step_ms : Optional[float]
        Step between windows. If None, defaults to win_ms / 4.
    p_low : float
        Lower percentile (0–100) for baseline estimation (e.g., 10th).
    p_high : float
        Upper percentile for envelope ceiling (e.g., 95th).
    on_gain : float
        Gain factor on local spread for θ_on.
    off_gain : float
        Gain factor on local spread for θ_off.
    hyst_ms : float
        Hysteresis time to prevent chatter between on/off.
    smooth_ms : float
        Optional moving-avg smoothing for θ traces.
    min_agree : int
        Minimum number of voters in `ensemble_vote`.
    """

    win_ms: float = 30.0
    step_ms: Optional[float] = None
    p_low: float = 10.0
    p_high: float = 95.0
    on_gain: float = 0.55  # conservative (spread fraction)
    off_gain: float = 0.35 # ensure θ_off < θ_on for hysteresis
    hyst_ms: float = 12.0
    smooth_ms: float = 8.0
    min_agree: int = 2


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------

def _rolling_view(x: NDArray[np.floating], win: int, step: int) -> NDArray[np.floating]:
    """Create a strided rolling window view with hop=step (no copies).
    Returns shape (n_win, win). Requires len(x) >= win.
    """
    from numpy.lib.stride_tricks import as_strided
    n = (len(x) - win) // step + 1
    if n <= 0:
        raise ValueError("Window longer than signal")
    return as_strided(
        x,
        shape=(n, win),
        strides=(x.strides[0] * step, x.strides[0]),
    )


def _win_step_samples(fs: float, win_ms: float, step_ms: Optional[float]) -> Tuple[int, int]:
    win = max(1, int(round(fs * win_ms / 1000.0)))
    step = int(round(fs * (step_ms if step_ms is not None else (win_ms / 4.0)) / 1000.0))
    step = max(1, min(step, win))
    return win, step


def _expand_to_samples(values: NDArray[np.floating], step: int, n_total: int) -> NDArray[np.floating]:
    """Repeat each window value for `step` samples and pad/truncate to `n_total`."""
    out = np.repeat(values, step)
    if len(out) < n_total:
        pad = np.full(n_total - len(out), values[-1], dtype=float)
        out = np.concatenate([out, pad])
    elif len(out) > n_total:
        out = out[:n_total]
    return out


def _moving_average(x: NDArray[np.floating], k: int) -> NDArray[np.floating]:
    if k <= 1:
        return x
    k = int(k)
    k = max(1, k)
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


# ----------------------------------------------------------------------------
# 1) Local statistics (P10/P95, MAD, IQR, rough SNR) – MINIMAL
# ----------------------------------------------------------------------------

def compute_local_stats(
    total: NDArray[np.floating],
    left: Optional[NDArray[np.floating]],
    right: Optional[NDArray[np.floating]],
    fs: float,
    params: AdaptiveParams,
) -> Dict[str, NDArray[np.floating]]:
    """Compute windowed local statistics for adaptive thresholds.

    Returns arrays aligned to `total` length via hop-repeat expansion.
    - p10, p95: local percentiles
    - mad: median(|x - median(x)|)
    - iqr: P75 - P25
    - snr_est: (p95 - p10) / (mad + eps)
    """
    x = np.asarray(total, dtype=float)
    n_total = len(x)
    if n_total == 0:
        return {k: np.array([], dtype=float) for k in ("p10","p95","mad","iqr","snr_est")}

    win, step = _win_step_samples(fs, params.win_ms, params.step_ms)
    try:
        mat = _rolling_view(x, win=win, step=step)
    except ValueError:
        # Fallback: single-window over entire signal
        mat = x[None, :]
        step = n_total  # expand will pad appropriately

    # Percentiles and robust spreads
    p10_w = np.nanpercentile(mat, params.p_low, axis=1)
    p95_w = np.nanpercentile(mat, params.p_high, axis=1)
    med_w = np.nanmedian(mat, axis=1)
    mad_w = np.nanmedian(np.abs(mat - med_w[:, None]), axis=1)
    p25_w = np.nanpercentile(mat, 25, axis=1)
    p75_w = np.nanpercentile(mat, 75, axis=1)
    iqr_w = p75_w - p25_w
    spread_w = np.maximum(_EPS, p95_w - p10_w)
    snr_w = spread_w / np.maximum(_EPS, mad_w)

    # Expand window stats to per-sample traces
    p10 = _expand_to_samples(p10_w, step, n_total)
    p95 = _expand_to_samples(p95_w, step, n_total)
    mad = _expand_to_samples(mad_w, step, n_total)
    iqr = _expand_to_samples(iqr_w, step, n_total)
    snr = _expand_to_samples(snr_w, step, n_total)

    return {"p10": p10, "p95": p95, "mad": mad, "iqr": iqr, "snr_est": snr}


# ----------------------------------------------------------------------------
# 2) Map local stats → θ_on/θ_off with hysteresis – MINIMAL
# ----------------------------------------------------------------------------

def adaptive_thresholds(
    stats: Dict[str, NDArray[np.floating]],
    fs: float,
    params: AdaptiveParams,
) -> Dict[str, NDArray[np.floating]]:
    """Generate θ_on/θ_off from local spread with optional smoothing.

    θ_on = p10 + on_gain * (p95 - p10)
    θ_off = p10 + off_gain * (p95 - p10)  (ensure θ_off < θ_on)
    """
    p10 = np.asarray(stats.get("p10"), dtype=float)
    p95 = np.asarray(stats.get("p95"), dtype=float)
    n = len(p10)
    spread = np.maximum(_EPS, p95 - p10)

    theta_on = p10 + params.on_gain * spread
    theta_off = p10 + params.off_gain * spread
    # Guarantee off <= on
    theta_off = np.minimum(theta_off, theta_on - 1e-6)

    # Optional smoothing of thresholds
    smooth_k = max(1, int(round(fs * params.smooth_ms / 1000.0)))
    if smooth_k > 1:
        theta_on = _moving_average(theta_on, smooth_k)
        theta_off = _moving_average(theta_off, smooth_k)

    hyst_samps = np.full(n, max(1, int(round(fs * params.hyst_ms / 1000.0)))), dtype=float)

    return {"theta_on": theta_on, "theta_off": theta_off, "hyst_samps": hyst_samps}


# ----------------------------------------------------------------------------
# 3) Ensemble voting across cues – MINIMAL (supports weights)
# ----------------------------------------------------------------------------

def ensemble_vote(
    votes: Iterable[NDArray[np.bool_]],
    params: AdaptiveParams,
    weights: Optional[Iterable[float]] = None,
) -> NDArray[np.bool_]:
    """Combine multiple boolean candidate edges via weighted majority.

    If `weights` is None → equal weights. Returns boolean mask length = N.
    """
    arrs = [np.asarray(v, dtype=bool) for v in votes]
    if not arrs:
        return np.zeros(0, dtype=bool)
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("All vote arrays must share the same length")

    if weights is None:
        weights = [1.0] * len(arrs)
    w = np.asarray(list(weights), dtype=float)
    w = np.maximum(0.0, w)
    if w.sum() <= 0:
        w = np.ones_like(w)

    stacked = np.vstack(arrs).astype(float)
    score = (w[:, None] * stacked).sum(axis=0)
    thresh = max(1.0, float(params.min_agree)) - 1e-9
    return score >= thresh


# ----------------------------------------------------------------------------
# 4) Coarse on/off detector with hysteresis guard – MINIMAL
# ----------------------------------------------------------------------------

def detect_edges_basic(
    total_env: NDArray[np.floating],
    theta_on: NDArray[np.floating],
    theta_off: NDArray[np.floating],
    hyst_samps: NDArray[np.floating],
    fs: float,
) -> Dict[str, Optional[float]]:
    """Return coarse onset/offset (ms) via simple crossing with hysteresis.

    Strategy
    --------
    - Find first index i where env crosses above θ_on and stays above for
      at least `hyst` samples.
    - From the tail, find first index j where env falls below θ_off and stays
      below for `hyst` samples.
    - If not found, return None for that edge.
    """
    x = np.asarray(total_env, dtype=float)
    th_on = np.asarray(theta_on, dtype=float)
    th_off = np.asarray(theta_off, dtype=float)
    h = int(np.nanmedian(hyst_samps).item()) if len(hyst_samps) else 1
    h = max(1, h)

    n = len(x)
    on_idx = None
    for i in range(n - h):
        if x[i] <= th_on[i] and np.all(x[i+1:i+1+h] > th_on[i+1:i+1+h]):
            on_idx = i + 1
            break

    off_idx = None
    for j in range(n - 1, h, -1):
        if x[j] >= th_off[j] and np.all(x[j-h:j] < th_off[j-h:j]):
            off_idx = j - h + 1
            break

    on_ms = (on_idx / fs * 1000.0) if on_idx is not None else None
    off_ms = (off_idx / fs * 1000.0) if off_idx is not None else None
    return {"on": on_ms, "off": off_ms}


# ----------------------------------------------------------------------------
# 5) Small bidirectional refinement of detected edges – PLACEHOLDER
# ----------------------------------------------------------------------------

def bidirectional_refine(
    edges: Dict[str, Optional[float]],
    signal: NDArray[np.floating],
    fs: float,
    params: AdaptiveParams,
) -> Dict[str, Optional[float]]:
    """Refine coarse onset/offset with short backward/forward scans (TODO).

    Currently a no‑op to keep Step 2 minimal and safe.
    """
    return {"on": edges.get("on"), "off": edges.get("off")}


# ----------------------------------------------------------------------------
# 6) Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 1000.0
    t = np.arange(0, 1.2, 1.0 / fs)
    env = 0.1 + 0.2 * np.sin(2*np.pi*2*t)
    env[int(0.25*fs):int(0.80*fs)] += 0.7  # burst region

    p = AdaptiveParams()
    S = compute_local_stats(env, None, None, fs, p)
    TH = adaptive_thresholds(S, fs, p)
    edges = detect_edges_basic(env, TH["theta_on"], TH["theta_off"], TH["hyst_samps"], fs)
    print("ON/OFF (ms):", edges)
