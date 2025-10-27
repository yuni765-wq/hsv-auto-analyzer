
"""
insight_v32.py
HSV Auto Analyzer v3.2 — Clinical Insight Release (addon)

Drop this file next to your app.py and do the 3 small edits in v32_patch_instructions.md.
It provides:
  - compute_quality_from_env(): QI score + label + indicators
  - clinical_note(): auto clinical interpretation line
  - render_quality_banner(): pinned quality badge across tabs
  - inject_css(): sticky banner styling
"""
from typing import Dict, Tuple, List
import numpy as np

VERSION_V32 = "HSV Auto Analyzer v3.2 — Clinical Insight Release"

def to_scalar(x):
    try:
        if x is None:
            return np.nan
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            try:
                x = x.item()
            except Exception:
                x = x[0]
        return float(x)
    except Exception:
        return np.nan

def compute_quality_from_env(env: Dict, ap_thr: float = 0.70, tp_thr: float = 0.85, ps_thr: float = 0.08) -> Dict:
    """Return dict with fields: score, label, notes(list), AP, TP, PS_dist, thresholds"""
    AP      = to_scalar(env.get("AP"))
    TP      = to_scalar(env.get("TP"))
    PS_dist = to_scalar(env.get("PS_dist"))
    score = 0
    notes: List[str] = []
    if np.isfinite(AP):
        if AP >= ap_thr:
            score += 1
        else:
            notes.append(f"AP<{ap_thr}")
    else:
        notes.append("AP=NaN")
    if np.isfinite(TP):
        if TP >= tp_thr:
            score += 1
        else:
            notes.append(f"TP<{tp_thr}")
    else:
        notes.append("TP=NaN")
    if np.isfinite(PS_dist):
        if PS_dist <= ps_thr:
            score += 1
        else:
            notes.append(f"PS_dist>{ps_thr}")
    else:
        notes.append("PS_dist=NaN")
    if score == 3:
        label = "High"
    elif score == 2:
        label = "Medium"
    else:
        label = "Low"
    return dict(score=score, label=label, notes=notes, AP=AP, TP=TP, PS_dist=PS_dist, thresholds=(ap_thr, tp_thr, ps_thr))

def clinical_note(qi: Dict) -> str:
    """Generate a short clinical note from QI components."""
    ap, tp, ps = qi["AP"], qi["TP"], qi["PS_dist"]
    ap_thr, tp_thr, ps_thr = qi["thresholds"]
    ap_ok = np.isfinite(ap) and ap >= ap_thr
    tp_ok = np.isfinite(tp) and tp >= tp_thr
    ps_ok = np.isfinite(ps) and ps <= ps_thr

    if ap_ok and tp_ok and ps_ok:
        return "정상 패턴으로 판단됩니다."
    # ULP heuristic: periodicity ok but phase mismatch
    if ap_ok and tp_ok and not ps_ok:
        return "좌우 위상 불일치가 커서 편측 성대마비(ULP) 패턴이 의심됩니다."
    # SD heuristic: both periodicities degraded
    if not ap_ok and not tp_ok and ps_ok:
        return "진폭·주기 안정성이 저하되어 경련성 발성장애(SD) 의심 소견입니다."
    # MTD-ish: timing down with relatively ok AP
    if ap_ok and not tp_ok:
        return "주기성 저하가 관찰되어 근긴장성 발성장애(MTD) 가능성이 있습니다."
    # fallback
    return "품질 지표가 혼재되어 추가 확인이 필요합니다."

def inject_css(st):
    st.markdown(
        """
        <style>
        .qi-sticky {
            position: sticky; top: 0; z-index: 999;
            padding: .6rem .9rem; margin-bottom: .5rem;
            border-radius: 12px;
            backdrop-filter: blur(4px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_quality_banner(st, qi: Dict, show_debug: bool=False, pinned: bool=True):
    """Render the badge + optional debug and clinical note. Call at top of each tab or once after tabs."""
    if not qi:
        return
    color = {"High":"#16a34a","Medium":"#f59e0b","Low":"#dc2626"}[qi["label"]]
    cls = "qi-sticky" if pinned else ""
    st.markdown(
        f"<div class='{cls}' style='background:{color}20;'>"
        f"<div style='display:inline-block;padding:.35rem .6rem;border-radius:999px;background:{color};color:white;font-weight:700'>"
        f"Quality: {qi['label']}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption("Clinical note: " + clinical_note(qi))
    if show_debug:
        st.caption(f"[QI debug] score={qi['score']} | AP={qi['AP']:.4f}, TP={qi['TP']:.4f}, PS_dist={qi['PS_dist']:.4f}")
        if qi["notes"]:
            st.caption("Indicators: " + " · ".join(qi["notes"]))
