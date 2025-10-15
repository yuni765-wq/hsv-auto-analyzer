# app_v2.4_clinical_final.py
# HSV Auto Analyzer – v2.4 Clinical FINAL
# - Clinical preset(default): baseline_s=0.06, k=0.90, M=40, W_ms=35, amp_frac=0.70, ap/tp=0.97
# - UI lower bounds extended (baseline_s ≥ 0.02s, k ≥ 0.50)
# - Hysteresis + Debounce + Steady gating + Robust fallbacks
# - Onset search starts AFTER baseline + 15ms (anti-early-trigger)
# - Built-in RMSE/MAE/Bias evaluator with CSV export

import numpy as np
import pandas as pd
from io import StringIO
import streamlit as st

try:
    from scipy.signal import savgol_filter
    _HAS_SAVITZKY = True
except Exception:
    _HAS_SAVITZKY = False

# --------------------------- Page ---------------------------------
st.set_page_config(page_title="HSV Auto Analyzer v2.4 – Clinical Final", layout="wide")
st.title("HSV Auto Analyzer v2.4 – Clinical FINAL")
st.caption("AP/TP/AS/PS + VOnT/VOffT (히스테리시스·디바운스·steady 게이팅 + 디버그 + RMSE)")

# --------------------------- Utils --------------------------------
def _norm_cols(cols): return [c.lower().strip().replace(" ", "_") for c in cols]

def _moving_rms(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return np.sqrt(np.maximum(x * x, 0.0))
    w = int(w); pad = w // 2
    xx = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(w) / float(w)
    m2 = np.convolve(xx * xx, ker, mode="valid")
    return np.sqrt(np.maximum(m2, 0.0))

def _smooth(signal: np.ndarray, fps: float) -> np.ndarray:
    n = len(signal)
    if n < 7: return signal.astype(float)
    base_w = int(max(7, min(21, round(fps * 0.007))))
    win = base_w if (base_w % 2 == 1) else base_w + 1
    win = min(win, n - 1) if n % 2 == 0 and win >= n else min(win, n - (1 - (n % 2)))
    if _HAS_SAVITZKY:
        try: return savgol_filter(signal.astype(float), window_length=win, polyorder=3, mode="interp")
        except Exception: pass
    pad = win // 2
    xx = np.pad(signal.astype(float), (pad, pad), mode="edge")
    ker = np.ones(win) / float(win)
    return np.convolve(xx, ker, mode="valid")

def _detect_peaks(y: np.ndarray) -> np.ndarray:
    if len(y) < 3: return np.array([], dtype=int)
    s = np.sign(y[1:] - y[:-1])
    return (np.where((s[:-1] > 0) & (s[1:] <= 0))[0] + 1).astype(int)

def _build_cycles(t: np.ndarray, sig: np.ndarray, min_frames: int=3) -> list:
    peaks = _detect_peaks(sig); cycles = []
    if len(peaks) < 2: return cycles
    for i in range(len(peaks)-1):
        s, e = int(peaks[i]), int(peaks[i+1])
        if (e - s) >= max(2, min_frames): cycles.append((s, e))
    return cycles

def _nanmean0(x):
    v = np.nanmean(x) if len(x) else np.nan
    return 0.0 if (v is None or np.isnan(v)) else float(v)

def _clamp01(x):
    if x is None or np.isnan(x): return np.nan
    return float(max(0.0, min(1.0, x)))

def _ap_tp(t, total, cycles):
    if len(cycles) < 3: return (np.nan, np.nan)
    amps, per = [], []
    for s,e in cycles:
        if e<=s: continue
        seg = total[s:e]
        amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
        per.append(float(t[e]-t[s]))
    amps, per = np.array(amps,float), np.array(per,float)
    def _p(v):
        m = np.nanmean(v); s = np.nanstd(v, ddof=1) if len(v)>1 else 0.0
        if not np.isfinite(m) or m<=0: return np.nan
        return _clamp01(1.0 - (s/m))
    return (_p(amps), _p(per))

def _as_range(left, right, cycles):
    if left is None or right is None or len(cycles)<1: return np.nan
    ratios=[]
    for s,e in cycles:
        if e<=s: continue
        L = float(np.nanmax(left[s:e]) - np.nanmin(left[s:e]))
        R = float(np.nanmax(right[s:e]) - np.nanmin(right[s:e]))
        m = max(L,R); ratios.append((min(L,R)/m) if m>0 else np.nan)
    return _clamp01(_nanmean0(ratios))

def _ps(left,right,t,cycles):
    if left is None or right is None or len(cycles)<1: return np.nan
    diffs=[]
    for s,e in cycles:
        if e<=s: continue
        li=s+int(np.nanargmax(left[s:e])); ri=s+int(np.nanargmax(right[s:e]))
        Ti=float(t[e]-t[s]) if t is not None else 1.0
        if Ti<=0: continue
        diffs.append(min(1.0, abs(float(t[li]-t[ri]))/Ti))
    if not len(diffs): return np.nan
    return _clamp01(1.0 - _nanmean0(diffs))

# ------------------------ Analyzer ------------------------------------
def analyze(df: pd.DataFrame, adv: dict, show_debug_charts: bool=False, force_total_for_onset: bool=False):
    cols=_norm_cols(df.columns.tolist()); df.columns=cols
    def pick(name):
        for c in cols:
            if name in c: return c
        return None
    tc=pick("time"); lc=pick("left"); rc=pick("right"); totc=pick("total")
    onc=pick("onset"); offc=pick("offset")
    if tc is None:
        empty=pd.DataFrame(); return (pd.DataFrame({"Parameter":[],"Value":[]}), empty, dict(fps=np.nan,n_cycles=0))

    t=df[tc].astype(float).values
    if np.nanmax(t)>10.0: t=t/1000.0

    if totc is not None: total = df[totc].astype(float).values
    elif lc is not None and rc is not None: total=(df[lc].astype(float).values + df[rc].astype(float).values)/2.0
    else:
        empty=pd.DataFrame(); return (pd.DataFrame({"Parameter":[],"Value":[]}), empty, dict(fps=np.nan,n_cycles=0))
    left=df[lc].astype(float).values if lc else None
    right=df[rc].astype(float).values if rc else None

    dt=np.median(np.diff(t)) if len(t)>1 else 0.0
    fps=(1.0/dt) if dt>0 else 1500.0
    total_s=_smooth(total,fps)
    left_s=_smooth(left,fps) if left is not None else None
    right_s=_smooth(right,fps) if right is not None else None

    min_frames=max(int(0.002*fps),5)
    cycles=_build_cycles(t,total_s,min_frames=min_frames)

    AP,TP=_ap_tp(t,total_s,cycles); AS=_as_range(left_s,right_s,cycles); PS=_ps(left_s,right_s,t,cycles)

    diff_total=np.abs(np.diff(total_s, prepend=total_s[0]))
    W=max(int(round((adv.get("W_ms",35.0)/1000.0)*fps)),3)
    E_total=_moving_rms(diff_total,W)

    onset_series=df[onc].astype(float).values if onc else None
    offset_series=df[offc].astype(float).values if offc else None

    if force_total_for_onset:
        E_on=E_total
    else:
        if onset_series is not None:
            cand=_moving_rms(np.abs(np.diff(onset_series, prepend=onset_series[0])), W)
            good=(np.nanstd(cand)>1e-6) and (np.count_nonzero(cand>np.nanmean(cand))>5)
            E_on=cand if good else E_total
        else:
            E_on=E_total
    E_off=_moving_rms(np.abs(np.diff(offset_series, prepend=offset_series[0])), W) if offset_series is not None else E_total

    if show_debug_charts:
        st.write("🔎 Quick Signal Check"); st.line_chart(pd.DataFrame({"E_total":E_total}))
        st.line_chart(pd.DataFrame({"E_on":E_on,"E_off":E_off}))

    # ---- thresholds from baseline ----
    baseline_s=adv.get("baseline_s",0.06)      # Clinical default
    nB=max(int(round(baseline_s*fps)),5)

    def _thr(E):
        base=E[:min(nB,len(E))]; mu0=float(np.mean(base)) if len(base) else 0.0
        s0=float(np.std(base,ddof=1)) if len(base)>1 else 0.0
        return mu0 + adv.get("k",0.90)*s0       # Clinical default k

    thr_on,thr_off=_thr(E_on),_thr(E_off)

    # ---- state machine params ----
    hysteresis_ratio=float(adv.get("hysteresis_ratio",0.78))
    min_duration_ms=float(adv.get("min_duration_ms",35))
    refractory_ms=float(adv.get("refractory_ms",30))
    Tlow_on=hysteresis_ratio*thr_on; Tlow_off=hysteresis_ratio*thr_off
    min_dur_n=max(1,int(round(min_duration_ms*fps/1000.0)))
    refrac_n=max(1,int(round(refractory_ms*fps/1000.0)))

    # ---- onset detection (start after baseline + 15ms) ----
    i_move=None; in_voiced=False; last_onset=-10**9; onset_idx=None
    start_allowed = nB + int(0.015*fps)
    for i in range(len(E_on)):
        x=E_on[i]
        if not in_voiced:
            if i>=start_allowed and x>thr_on and (i-last_onset)>=refrac_n:
                in_voiced=True; onset_idx=i; last_onset=i
        else:
            if x<Tlow_on and (i-onset_idx)>=min_dur_n:
                i_move=onset_idx; in_voiced=False; break
    if i_move is None:
        in_voiced=False; onset_idx=None; thr_rel=thr_on*0.95; dur_rel=max(1,int(0.7*min_dur_n))
        for i in range(start_allowed,len(E_on)):
            x=E_on[i]
            if not in_voiced:
                if x>thr_rel: in_voiced=True; onset_idx=i
            else:
                if x<Tlow_on and (i-onset_idx)>=dur_rel:
                    i_move=onset_idx; break

    # ---- steady & offset ----
    VOnT=np.nan; VOffT=np.nan; i_steady=None; i_last=None; i_end=None
    if len(cycles)>=3 and i_move is not None:
        g_amp=float(np.nanmax([np.nanmax(total_s[s:e])-np.nanmin(total_s[s:e]) for s,e in cycles]))
        t_move=float(t[i_move])
        amp_frac=adv.get("amp_frac",0.70); ap_thr=adv.get("ap_thr",0.97); tp_thr=adv.get("tp_thr",0.97)

        def _local_p(s,e):
            amps=[]; pers=[]; cnt=0
            for cs,ce in cycles:
                if ce<=s: continue
                if cs>=e: break
                seg=total_s[cs:ce]
                amps.append(float(np.nanmax(seg)-np.nanmin(seg)))
                pers.append(float(t[ce]-t[cs])); cnt+=1
                if cnt>=3: break
            if len(amps)<2 or len(pers)<2: return (0.0,0.0)
            def _p(v): m=np.nanmean(v); sd=np.nanstd(v,ddof=1); return max(0.0,min(1.0,1.0-(sd/m))) if m>0 else 0.0
            return (_p(amps), _p(pers))

        for s,e in cycles:
            if s<=i_move: continue
            amp=float(np.nanmax(total_s[s:e])-np.nanmin(total_s[s:e]))
            if g_amp>0 and amp>=amp_frac*g_amp:
                AP_loc,TP_loc=_local_p(s,e)
                if AP_loc>=ap_thr and TP_loc>=tp_thr: i_steady=int(s); break
        if i_steady is None:
            for s,e in cycles:
                if s>i_move: i_steady=int(s); break
        if i_steady is None: i_steady=min(i_move+1, len(t)-1)

        for s,e in reversed(cycles):
            amp=float(np.nanmax(total_s[s:e])-np.nanmin(total_s[s:e]))
            if g_amp>0 and amp>=amp_frac*g_amp: i_last=int(s); break
        if i_last is None: i_last=cycles[-1][1]

        in_voiced=False; seg_start=None; start_i=max(0,int(i_last))
        for i in range(start_i,len(E_off)):
            x=E_off[i]
            if not in_voiced:
                if x>thr_off: in_voiced=True; seg_start=i
            else:
                if x<Tlow_off and (i-seg_start)>=min_dur_n:
                    i_end=i; in_voiced=False
        if i_end is None:
            M=int(adv.get("M",40))
            above=(E_off>thr_off).astype(int)
            run=np.convolve(above, np.ones(M,int), mode="same")
            edges=np.diff(np.r_[0,(run>=M).astype(int),0])
            starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]-1
            m=np.where(starts>=i_last)[0]
            i_end=int(ends[m[-1]]) if len(m) else int(cycles[-1][1])

        t_steady=float(t[i_steady]); t_last=float(t[i_last]); t_end=float(t[min(i_end,len(t)-1)])
        VOnT=float(t_steady - t_move); VOffT=float(t_end - t_last)

    # ---- last-resort fallbacks ----
    def _first_cross_after(E,thr,start,min_len):
        run=0
        for i in range(start,len(E)):
            if E[i]>thr:
                run+=1
                if run>=min_len: return i-(min_len-1)
            else: run=0
        return None

    if (VOnT is None) or (isinstance(VOnT,float) and np.isnan(VOnT)) or (i_move is None or i_steady is None):
        i0=_first_cross_after(E_on,thr_on,nB,max(1,int(round(35*fps/1000.0))))
        if i0 is not None:
            s2=None
            for s,e in cycles:
                if s>i0: s2=s; break
            if s2 is None: s2=min(i0+max(1,int(round(35*fps/1000.0))), len(t)-1)
            i_move=i0 if i_move is None else i_move
            i_steady=s2 if i_steady is None else i_steady
            VOnT=float(t[i_steady]-t[i_move])

    if (('i_end' not in locals()) or (i_end is None)) and (len(E_off)>0):
        start_i=int(i_last) if (i_last is not None) else 0
        last_i=None; run=0
        for i in range(start_i,len(E_off)):
            if E_off[i]>thr_off: run+=1
            else:
                if run>=min_dur_n: last_i=i
                run=0
        if last_i is None: last_i=len(E_off)-1
        i_end=last_i
        if i_last is None and len(cycles)>0: i_last=cycles[-1][0]
        if i_last is not None: VOffT=float(t[min(i_end,len(t)-1)] - t[i_last])

    if (VOnT is None) or (isinstance(VOnT,float) and np.isnan(VOnT)): VOnT=0.0
    if (VOffT is None) or (isinstance(VOffT,float) and np.isnan(VOffT)): VOffT=0.0

    st.write(f"DEBUG ▶ FPS={fps:.1f}, cycles={len(cycles)}, nB={nB}, "
             f"thr_on={thr_on:.4g}, Tlow_on={Tlow_on:.4g}, thr_off={thr_off:.4g}, Tlow_off={Tlow_off:.4g}, "
             f"VOnT={VOnT*1000:.2f} ms, VOffT={VOffT*1000:.2f} ms")

    per_cycle=pd.DataFrame(dict(cycle=[], start_time=[], end_time=[]))
    summary=pd.DataFrame({
        "Parameter":[
            "Amplitude Periodicity (AP)","Time Periodicity (TP)",
            "Amplitude Symmetry (AS)","Phase Symmetry (PS)",
            "Voice Onset Time (VOnT, ms)","Voice Offset Time (VOffT, ms)"
        ],
        "Value":[AP,TP,AS,PS,VOnT*1000.0,VOffT*1000.0]
    })
    extras=dict(fps=fps, n_cycles=len(cycles), VOnT_ms=VOnT*1000.0, VOffT_ms=VOffT*1000.0)
    return summary, per_cycle, extras

# ---------------------------- UI -----------------------------------
uploaded = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx","csv"])

# Preset quick button
if "preset_applied" not in st.session_state:
    st.session_state.preset_applied=False

colp1, colp2 = st.columns([1,1])
if colp1.button("🎛 Clinical Optimized Preset 적용", disabled=st.session_state.preset_applied is True):
    st.session_state["baseline_s"]=0.06; st.session_state["k"]=0.90
    st.session_state["M"]=40; st.session_state["W_ms"]=35.0
    st.session_state["amp_frac"]=0.70; st.session_state["ap_thr"]=0.97; st.session_state["tp_thr"]=0.97
    st.session_state["force_total"]=False; st.session_state.preset_applied=True
    st.experimental_rerun()

if colp2.button("🔁 기본값으로 재설정"):
    for k_ in ["baseline_s","k","M","W_ms","amp_frac","ap_thr","tp_thr","force_total","preset_applied"]:
        if k_ in st.session_state: del st.session_state[k_]
    st.experimental_rerun()

with st.expander("⚙ 고급 설정 (필요 시만 조정)", expanded=False):
    c1,c2,c3,c4,c5 = st.columns(5)
    baseline_s = c1.number_input("Baseline 구간(s)", min_value=0.02, max_value=0.50,
                                 value=st.session_state.get("baseline_s",0.06), step=0.01, key="baseline_s")
    k          = c2.number_input("임계 배수 k", min_value=0.50, max_value=6.0,
                                 value=st.session_state.get("k",0.90), step=0.05, key="k")
    M          = c3.number_input("연속 프레임 M", 1, 150, value=st.session_state.get("M",40), step=1, key="M")
    W_ms       = c4.number_input("에너지 창(ms)", 2.0, 40.0, value=st.session_state.get("W_ms",35.0), step=1.0, key="W_ms")
    amp_frac   = c5.slider("정상화 최소 진폭 (max 비율)", 0.10, 0.90, value=st.session_state.get("amp_frac",0.70), step=0.01, key="amp_frac")

    c6,c7,c8 = st.columns(3)
    ap_thr    = c6.slider("AP 임계값(steady 힌트)", 0.70, 1.00, value=st.session_state.get("ap_thr",0.97), step=0.01, key="ap_thr")
    tp_thr    = c7.slider("TP 임계값(steady 힌트)", 0.70, 1.00, value=st.session_state.get("tp_thr",0.97), step=0.01, key="tp_thr")
    force_total = c8.checkbox("onset 열 무시(총 에너지 사용)", value=st.session_state.get("force_total",False), key="force_total")

with st.expander("🔬 Hysteresis/Debounce & Debug", expanded=False):
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Hysteresis Ratio","0.78"); c2.metric("Min Duration (ms)","35")
    c3.metric("Refractory (ms)","30"); show_debug = c4.checkbox("Quick charts 보기", value=False)

# RMSE buffer
if "rmse_rows" not in st.session_state:
    st.session_state.rmse_rows=[]

adv=dict(
    baseline_s=baseline_s, k=k, M=M, W_ms=W_ms, amp_frac=amp_frac,
    ap_thr=ap_thr, tp_thr=tp_thr, hysteresis_ratio=0.78,
    min_duration_ms=35, refractory_ms=30
)

st.markdown("---")

if uploaded is not None:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    with st.spinner("분석 중..."):
        summary, per_cycle, extras = analyze(df, adv, show_debug_charts=show_debug, force_total_for_onset=force_total)

    st.subheader("✅ 결과 요약")
    st.dataframe(summary, use_container_width=True)
    st.write(f"FPS: {extras.get('fps', np.nan):.1f}, 검출된 사이클 수: {extras.get('n_cycles', 0)}")

    # RMSE evaluator
    st.markdown("### 📏 정량검증 (RMSE/MAE/BIAS)")
    ca, cb, cc, cd, ce = st.columns([1.2,1,1,1,1.2])
    case_id = ca.text_input("케이스 ID", value=uploaded.name)
    man_on  = cb.number_input("수동 Onset(ms)", value=0.0, step=0.01)
    man_off = cc.number_input("수동 Offset(ms)", value=0.0, step=0.01)
    auto_on = float(extras.get("VOnT_ms",0.0)); auto_off=float(extras.get("VOffT_ms",0.0))
    cd.metric("자동 Onset(ms)", f"{auto_on:.2f}"); ce.metric("자동 Offset(ms)", f"{auto_off:.2f}")

    if st.button("➕ 버퍼에 추가"):
        st.session_state.rmse_rows.append(
            dict(case=case_id, auto_on=auto_on, auto_off=auto_off, man_on=man_on, man_off=man_off,
                 err_on=auto_on-man_on, err_off=auto_off-man_off)
        )

    if st.session_state.rmse_rows:
        df_rmse=pd.DataFrame(st.session_state.rmse_rows)
        st.dataframe(df_rmse, use_container_width=True)

        def _rmse(x): return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan
        def _mae(x):  return float(np.mean(np.abs(x))) if len(x) else np.nan
        def _bias(x): return float(np.mean(x)) if len(x) else np.nan

        rmse_on=_rmse(df_rmse["err_on"].values); rmse_off=_rmse(df_rmse["err_off"].values)
        mae_on=_mae(df_rmse["err_on"].values);  mae_off=_mae(df_rmse["err_off"].values)
        bias_on=_bias(df_rmse["err_on"].values); bias_off=_bias(df_rmse["err_off"].values)

        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("RMSE Onset (ms)", f"{rmse_on:.2f}")
        m2.metric("RMSE Offset (ms)", f"{rmse_off:.2f}")
        m3.metric("MAE Onset (ms)", f"{mae_on:.2f}")
        m4.metric("MAE Offset (ms)", f"{mae_off:.2f}")
        m5.metric("Bias Onset (ms)", f"{bias_on:+.2f}")
        m6.metric("Bias Offset (ms)", f"{bias_off:+.2f}")

        colx, coly = st.columns(2)
        if colx.button("🧹 버퍼 초기화"):
            st.session_state.rmse_rows=[]; st.experimental_rerun()

        csv_buf=StringIO(); df_rmse.to_csv(csv_buf,index=False)
        st.download_button("⬇️ 결과 CSV 다운로드", data=csv_buf.getvalue(),
                           file_name="hsv_rmse_results.csv", mime="text/csv")
else:
    st.info("샘플 파일(시간 + 좌/우 또는 total, 선택적으로 onset/offset 컬럼)을 업로드해 주세요.")
