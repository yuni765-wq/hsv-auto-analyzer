# -*- coding: utf-8 -*-
"""
R&D Analysis – v3β Full UI (UIrev1)
Whiteboard-tone / Clean layout / Minimal distraction

Assumes:
- modules/
    - stats_tab.py   (Statistical Analysis / Dunn FDR matrices)
    - parameter_tab.py
    - duration_tab.py
- Result CSVs in BASE_DIR:
    - desc_by_group_analysis_all.csv
    - kruskal_all_analysis_all.csv
    - dunn_matrix_<PARAM>_log1p_analysis_all.csv (AP, TP, AS_corr, PS_dist)
- Optional original data: analysis_all.xlsx (for Param/Duration tabs)
"""

import os
import streamlit as st

# ---- THEME HINT (run with: streamlit run app_v3beta_full_UIrev1.py) ----
# You can also set these in .streamlit/config.toml
WHITEBOARD_BG = "#ffffff"
SOFT_GRAY = "#f5f7fb"
ACCENT = "#1f6feb"   # subtle blue
MUTED = "#657388"

# ---- ENV / PATHS ----
BASE_DIR = os.environ.get("STAT_BASE_DIR", os.getcwd())
ANALYSIS_FILE = os.environ.get("STAT_ANALYSIS_FILE", os.path.join(BASE_DIR, "analysis_all.xlsx"))

# ---- IMPORT MODULES ----
from modules.stats_tab import render_dunn_heatmap, format_kruskal_table, StatTabData
from modules.parameter_tab import render_parameter_tab
from modules.duration_tab import render_duration_tab


# --------------------------- UI HELPERS --------------------------- #
def _whiteboard_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {WHITEBOARD_BG};
        }}
        section.main > div {{
            padding-top: 0.5rem;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }}
        .metric, .stDataFrame, .stTable {{
            background: {SOFT_GRAY};
            border-radius: 16px;
            padding: 4px 8px;
        }}
        .e1f1d6gn0 {{
            background: {SOFT_GRAY} !important;  /* dataframe toolbar area */
        }}
        .smallnote {{
            color: {MUTED};
            font-size: 0.9rem;
        }}
        .chip {{
            display: inline-block; padding: 2px 8px; border-radius: 999px;
            background: #eef2ff; color: #334155; border: 1px solid #e5e7eb; margin-right: 4px;
        }}
        .divider {{
            height: 1px; background: #e7e9ee; margin: 12px 0 16px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:1.6rem;font-weight:700;">R&D Analysis – v3β</div>
            <div class="chip">Whiteboard</div>
            <div class="chip">Log1p + Dunn(FDR)</div>
        </div>
        <div class="smallnote">AP · TP · AS · PS 비교 / 임계치 마킹(* ≤ 0.05, · ≤ 0.1)</div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar():
    with st.sidebar:
        st.markdown("### Settings")
        st.write(f"**BASE_DIR**: `{BASE_DIR}`")
        st.write(f"**ANALYSIS_FILE**: `{ANALYSIS_FILE}`")
        st.caption("환경변수로 바꾸려면:\n"
                   "`STAT_BASE_DIR`, `STAT_ANALYSIS_FILE` 사용")
        st.markdown("---")
        st.markdown("#### Quick Links")
        st.markdown("- Parameter Comparison")
        st.markdown("- Statistical Analysis")
        st.markdown("- Auto Duration")
        st.markdown("---")
        st.caption("Tip: 결과 CSV를 BASE_DIR에 두면 자동 인식됩니다.")


# --------------------------- TABS --------------------------- #
def tab_parameter():
    st.subheader("Parameter Comparison")
    st.caption("Boxplot + group-wise descriptive (optional log1p).")
    render_parameter_tab(ANALYSIS_FILE, group_col="group")


def tab_statistical():
    st.subheader("Statistical Analysis")
    st.caption("Dunn FDR (log1p) matrix + Kruskal + Descriptive.")

    data = StatTabData(BASE_DIR)
    params, pretty = data.list_params()

    if not params:
        st.warning("No Dunn matrices found.\nExpected: dunn_matrix_<PARAM>_log1p_analysis_all.csv")
        return

    # Selector row
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        disp_names = [pretty[p] for p in params]
        choice_disp = st.selectbox("Parameter", disp_names, index=0)
        choice_param = params[disp_names.index(choice_disp)]
    with c2:
        thr = st.slider("Primary threshold ‘*’", 0.01, 0.10, 0.05, 0.01)
    with c3:
        thr2 = st.slider("Secondary ‘·’", 0.01, 0.20, 0.10, 0.01)

    mat = data.load_matrix(choice_param)
    st.write("#### Dunn FDR (log1p) – Matrix")
    st.dataframe(mat.style.format("{:.3f}"), use_container_width=True)

    fig = render_dunn_heatmap(mat, p_thr=thr, p_thr2=thr2, title=f"Dunn FDR (log1p) – {choice_disp}")
    st.pyplot(fig)

    with st.expander("Kruskal–Wallis"):
        kw = data.load_kw()
        if kw is not None:
            st.dataframe(format_kruskal_table(kw), use_container_width=True)
        else:
            st.info("kruskal_all_analysis_all.csv not found.")

    with st.expander("Descriptive summary"):
        desc = data.load_desc()
        if desc is not None:
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("desc_by_group_analysis_all.csv not found.")

    # Export row
    st.markdown("##### Export")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download matrix CSV",
            data=mat.to_csv().encode("utf-8"),
            file_name=f"dunn_matrix_{choice_param}_log1p_analysis_all.csv",
            mime="text/csv",
        )
    with d2:
        import io
        buf = io.BytesIO()
        fig2 = render_dunn_heatmap(mat, p_thr=thr, p_thr2=thr2, title=f"Dunn FDR (log1p) – {choice_disp}")
        fig2.savefig(buf, format="png", dpi=180)
        st.download_button(
            "Download heatmap PNG",
            data=buf.getvalue(),
            file_name=f"dunn_matrix_{choice_param}_log1p.png",
            mime="image/png",
        )


def tab_duration():
    st.subheader("Auto Duration")
    st.caption("Summaries for AutoDuration_ms, or VOnT/VOffT → AutoDuration.")
    render_duration_tab(ANALYSIS_FILE)


# --------------------------- MAIN --------------------------- #
def main():
    st.set_page_config(page_title="R&D v3β – UIrev1", layout="wide")
    _whiteboard_css()
    _sidebar()
    _header()

    tabs = st.tabs(["Parameter Comparison", "Statistical Analysis", "Auto Duration"])
    with tabs[0]:
        tab_parameter()
    with tabs[1]:
        tab_statistical()
    with tabs[2]:
        tab_duration()


if __name__ == "__main__":
    main()
