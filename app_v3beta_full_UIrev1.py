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
import io
import tempfile
import streamlit as st

# ---- THEME HINT ----
WHITEBOARD_BG = "#ffffff"
SOFT_GRAY = "#f5f7fb"
ACCENT = "#1f6feb"   # subtle blue
MUTED = "#657388"

# ---- PAGE CONFIG (상단 잘림 방지: 가장 먼저 선언) ----
st.set_page_config(page_title="R&D v3β – UIrev1", layout="wide")
st.markdown(
    """
    <style>
      /* 상단/헤더가 탭과 겹치지 않도록 여백 보정 */
      .block-container { padding-top: 2.2rem !important; padding-bottom: 2rem; max-width: 1100px; }
      header { margin-bottom: 0.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- ENV / PATHS ----
DEFAULT_BASE_DIR = os.environ.get("STAT_BASE_DIR", os.getcwd())
DEFAULT_ANALYSIS_FILE = os.environ.get("STAT_ANALYSIS_FILE", os.path.join(DEFAULT_BASE_DIR, "analysis_all.xlsx"))

# ---- IMPORT MODULES ----
from modules.stats_tab import render_dunn_heatmap, format_kruskal_table, StatTabData
from modules.parameter_tab import render_parameter_tab
from modules.duration_tab import render_duration_tab


# --------------------------- UTILS --------------------------- #
def _on_cloud() -> bool:
    # Streamlit Cloud에서는 이 값이 존재
    return os.environ.get("STREAMLIT_SERVER_BASE_URL") is not None

def _whiteboard_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {WHITEBOARD_BG}; }}
        section.main > div {{ padding-top: 0.5rem; }}
        .metric, .stDataFrame, .stTable {{
            background: {SOFT_GRAY};
            border-radius: 16px;
            padding: 4px 8px;
        }}
        .e1f1d6gn0 {{ background: {SOFT_GRAY} !important; }}  /* dataframe toolbar area */
        .smallnote {{ color: {MUTED}; font-size: 0.9rem; }}
        .chip {{
            display: inline-block; padding: 2px 8px; border-radius: 999px;
            background: #eef2ff; color: #334155; border: 1px solid #e5e7eb; margin-right: 4px;
        }}
        .divider {{ height: 1px; background: #e7e9ee; margin: 12px 0 16px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _header():
    st.markdown(
        """
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

def _save_uploaded_to_temp(uploaded_file) -> str:
    """
    UploadedFile을 임시 경로로 저장해 모듈들이 '파일 경로'로 읽을 수 있게 함.
    """
    suffix = ".csv" if uploaded_file.name.lower().endswith(".csv") else ".xlsx"
    fd, tmp_path = tempfile.mkstemp(prefix="analysis_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path


# --------------------------- SIDEBAR --------------------------- #
def _sidebar():
    # session_state에 현재 분석 파일 경로를 보관
    if "analysis_file_path" not in st.session_state:
        st.session_state.analysis_file_path = DEFAULT_ANALYSIS_FILE

    with st.sidebar:
        st.markdown("### Upload data")
        up = st.file_uploader(
            "CSV / XLSX 파일을 드래그&드롭하거나 선택하세요",
            type=["csv", "xlsx"], accept_multiple_files=False,
            help="analysis_all.xlsx 형식 권장 (columns: case_id, group, AP, TP, AS_corr, PS_dist …)",
        )

        if up is not None:
            try:
                tmp_path = _save_uploaded_to_temp(up)
                st.session_state.analysis_file_path = tmp_path
                st.success(f"업로드 완료: {up.name}")
            except Exception as e:
                st.error(f"업로드 파일 저장 중 오류: {e}")

        # 로컬 개발용: 경로 직접 입력(Cloud에서는 숨김)
        if not _on_cloud():
            st.markdown("---")
            new_path = st.text_input(
                "또는 파일 경로(로컬 개발용):",
                value=st.session_state.analysis_file_path or "",
                help="CSV/XLSX 경로 입력. Cloud에서는 의미가 없습니다."
            )
            if new_path and new_path != st.session_state.analysis_file_path:
                st.session_state.analysis_file_path = new_path

        st.caption(f"현재 분석 파일: `{st.session_state.analysis_file_path}`")

        # Stats용 BASE_DIR 안내 (결과 CSV가 있는 폴더)
        st.markdown("---")
        st.markdown("#### Quick Links")
        st.markdown("- Parameter Comparison")
        st.markdown("- Statistical Analysis")
        st.markdown("- Auto Duration")

        if not _on_cloud():
            st.markdown("---")
            st.markdown("### Settings (local)")
            st.code(f"BASE_DIR: {DEFAULT_BASE_DIR}")
            st.code(f"ANALYSIS_FILE (default): {DEFAULT_ANALYSIS_FILE}")
            st.caption("환경변수: STAT_BASE_DIR, STAT_ANALYSIS_FILE")


# --------------------------- TABS --------------------------- #
def tab_parameter():
    st.subheader("Parameter Comparison")
    st.caption("Boxplot + group-wise descriptive (optional log1p).")

    analysis_file = st.session_state.get("analysis_file_path", DEFAULT_ANALYSIS_FILE)
    if not analysis_file or (not os.path.exists(analysis_file) and not _on_cloud()):
        st.error("분석 파일을 찾을 수 없습니다. 좌측에서 업로드하거나 경로를 확인하세요.")
        return

    # parameter_tab은 '경로'를 기대하므로 그대로 전달
    render_parameter_tab(analysis_file, group_col="group")


def tab_statistical():
    st.subheader("Statistical Analysis")
    st.caption("Dunn FDR (log1p) matrix + Kruskal + Descriptive.")

    # BASE_DIR의 결과 CSV들을 읽음
    data = StatTabData(DEFAULT_BASE_DIR)
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
        buf = io.BytesIO()
        fig2 = render_dunn_heatmap(mat, p_thr=thr, p_thr2=thr2, title=f"Dunn FDR (log1p) – {choice_disp}")
        fig2.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        st.download_button(
            "Download heatmap PNG",
            data=buf.getvalue(),
            file_name=f"dunn_matrix_{choice_param}_log1p.png",
            mime="image/png",
        )


def tab_duration():
    st.subheader("Auto Duration")
    st.caption("Summaries for AutoDuration_ms, or VOnT/VOffT → AutoDuration.")

    analysis_file = st.session_state.get("analysis_file_path", DEFAULT_ANALYSIS_FILE)
    if not analysis_file or (not os.path.exists(analysis_file) and not _on_cloud()):
        st.error("분석 파일을 찾을 수 없습니다. 좌측에서 업로드하거나 경로를 확인하세요.")
        return

    render_duration_tab(analysis_file)


# --------------------------- MAIN --------------------------- #
def main():
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
