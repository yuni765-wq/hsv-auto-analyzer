# rnd_v3beta.py
# Adapter shim to render the existing R&D UI script inside app.py

import os
import runpy
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_THIS_DIR, "app_v3beta_full_UIrev1.py")

def render_rnd_v3beta():
    """Run the legacy R&D UI script inside the current Streamlit session."""
    if not os.path.exists(_TARGET):
        st.error("R&D 스크립트(app_v3beta_full_UIrev1.py)를 찾을 수 없습니다.")
        st.caption(f"찾은 경로: {_TARGET}")
        return

    # 선택사항: 좌측 상단으로 돌아가기 아이콘 숨기기 등, 필요 시 UI 조정 가능
    # st.markdown("<style>[data-testid='stSidebarNav'] { display:none; }</style>", unsafe_allow_html=True)

    # run the script as if it were __main__
    try:
        runpy.run_path(_TARGET, run_name="__main__")
    except Exception as e:
        st.error("R&D Analysis 스크립트 실행 중 오류가 발생했습니다.")
        with st.expander("오류 상세"):
            st.exception(e)
