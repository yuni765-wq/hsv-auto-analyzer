# modules/parameter_tab.py
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def render_parameter_tab(base_csv_path: str, group_col: str = "group"):
    st.header("Parameter Comparison")

    # 원본(분석) 파일 로드: analysis_all.xlsx OR CSV
    data_source = st.text_input("Data file path", value=base_csv_path)
    if not data_source:
        st.warning("Please provide a data file path.")
        return

    try:
        if data_source.lower().endswith(".xlsx"):
            df = pd.read_excel(data_source)
        else:
            df = pd.read_csv(data_source)
    except Exception as e:
        st.error(f"Failed to read data: {e}")
        return

    if group_col not in df.columns:
        st.error(f"'{group_col}' column not found.")
        return

    # 파라미터 후보 탐지
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns found.")
        return

    # UI
    cols = st.multiselect("Parameters", numeric_cols, default=numeric_cols[:4])
    grp_order = st.multiselect("Group order (optional)", df[group_col].unique().tolist(),
                               default=list(df[group_col].unique()))
    log1p = st.checkbox("Apply log1p", value=False)

    if not cols:
        st.info("Select at least one parameter.")
        return

    # 표: 그룹별 기술통계
    st.subheader("Descriptive by group")
    desc_rows = []
    for p in cols:
        x = np.log1p(df[p]) if log1p else df[p]
        for g, sub in df.groupby(group_col):
            s = x.loc[sub.index]
            desc_rows.append({
                "parameter": p, group_col: g,
                "count": int(s.notna().sum()),
                "mean": float(np.nanmean(s)),
                "std": float(np.nanstd(s)),
                "median": float(np.nanmedian(s))
            })
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True)

    # 간단 분포 플롯(박스플롯)
    st.subheader("Distribution (Boxplot)")
    for p in cols:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        plot_groups = grp_order if grp_order else df[group_col].unique().tolist()
        data = [np.log1p(df.loc[df[group_col] == g, p].dropna()) if log1p
                else df.loc[df[group_col] == g, p].dropna() for g in plot_groups]
        ax.boxplot(data, labels=plot_groups, showfliers=False)
        ax.set_title(f"{p} (log1p={log1p})")
        ax.set_ylabel(p)
        st.pyplot(fig)
