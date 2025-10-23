# modules/duration_tab.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def render_duration_tab(base_csv_path: str):
    st.header("Auto Duration & On/Off Times")

    file_path = st.text_input("Data file for duration", value=base_csv_path)
    if not file_path:
        st.warning("Provide a CSV/XLSX path that includes VOnT_ms / VOffT_ms or AutoDuration_ms.")
        return

    try:
        if file_path.lower().endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to read data: {e}")
        return

    # AutoDuration 보정
    if "AutoDuration_ms" not in df.columns:
        if {"VOnT_ms", "VOffT_ms"}.issubset(df.columns):
            df["AutoDuration_ms"] = df["VOffT_ms"] - df["VOnT_ms"]
        else:
            st.info("No AutoDuration_ms or VOnT/VOffT columns found.")
            return

    group_col_guess = "group" if "group" in df.columns else st.text_input("Group column name", value="group")
    if group_col_guess not in df.columns:
        st.info(f"'{group_col_guess}' not found — set the correct group column and rerun.")
        return

    # 표: 그룹별 AutoDuration 요약
    st.subheader("AutoDuration summary by group")
    out = df.groupby(group_col_guess)["AutoDuration_ms"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    st.dataframe(out, use_container_width=True)

    # 플롯
    st.subheader("AutoDuration distribution")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    groups = df[group_col_guess].unique().tolist()
    data = [df.loc[df[group_col_guess] == g, "AutoDuration_ms"].dropna() for g in groups]
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.set_ylabel("AutoDuration_ms")
    st.pyplot(fig)
