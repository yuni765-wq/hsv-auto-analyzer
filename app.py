import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="HSV Auto Analyzer", layout="wide")

st.title("🪄 HSV Auto Analyzer")
st.markdown("Amplitude & Time Periodicity, Amplitude & Phase Symmetry 자동 계산기")

uploaded_file = st.file_uploader("엑셀(.xlsx) 또는 CSV(.csv) 파일을 업로드하세요", type=["xlsx", "csv"])

if uploaded_file:
    # 파일 읽기
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 데이터 미리보기")
    st.dataframe(df.head())

    # 열 이름 자동 탐지
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    left_col = [c for c in df.columns if 'left' in c.lower()][0]
    right_col = [c for c in df.columns if 'right' in c.lower()][0]

    time = df[time_col].values
    left = df[left_col].values
    right = df[right_col].values

    N = len(time)

    # ---------- 1. Amplitude Periodicity ----------
    amp_min = np.minimum(left[:-1], left[1:])
    amp_max = np.maximum(left[:-1], left[1:])
    amplitude_periodicity = np.mean(amp_min / amp_max)

    # ---------- 2. Time Periodicity ----------
    Ti = np.diff(time)
    time_periodicity = np.mean(np.minimum(Ti[:-1], Ti[1:]) / np.maximum(Ti[:-1], Ti[1:]))

    # ---------- 3. Amplitude Symmetry ----------
    amplitude_symmetry = np.mean(np.maximum(left, right) / np.minimum(left, right))
    
    # ---------- 4. Phase Symmetry ----------
    Ti_total = np.mean(Ti)
    phase_symmetry = np.mean(np.abs((left - right) / (left + right + 1e-9)))

    # ---------- 결과 요약 ----------
    results = pd.DataFrame({
        "Parameter": ["Amplitude Periodicity", "Time Periodicity", "Amplitude Symmetry", "Phase Symmetry"],
        "Value": [amplitude_periodicity, time_periodicity, amplitude_symmetry, phase_symmetry]
    })

    st.subheader("✅ 결과 요약")
    st.dataframe(results, use_container_width=True)

    # ---------- 5. 그래프 ----------
    st.subheader("📈 좌우 Gray Value 변화")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, left, label="Left gray", color="blue")
    ax.plot(time, right, label="Right gray", color="red", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gray value")
    ax.legend()
    st.pyplot(fig)

    # ---------- 6. 다운로드 ----------
    csv = results.to_csv(index=False).encode('utf-8-sig')
    st.download_button("💾 결과 다운로드 (CSV)", csv, file_name="HSV_results.csv", mime="text/csv")

else:
    st.info("분석할 파일을 업로드하면 자동으로 계산됩니다.")
