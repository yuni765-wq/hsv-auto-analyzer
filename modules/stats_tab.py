# stats_tab.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Data loader ----
class StatTabData:
    def __init__(self, base_dir: str):
        self.base = base_dir
        self.desc_path = os.path.join(base_dir, "desc_by_group_analysis_all.csv")
        self.kw_path   = os.path.join(base_dir, "kruskal_all_analysis_all.csv")

    def list_params(self):
        # Detect available matrix files
        files = glob.glob(os.path.join(self.base, "dunn_matrix_*_log1p_analysis_all.csv"))
        params = []
        for f in files:
            name = os.path.basename(f)
            # dunn_matrix_<PARAM>_log1p_analysis_all.csv
            mid = name.replace("dunn_matrix_", "").replace("_log1p_analysis_all.csv", "")
            params.append(mid)
        # UI용 표기(AS_corr->AS, PS_dist->PS)
        pretty = {p: ("AS" if p=="AS_corr" else ("PS" if p=="PS_dist" else p)) for p in params}
        return params, pretty

    def load_desc(self):
        return pd.read_csv(self.desc_path) if os.path.exists(self.desc_path) else None

    def load_kw(self):
        return pd.read_csv(self.kw_path) if os.path.exists(self.kw_path) else None

    def load_matrix(self, param: str):
        path = os.path.join(self.base, f"dunn_matrix_{param}_log1p_analysis_all.csv")
        df = pd.read_csv(path, index_col=0)
        # ensure same order for rows/cols
        df = df.loc[df.index, df.index]
        return df

# ---- Renderer ----
def render_dunn_heatmap(mat: pd.DataFrame, p_thr=0.05, p_thr2=0.1, title="Dunn FDR (log1p)"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat.values, interpolation="nearest")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(mat.columns)
    ax.set_yticklabels(mat.index)
    ax.set_title(title)

    # annotate with p-values and significance marks
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j: 
                continue
            val = float(mat.values[i, j])
            mark = "*" if val <= p_thr else ("·" if val <= p_thr2 else "")
            ax.text(j, i, f"{val:.3f}{mark}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# ---- Small helper: table preview (Kruskal or Descriptive) ----
def format_kruskal_table(kw_df: pd.DataFrame):
    if kw_df is None: 
        return None
    cols = [c for c in kw_df.columns if c.lower() in ["parameter", "p", "h", "kruskal_h"]]
    return kw_df[cols].copy() if cols else kw_df.copy()
