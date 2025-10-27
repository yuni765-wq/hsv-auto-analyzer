
# HSV Auto Analyzer v3.2 — Clinical Insight Release (Patch Guide)

This guide turns your current v3.1 app into **v3.2** with three features:

1) Debug View toggle (hidden by default; visible in Advanced/Research mode or via a toggle)
2) Automatic clinical interpretation line under the badge
3) Quality badge pinned across tabs

You will add one helper file and make **3 small edits** in `app.py`.

---

## STEP 0. Add the helper file
Place `insight_v32.py` next to your `app.py`.
(If you're reading these instructions here, the file is already generated for you to download.)

---

## STEP 1. Import and version label
At the top of `app.py` (with other imports), add:

```python
from insight_v32 import (
    VERSION_V32, compute_quality_from_env, render_quality_banner, inject_css
)
```

Then change your current version label block to use v3.2:

```python
VERSION_LABEL = VERSION_V32
st.set_page_config(page_title=VERSION_LABEL, layout="wide")
st.title(VERSION_LABEL)
st.caption("Isaka × Lian | Stable preset + Stats auto-load + Quality indicator + Clinical notes + Pinned banner")
inject_css(st)
```

---

## STEP 2. Debug view toggle (sidebar)
Inside your existing `with st.sidebar:` block, add a toggle. For example, right after the Preset section:

```python
st.markdown("---")
st.markdown("### 🔬 Debug / Research")
debug_view = st.toggle("Show debug info (연구자 전용)", value=False, key="debug_view")
```

This sets `st.session_state['debug_view']` you can reuse.

---

## STEP 3. Compute and store QI in render_overview(), render pinned badge
Inside your `render_overview(env: dict, ...)` function,
after you have `env` populated and before you display the table, add:

```python
# Compute QI and store for global (pinned) rendering
qi = compute_quality_from_env(env)
st.session_state['__qi_latest__'] = qi

# Render badge + clinical note for this page
render_quality_banner(st, qi, show_debug=st.session_state.get('debug_view', False), pinned=False)
```

If you already have a hand-written QI block, you can remove it now, since this one supersedes it.

Finally, after you create your tabs (e.g., `tabs = st.tabs([...])`) and **before** you draw content for each tab, render the **pinned** banner once using the stored value:

```python
qi_latest = st.session_state.get('__qi_latest__')
render_quality_banner(st, qi_latest, show_debug=st.session_state.get('debug_view', False), pinned=True)
```

This ensures the badge appears at the top even when switching tabs.

---

## Optional: remove old QI code
If you previously added a manual QI block, remove or comment it out to avoid duplicate badges and keep logic in one place:

```python
# [REMOVE/COMMENT] old AP_v/TP_v/PSD_v based QI calculations and badge rendering
```

---

## That’s it
Run `streamlit run app.py` and you should see:
- v3.2 title
- a sidebar toggle for Debug/Research
- a Quality badge with a clinical note on Overview
- a pinned Quality badge across tabs
