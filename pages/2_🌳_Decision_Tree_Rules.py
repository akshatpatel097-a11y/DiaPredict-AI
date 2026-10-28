import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import plot_tree

from diabetes_model import (
    FEATURES,
    get_tree_rules_text,
    load_model,
)

st.set_page_config(
    page_title="Clinical Diagnostic Tree - DiaPredict AI",
    page_icon="🌳",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit Deploy Button and Header Actions */
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    [data-testid="stDeployButton"],
    header [data-testid="stToolbarActions"],
    header [data-testid="stToolbarActions"] > button,
    .stDeployButton,
    button:has(svg[data-testid="stDeployIcon"]),
    [data-testid="manage-app-button"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .stApp {
        background: radial-gradient(circle at 20% 15%, #081a2e 0%, #060e1a 60%, #03070f 100%) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.markdown("### 🧭 App Navigation")
st.sidebar.page_link("app.py", label="Patient Intake & Triage", icon="🩺")
st.sidebar.page_link("pages/1_📊_Model_Performance.py", label="CDSS Diagnostic Performance", icon="📊")
st.sidebar.page_link("pages/2_🌳_Decision_Tree_Rules.py", label="Clinical Diagnostic Tree", icon="🌳")
st.sidebar.page_link("pages/3_📈_Exploratory_Data_Analysis.py", label="Epidemiological Population Data", icon="📈")
st.sidebar.page_link("pages/4_🧪_Automated_Test_Suite.py", label="Clinical Validation Suite", icon="🧪")

st.title("🌳 Clinical Diagnostic Decision Tree")
st.caption("Inspect the exact hierarchical diagnostic branching thresholds and criteria of the trained decision tree.")

model = load_model()
if model is None:
    st.error("Model file not found. Please train the model first.")
    st.stop()

tree_step = model.named_steps.get("tree", model.named_steps.get("classifier"))
actual_depth = tree_step.get_depth()
n_leaves = tree_step.get_n_leaves()

c1, c2 = st.columns([2, 1])
with c1:
    st.info(f"🌲 **Clinical Tree Architecture:** Hierarchy Depth = **{actual_depth}** | Terminal Decision Leaves = **{n_leaves}** | Splitting Metric = **Information Gain (Entropy)**")
with c2:
    display_depth = st.slider("Diagnostic Hierarchy Zoom", min_value=1, max_value=min(actual_depth, 6), value=min(actual_depth, 3))

# Render visual plot_tree with clinical healthcare styling
st.markdown("##### 🔍 Clinical Diagnostic Tree Flowchart")
fig_tree, ax_tree = plt.subplots(figsize=(15, 7.5), dpi=150, facecolor="#08101d")
ax_tree.set_facecolor("#08101d")

plot_tree(
    tree_step,
    max_depth=display_depth,
    feature_names=FEATURES,
    class_names=["Low Risk (0)", "High Risk (1)"],
    filled=True,
    rounded=True,
    precision=2,
    fontsize=8,
    ax=ax_tree
)
fig_tree.tight_layout()
st.pyplot(fig_tree)
plt.close(fig_tree)

st.markdown("---")
st.markdown("##### 📜 Clinical Branching Protocol Logic (`export_text`)")
st.caption("Note: Features are normalized by `StandardScaler` in the clinical pipeline prior to branching.")
rules_text = get_tree_rules_text(model)
st.code(rules_text, language="text")
