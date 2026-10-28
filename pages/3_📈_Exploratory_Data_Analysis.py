import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from diabetes_model import (
    FEATURES,
    TARGET,
    VALID_RANGES,
    ZERO_AS_MISSING,
    load_dataset,
)

st.set_page_config(
    page_title="Epidemiological Population Data - DiaPredict AI",
    page_icon="📈",
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

st.title("📈 Epidemiological Cohort & Population Biomarker Analysis")
st.caption("Deep clinical analytics into the Pima cohort: missing measurement rates, physiological distributions, and disease correlations.")

df_raw, df_clean = load_dataset()

# Row 1: Dataset table and Missing Values
col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown("##### 📑 Clinical Patient Cohort Data Preview")
    show_imputed = st.checkbox("View clean dataset (Zeroes replaced with NaN for unmeasured tests)", value=True)
    active_df = df_clean if show_imputed else df_raw
    st.dataframe(active_df.head(15), use_container_width=True)

with col2:
    st.markdown("##### ⚠️ Unmeasured Clinical Test Imputation Rates")
    missing_counts = (df_raw[ZERO_AS_MISSING] == 0).sum()
    missing_pct = (missing_counts / len(df_raw)) * 100

    fig_miss, ax_miss = plt.subplots(figsize=(5.5, 3.8), facecolor="#08101d")
    ax_miss.set_facecolor("#08101d")
    bars = ax_miss.barh(ZERO_AS_MISSING, missing_pct, color="#f43f5e", height=0.6)
    ax_miss.set_xlabel("% Patients Requiring Imputation (Unmeasured)", color="#94a3b8", fontsize=8)
    ax_miss.tick_params(colors="#cbd5e1", labelsize=8)
    ax_miss.grid(axis="x", color="#162a45", linestyle="--", alpha=0.6)
    for spine in ax_miss.spines.values():
        spine.set_color("#162a45")

    for i, v in enumerate(missing_pct):
        ax_miss.text(v + 1, i, f"{v:.1f}%", va="center", color="#f8fafc", fontsize=8, weight="bold")

    fig_miss.tight_layout()
    st.pyplot(fig_miss)
    plt.close(fig_miss)

st.markdown("---")

# Row 2: Distributions and Correlation Heatmap
col_dist, col_corr = st.columns([1.2, 1.2])

with col_dist:
    st.markdown("##### 🔬 Physiological Biomarker Distribution by Diagnostic Outcome")
    selected_feature = st.selectbox("Select Biomarker to Visualize", FEATURES, index=1)
    
    lo, hi, desc = VALID_RANGES[selected_feature]
    st.caption(f"**Clinical Description:** {desc} | **Standard Range:** {lo} – {hi}")

    g0 = df_clean[df_clean[TARGET] == 0][selected_feature].dropna()
    g1 = df_clean[df_clean[TARGET] == 1][selected_feature].dropna()

    fig_dist, ax_dist = plt.subplots(figsize=(6, 3.8), facecolor="#08101d")
    ax_dist.set_facecolor("#08101d")

    ax_dist.hist(g0, bins=20, alpha=0.65, color="#10b981", label="Normoglycemic Cohort (0)", density=True, edgecolor="#08101d")
    ax_dist.hist(g1, bins=20, alpha=0.65, color="#ef4444", label="Diabetic Cohort (1)", density=True, edgecolor="#08101d")

    ax_dist.set_xlabel(selected_feature, color="#cbd5e1", fontsize=9)
    ax_dist.set_ylabel("Normalized Density", color="#94a3b8", fontsize=9)
    ax_dist.tick_params(colors="#cbd5e1", labelsize=8)
    ax_dist.legend(facecolor="#0d1a2d", edgecolor="#162a45", labelcolor="#f8fafc", fontsize=8)
    ax_dist.grid(color="#162a45", linestyle="--", alpha=0.5)
    for spine in ax_dist.spines.values():
        spine.set_color("#162a45")

    fig_dist.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)

with col_corr:
    st.markdown("##### 🧬 Clinical Biomarker Correlation Matrix")
    corr = df_clean.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6.2, 4.2), facecolor="#08101d")
    ax_corr.set_facecolor("#08101d")
    cax = ax_corr.matshow(corr, cmap="YlGnBu", vmin=-0.4, vmax=1.0)
    fig_corr.colorbar(cax, fraction=0.046, pad=0.04)

    cols = df_clean.columns.tolist()
    ax_corr.set_xticks(range(len(cols)))
    ax_corr.set_yticks(range(len(cols)))
    ax_corr.set_xticklabels(cols, rotation=45, ha="left", color="#cbd5e1", fontsize=7)
    ax_corr.set_yticklabels(cols, color="#cbd5e1", fontsize=7)
    ax_corr.tick_params(colors="#cbd5e1")
    for spine in ax_corr.spines.values():
        spine.set_color("#162a45")

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            text_color = "#041b2d" if val > 0.4 else "#e2e8f0"
            ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5)

    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)
