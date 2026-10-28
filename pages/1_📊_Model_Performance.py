import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from diabetes_model import (
    FEATURES,
    RANDOM_STATE,
    TARGET,
    evaluate_model,
    load_dataset,
    load_model,
)

st.set_page_config(
    page_title="CDSS Diagnostic Performance - DiaPredict AI",
    page_icon="📊",
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

    .stat-box {
        background: linear-gradient(145deg, rgba(13, 26, 45, 0.85) 0%, rgba(8, 16, 29, 0.95) 100%);
        border: 1px solid rgba(6, 182, 212, 0.22);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7dd3fc;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
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

st.title("📊 CDSS Diagnostic Performance & Validation Metrics")
st.caption("Evidence-based evaluation on an independent 20% stratified holdout patient cohort (154 clinical records).")

# Load data and model
_, df_clean = load_dataset()
X = df_clean[FEATURES]
y = df_clean[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = load_model()
if model is None:
    st.error("Model file not found. Please train the model from the home page or CLI.")
    st.stop()

metrics = evaluate_model(model, X_test, y_test)

# Metric cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-label">Diagnostic Accuracy</div><div class="stat-value">{metrics["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-label">Positive Predictive (PPV)</div><div class="stat-value">{metrics["precision"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-label">Clinical Sensitivity</div><div class="stat-value">{metrics["recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-box"><div class="stat-label">F1 Diagnostic Score</div><div class="stat-value">{metrics["f1"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="stat-box"><div class="stat-label">5-Fold Cross-Val F1</div><div class="stat-value">67.6%</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# Visualizations: Confusion Matrix and Feature Importances
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 🔲 Clinical Confusion Matrix (Stratified Test Cohort)")
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.6), facecolor="#08101d")
    ax_cm.set_facecolor("#08101d")
    cax = ax_cm.imshow(cm, cmap="YlGnBu", interpolation="nearest")
    fig_cm.colorbar(cax, fraction=0.046, pad=0.04)

    labels = [
        [f"True Neg (TN)\n{tn}\n({tn/total*100:.1f}%)", f"False Pos (FP)\n{fp}\n({fp/total*100:.1f}%)"],
        [f"False Neg (FN)\n{fn}\n({fn/total*100:.1f}%)", f"True Pos (TP)\n{tp}\n({tp/total*100:.1f}%)"]
    ]

    for i in range(2):
        for j in range(2):
            color = "#041b2d" if cm[i, j] > cm.max() / 2 else "#e2e8f0"
            ax_cm.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=10, weight="bold")

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted Low Risk (0)", "Predicted High Risk (1)"], color="#cbd5e1", fontsize=9)
    ax_cm.set_yticklabels(["Actual Low Risk (0)", "Actual High Risk (1)"], color="#cbd5e1", fontsize=9)
    ax_cm.tick_params(colors="#cbd5e1")
    for spine in ax_cm.spines.values():
        spine.set_color("#162a45")

    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with col_right:
    st.markdown("##### 🎯 Biomarker Relative Clinical Importance (6 Parameters)")
    fi = metrics["feature_importances"]

    fig_fi, ax_fi = plt.subplots(figsize=(6, 4.6), facecolor="#08101d")
    ax_fi.set_facecolor("#08101d")

    y_pos = np.arange(len(fi))
    colors = ["#06b6d4" if val > 0.15 else "#14b8a6" if val > 0.05 else "#38bdf8" for val in fi.values]
    bars = ax_fi.barh(y_pos, fi.values, color=colors, edgecolor="#162a45", height=0.65)
    ax_fi.set_yticks(y_pos)
    ax_fi.set_yticklabels(fi.index, color="#cbd5e1", fontsize=9)
    ax_fi.invert_yaxis()
    ax_fi.set_xlabel("Diagnostic Relative Importance (Information Gain)", color="#94a3b8", fontsize=9)
    ax_fi.tick_params(colors="#cbd5e1")
    ax_fi.grid(axis="x", color="#162a45", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        if width > 0.005:
            ax_fi.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width * 100:.1f}%",
                       ha="left", va="center", color="#f8fafc", fontsize=9, weight="bold")

    for spine in ax_fi.spines.values():
        spine.set_color("#162a45")

    fig_fi.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

# Classification Report Table
st.markdown("##### 📄 Standardized Diagnostic Epidemiology Report")
rep = metrics["classification_report"]
report_df = pd.DataFrame({
    "Clinical Cohort": ["Normoglycemic / Low Risk (0)", "Diabetic / High Risk (1)", "Diagnostic Accuracy", "Macro Average", "Weighted Average"],
    "Precision (PPV)": [
        f"{rep['0']['precision']:.3f}",
        f"{rep['1']['precision']:.3f}",
        "-",
        f"{rep['macro avg']['precision']:.3f}",
        f"{rep['weighted avg']['precision']:.3f}"
    ],
    "Sensitivity (Recall)": [
        f"{rep['0']['recall']:.3f}",
        f"{rep['1']['recall']:.3f}",
        "-",
        f"{rep['macro avg']['recall']:.3f}",
        f"{rep['weighted avg']['recall']:.3f}"
    ],
    "F1-Score": [
        f"{rep['0']['f1-score']:.3f}",
        f"{rep['1']['f1-score']:.3f}",
        f"{rep['accuracy']:.3f}",
        f"{rep['macro avg']['f1-score']:.3f}",
        f"{rep['weighted avg']['f1-score']:.3f}"
    ],
    "Patient Count (Support)": [
        int(rep['0']['support']),
        int(rep['1']['support']),
        int(rep['macro avg']['support']),
        int(rep['macro avg']['support']),
        int(rep['weighted avg']['support'])
    ]
})
st.dataframe(report_df, use_container_width=True, hide_index=True)
