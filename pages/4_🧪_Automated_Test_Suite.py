import streamlit as st
import pandas as pd
from diabetes_model import FEATURES, VALID_RANGES, load_model
from test_cases import execute_test_suite, TEST_CASES

# Page Configuration
st.set_page_config(
    page_title="Clinical Validation Suite - DiaPredict AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: Healthcare Clinical Theme & Deploy Button Suppression
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

    .badge-pass {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
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
st.sidebar.markdown("---")

st.sidebar.markdown("### 🧪 Protocol Execution")
if st.sidebar.button("▶️ Start the Test", key="sidebar_suite_run_btn", type="primary", use_container_width=True):
    st.session_state["suite_tests_executed"] = True

# Main Page Header
st.title("🧪 Standardized Clinical Protocol Validation Suite")
st.caption(
    "Clinical verification harness executing 8 standardized patient archetypes against the 6-biomarker decision tree pipeline "
    "(Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age)."
)

model = load_model()
if model is None:
    st.error("Model file not found (`model/diabetes_6feature_model.joblib`). Please train the model first.")
    st.stop()

# Action Controls
col_action1, col_action2, _ = st.columns([1.5, 1.5, 3])
with col_action1:
    start_test = st.button("▶️ Start the Test", key="main_suite_run_btn", type="primary", use_container_width=True)
with col_action2:
    auto_run = st.checkbox("Auto-run on page load", value=True)

if start_test:
    st.session_state["suite_tests_executed"] = True

should_run = st.session_state.get("suite_tests_executed", False) or auto_run

if should_run:
    with st.spinner("Executing 8 clinical benchmark test cases..."):
        passed, total, results = execute_test_suite(model)

    # Top metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="stat-box"><div class="stat-label">Total Clinical Archetypes</div><div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="stat-box"><div class="stat-label">Verified Concordant</div><div class="stat-value" style="color: #34d399;">{passed}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-box"><div class="stat-label">Discordant Cases</div><div class="stat-value" style="color: {"#f87171" if total - passed > 0 else "#94a3b8"};">{total - passed}</div></div>', unsafe_allow_html=True)
    with m4:
        acc_pct = (passed / total) * 100
        st.markdown(f'<div class="stat-box"><div class="stat-label">Protocol Pass Rate</div><div class="stat-value" style="color: #38bdf8;">{acc_pct:.0f}%</div></div>', unsafe_allow_html=True)

    st.write("")

    if passed == total:
        st.success(f"✅ **All {total}/{total} clinical protocol benchmarks passed with 100% concordance!** Model stratified every patient archetype according to medical guidelines.")
    else:
        st.error(f"❌ **{total - passed} clinical archetype(s) discordant out of {total}.** Check case diagnostics below.")

    st.markdown("### 📋 Protocol Verification Telemetry")

    # Format table for presentation
    table_rows = []
    for r in results:
        bm = r["Biomarkers"]
        bm_str = f"Preg: {bm['Pregnancies']} | FPG: {bm['Glucose']} mg/dL | DBP: {bm['BloodPressure']} mm Hg | BMI: {bm['BMI']} | DPF: {bm['DiabetesPedigreeFunction']} | Age: {bm['Age']}y"
        table_rows.append({
            "Case ID": r["Test ID"],
            "Validation Status": "✅ PASS" if r["Result"] == "PASS" else "❌ FAIL",
            "Clinical Ground Truth": r["Expected"],
            "CDSS Inferred Risk": r["Predicted"],
            "Decision Confidence": r["Confidence"],
            "Patient Physiological Profile (6 Biomarkers)": bm_str
        })

    df_results = pd.DataFrame(table_rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Detailed Accordion Breakdown
    st.markdown("### 🔬 In-Depth Archetype Clinical Analysis")
    for r in results:
        with st.expander(f"{r['Test ID']} — Ground Truth: {r['Expected']} | Inferred: {r['Predicted']} ({r['Confidence']}) — [{'CONCORDANT' if r['Result'] == 'PASS' else 'DISCORDANT'}]"):
            bm = r["Biomarkers"]
            c_left, c_right = st.columns(2)
            with c_left:
                st.write("**Patient Vital Signs & Metabolic Labs:**")
                st.write(f"- **Obstetric Parity (Pregnancies):** `{bm['Pregnancies']}`")
                st.write(f"- **Fasting Plasma Glucose:** `{bm['Glucose']} mg/dL`")
                st.write(f"- **Diastolic Blood Pressure:** `{bm['BloodPressure']} mm Hg`")
            with c_right:
                st.write(f"- **Body Mass Index:** `{bm['BMI']} kg/m²`")
                st.write(f"- **Diabetes Pedigree Function:** `{bm['DiabetesPedigreeFunction']}`")
                st.write(f"- **Patient Age:** `{bm['Age']} years`")
            
            st.caption(f"Protocol Concordance: **{r['Result']}** | Confidence: **{r['Confidence']}**")
else:
    st.info("Click **'▶️ Start the Test'** above or in the sidebar to execute the standardized clinical validation protocol.")
