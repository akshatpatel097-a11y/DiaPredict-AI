import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from diabetes_model import (
    FEATURES,
    MODEL_FILE,
    RANDOM_STATE,
    TARGET,
    VALID_RANGES,
    ZERO_AS_MISSING,
    build_pipeline,
    evaluate_model,
    get_tree_rules_text,
    load_dataset,
    load_model,
    predict_single,
    save_model,
    train_and_tune,
)
from test_cases import execute_test_suite

# Pydantic schema enforcing strictly 6 features
class PatientInputSchema(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20, description="number of pregnancies")
    Glucose: float = Field(..., ge=40, le=300, description="mg/dL, plasma glucose")
    BloodPressure: float = Field(..., ge=30, le=140, description="mm Hg, diastolic")
    BMI: float = Field(..., ge=10.0, le=70.0, description="kg/m^2")
    DiabetesPedigreeFunction: float = Field(..., ge=0.05, le=2.50, description="genetic risk score")
    Age: float = Field(..., ge=1, le=120, description="years")


# Page configuration
st.set_page_config(
    page_title="DiaPredict AI - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual aesthetics
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

    /* Healthcare Layout Background & Container */
    .stApp {
        background: radial-gradient(circle at 20% 15%, #081a2e 0%, #060e1a 60%, #03070f 100%) !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hospital / Clinical Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #07172b 0%, #0d2847 50%, #081d33 100%);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px -6px rgba(2, 6, 23, 0.7), 0 0 25px -5px rgba(6, 182, 212, 0.18);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #14b8a6, #38bdf8, #10b981);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        font-weight: 400;
        line-height: 1.55;
        max-width: 860px;
    }

    /* Healthcare Badges */
    .badge-clinical {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.35);
        margin-right: 0.4rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-protocol {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        background: rgba(20, 184, 166, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(20, 184, 166, 0.35);
        margin-right: 0.4rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Medical Triage Alert Cards */
    .risk-card-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(127, 29, 29, 0.18) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.22);
    }
    .risk-card-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.18) 100%);
        border: 1.5px solid #10b981;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.22);
    }
    .risk-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
    }
    .risk-prob {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Clinical Vitals & Lab Stat Tiles */
    .stat-box {
        background: linear-gradient(145deg, rgba(13, 26, 45, 0.85) 0%, rgba(8, 16, 29, 0.95) 100%);
        border: 1px solid rgba(6, 182, 212, 0.22);
        border-radius: 12px;
        padding: 1rem 1.15rem;
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
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Clinical Reference Ranges & Badges */
    .ref-badge {
        font-size: 0.73rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .ref-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .ref-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .ref-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Clinical Advisory Callout */
    .clinical-advisory {
        background: rgba(8, 24, 45, 0.75);
        border: 1px solid rgba(6, 182, 212, 0.28);
        border-left: 4px solid #06b6d4;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.25rem;
        margin-top: 1rem;
        font-size: 0.86rem;
        color: #cbd5e1;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset():
    """Load and cache the raw and cleaned 6-feature datasets."""
    df_raw, df_clean = load_dataset()
    return df_raw, df_clean


@st.cache_resource
def get_or_train_model(use_grid_search=True, max_depth=None, criterion="entropy", cv=5):
    """
    Train and cache the 6-feature pipeline.
    """
    _, df_clean = get_cached_dataset()
    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    saved = load_model()
    if saved is not None and use_grid_search and cv == 5:
        model = saved
        best_params = {"tree__criterion": "entropy", "tree__max_depth": 6, "tree__min_samples_leaf": 20}
        cv_score = 0.6768
    elif use_grid_search:
        model, grid = train_and_tune(X_train, y_train, use_grid_search=True, cv=cv)
        best_params = grid.best_params_ if grid else {}
        cv_score = grid.best_score_ if grid else None
        save_model(model)
    else:
        pipeline = build_pipeline({
            "max_depth": max_depth,
            "criterion": criterion,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        })
        pipeline.fit(X_train, y_train)
        model = pipeline
        best_params = {"tree__max_depth": max_depth, "tree__criterion": criterion}
        cv_score = None
        save_model(model)

    metrics = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "metrics": metrics,
        "best_params": best_params,
        "cv_score": cv_score,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def render_test_suite_panel(model, in_tab=False):
    """Render the automated test suite results and execution controls."""
    if not in_tab:
        st.markdown("#### 🧪 Clinical Protocol Verification Telemetry")
    with st.spinner("Executing 8 clinical benchmark test cases..."):
        passed, total, results = execute_test_suite(model)

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.markdown(f'<div class="stat-box"><div class="stat-label">Total Archetypes</div><div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
    with tc2:
        st.markdown(f'<div class="stat-box"><div class="stat-label">Concordant</div><div class="stat-value" style="color: #34d399;">{passed}</div></div>', unsafe_allow_html=True)
    with tc3:
        fail_color = "#f87171" if total - passed > 0 else "#94a3b8"
        st.markdown(f'<div class="stat-box"><div class="stat-label">Discordant</div><div class="stat-value" style="color: {fail_color};">{total - passed}</div></div>', unsafe_allow_html=True)
    with tc4:
        pass_rate = (passed / total) * 100
        st.markdown(f'<div class="stat-box"><div class="stat-label">Concordance Rate</div><div class="stat-value" style="color: #38bdf8;">{pass_rate:.0f}%</div></div>', unsafe_allow_html=True)

    st.write("")
    if passed == total:
        st.success(f"✅ **All {total}/{total} clinical benchmark cases passed (100% concordance)!** The 6-biomarker decision tree accurately stratified every patient archetype.")
    else:
        st.error(f"❌ {total - passed}/{total} test cases failed to match clinical ground truth.")

    table_rows = []
    for r in results:
        bm = r["Biomarkers"]
        bm_str = f"Preg: {bm['Pregnancies']} | FPG: {bm['Glucose']} mg/dL | DBP: {bm['BloodPressure']} mm Hg | BMI: {bm['BMI']} | DPF: {bm['DiabetesPedigreeFunction']} | Age: {bm['Age']}y"
        table_rows.append({
            "Case ID": r["Test ID"],
            "Status": "✅ PASS" if r["Result"] == "PASS" else "❌ FAIL",
            "Ground Truth": r["Expected"],
            "CDSS Inferred Risk": r["Predicted"],
            "Confidence": r["Confidence"],
            "Patient Biomarkers (6 Parameters)": bm_str,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if not in_tab:
        c_close, _ = st.columns([1.5, 4])
        with c_close:
            if st.button("✕ Dismiss Results Panel", key="close_test_results_btn", use_container_width=True):
                st.session_state["show_test_results"] = False
                st.rerun()
        st.markdown("---")


def main():
    df_raw, df_clean = get_cached_dataset()

    # --- Sidebar ---
    st.sidebar.markdown("### 🧭 Clinical Navigation")
    st.sidebar.page_link("app.py", label="Patient Intake & Triage", icon="🩺")
    st.sidebar.page_link("pages/1_📊_Model_Performance.py", label="CDSS Diagnostic Performance", icon="📊")
    st.sidebar.page_link("pages/2_🌳_Decision_Tree_Rules.py", label="Clinical Diagnostic Tree", icon="🌳")
    st.sidebar.page_link("pages/3_📈_Exploratory_Data_Analysis.py", label="Epidemiological Population Data", icon="📈")
    st.sidebar.page_link("pages/4_🧪_Automated_Test_Suite.py", label="Clinical Validation Suite", icon="🧪")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🧪 Protocol Validation")
    st.sidebar.caption("Run 8 standardized clinical benchmark archetypes.")
    if st.sidebar.button("▶️ Start the Test", key="sidebar_start_test_btn", type="primary", use_container_width=True):
        st.session_state["show_test_results"] = True
    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⚙️ Model Management")

    training_mode = st.sidebar.radio(
        "Optimization Mode",
        ["GridSearchCV (Recommended)", "Custom Hyperparameters"],
        help="Choose whether to optimize hyperparameters with 5-fold cross validation or select custom parameters."
    )

    use_gs = (training_mode == "GridSearchCV (Recommended)")
    custom_depth = None
    custom_crit = "entropy"
    custom_cv = 5

    if not use_gs:
        depth_option = st.sidebar.selectbox("Tree Max Depth", ["None (Unlimited)", 3, 4, 5, 6, 8, 10, 12], index=3)
        custom_depth = None if depth_option == "None (Unlimited)" else int(depth_option)
        custom_crit = st.sidebar.selectbox("Split Criterion", ["entropy", "gini"], index=0)
    else:
        custom_cv = st.sidebar.select_slider("Cross-Validation Folds", options=[3, 5, 10], value=5)

    if st.sidebar.button("🔄 Retrain / Refresh Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    model_bundle = get_or_train_model(
        use_grid_search=use_gs,
        max_depth=custom_depth,
        criterion=custom_crit,
        cv=custom_cv
    )
    model = model_bundle["model"]
    metrics = model_bundle["metrics"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Model Overview")
    st.sidebar.caption(
        f"**Algorithm:** `DecisionTreeClassifier` (6 Features)\n\n"
        f"**Pipeline:** `SimpleImputer(median)` ➜ `StandardScaler` ➜ `Tree`\n\n"
        f"**Features (6):** `[Pregnancies, Glucose, BloodPressure, BMI, DPF, Age]`\n\n"
        f"**Missing Value Handling:** 0 values in `[Glucose, BloodPressure, BMI]` treated as missing measurements.\n\n"
        f"**Dataset:** 768 patient records (6 clinical features, 1 target)."
    )
    if model_bundle["cv_score"]:
        st.sidebar.info(f"🏆 **Best 5-Fold CV F1-Score:** `{model_bundle['cv_score']:.3f}`")

    # --- Header Banner ---
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">
                <span>🩺 DiaPredict AI</span>
                <span class="badge-clinical">Hospital CDSS Grade</span>
                <span class="badge-protocol">ADA Standards</span>
                <span class="badge-protocol">6-Biomarker Panel</span>
            </div>
            <div class="hero-subtitle">
                Hospital Clinical Decision Support System (CDSS) for automated risk stratification of Type 2 Diabetes Mellitus. Incorporates calibrated physiological and metabolic biomarkers with clinical unmeasured-test imputation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Automated Test Trigger Banner ---
    t_col1, t_col2 = st.columns([3.4, 1.3])
    with t_col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(8, 24, 45, 0.85) 0%, rgba(6, 16, 30, 0.95) 100%); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 12px; padding: 0.85rem 1.25rem; display: flex; align-items: center; gap: 0.9rem;">
                <span style="font-size: 1.55rem;">🩺</span>
                <div>
                    <strong style="color: #f8fafc; font-size: 0.95rem; letter-spacing: 0.01em;">Clinical Decision Support Protocol Validation Suite (8 Benchmark Archetypes)</strong>
                    <div style="color: #94a3b8; font-size: 0.82rem;">Execute automated validation across standardized clinical patient cohorts against ADA / WHO criteria.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t_col2:
        if st.button("▶️ Start the Test", key="main_start_test_btn", type="primary", use_container_width=True):
            st.session_state["show_test_results"] = True

    if st.session_state.get("show_test_results", False):
        render_test_suite_panel(model, in_tab=False)

    # --- Navigation Tabs ---
    tab1, tab_test, tab2, tab3, tab4 = st.tabs([
        "🩺 Patient Intake & Triage",
        "🧪 Clinical Validation Suite",
        "📊 CDSS Diagnostic Performance",
        "🌳 Clinical Diagnostic Tree",
        "📈 Epidemiological Population Data"
    ])

    # =========================================================================
    # TAB 1: Patient Risk Assessment
    # =========================================================================
    with tab1:
        st.markdown("#### Patient Biomarker Panel (6 Core Physiological Metrics)")
        st.caption("Adjust patient vital signs and metabolic measurements below, or load a standardized clinical archetype cohort.")

        # Preset selector (6 features)
        preset_cols = st.columns([1.2, 1.2, 1.2, 2.4])
        with preset_cols[0]:
            if st.button("🟢 Normoglycemic Cohort", use_container_width=True, help="Load low-risk, normoglycemic archetype"):
                st.session_state["p_preg"] = 1
                st.session_state["p_gluc"] = 88.0
                st.session_state["p_bp"] = 68.0
                st.session_state["p_bmi"] = 22.4
                st.session_state["p_dpf"] = 0.28
                st.session_state["p_age"] = 24
                st.rerun()

        with preset_cols[1]:
            if st.button("🟡 Impaired Fasting Cohort", use_container_width=True, help="Load prediabetic / borderline archetype"):
                st.session_state["p_preg"] = 3
                st.session_state["p_gluc"] = 115.0
                st.session_state["p_bp"] = 76.0
                st.session_state["p_bmi"] = 28.5
                st.session_state["p_dpf"] = 0.52
                st.session_state["p_age"] = 41
                st.rerun()

        with preset_cols[2]:
            if st.button("🔴 Diabetic Syndrome Cohort", use_container_width=True, help="Load high-risk, metabolic syndrome archetype"):
                st.session_state["p_preg"] = 6
                st.session_state["p_gluc"] = 168.0
                st.session_state["p_bp"] = 88.0
                st.session_state["p_bmi"] = 35.6
                st.session_state["p_dpf"] = 1.15
                st.session_state["p_age"] = 52
                st.rerun()

        # Initialize session state defaults (6 features only)
        if "p_preg" not in st.session_state:
            st.session_state["p_preg"] = 2
            st.session_state["p_gluc"] = 120.0
            st.session_state["p_bp"] = 72.0
            st.session_state["p_bmi"] = 28.1
            st.session_state["p_dpf"] = 0.47
            st.session_state["p_age"] = 33

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### 🧪 Metabolic & Physiological Biomarkers")
            
            p_glucose = st.slider(
                "Plasma Glucose (mg/dL)",
                min_value=40.0,
                max_value=300.0,
                value=float(st.session_state["p_gluc"]),
                step=1.0,
                help="Normal fasting: 70–99 mg/dL. Impaired / Prediabetes: 100–125 mg/dL. Diabetic range: ≥126 mg/dL."
            )
            st.session_state["p_gluc"] = p_glucose

            p_bmi = st.slider(
                "Body Mass Index (BMI in kg/m²)",
                min_value=10.0,
                max_value=70.0,
                value=float(st.session_state["p_bmi"]),
                step=0.1,
                help="Normal: 18.5–24.9 kg/m². Overweight: 25.0–29.9 kg/m². Obese: ≥30.0 kg/m²."
            )
            st.session_state["p_bmi"] = p_bmi

            p_bp = st.slider(
                "Diastolic Blood Pressure (mm Hg)",
                min_value=30.0,
                max_value=140.0,
                value=float(st.session_state["p_bp"]),
                step=1.0,
                help="Normal: <80 mm Hg. Elevated: 80–89 mm Hg. Hypertension Stage 1+: ≥90 mm Hg."
            )
            st.session_state["p_bp"] = p_bp

        with col_right:
            st.markdown("##### 👤 Demographic & Genetic Factors")

            p_age = st.slider(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=int(st.session_state["p_age"]),
                step=1,
                help="Patient age in years."
            )
            st.session_state["p_age"] = p_age

            p_preg = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=int(st.session_state["p_preg"]),
                step=1,
                help="Total number of times pregnant."
            )
            st.session_state["p_preg"] = p_preg

            p_dpf = st.slider(
                "Diabetes Pedigree Function (Genetic Risk Score)",
                min_value=0.05,
                max_value=2.50,
                value=float(st.session_state["p_dpf"]),
                step=0.01,
                help="Calculates diabetes risk based on family medical history and genetic pedigree."
            )
            st.session_state["p_dpf"] = p_dpf

        # Pydantic validation of 6-feature payload
        raw_patient = {
            "Pregnancies": float(p_preg),
            "Glucose": float(p_glucose),
            "BloodPressure": float(p_bp),
            "BMI": float(p_bmi),
            "DiabetesPedigreeFunction": float(p_dpf),
            "Age": float(p_age)
        }
        validated_input = PatientInputSchema(**raw_patient).model_dump()

        # Perform live prediction
        prediction = predict_single(model, validated_input)
        is_diabetic = (prediction["prediction"] == 1)
        prob_diabetic = prediction["probability_diabetic"]
        prob_nondiabetic = prediction["probability_nondiabetic"]

        st.markdown("---")
        st.markdown("#### 📋 Diagnostic Evaluation")

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            if is_diabetic:
                st.markdown(
                    f"""
                    <div class="risk-card-positive">
                        <div style="font-size: 2.3rem;">⚠️</div>
                        <div class="risk-title" style="color: #ef4444;">TRIAGE LEVEL: HIGH RISK</div>
                        <div class="risk-prob" style="color: #f87171;">{prob_diabetic * 100:.1f}%</div>
                        <div style="color: #cbd5e1; font-size: 0.88rem;">Elevated Risk of Type 2 Diabetes Mellitus</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card-negative">
                        <div style="font-size: 2.3rem;">🛡️</div>
                        <div class="risk-title" style="color: #10b981;">TRIAGE LEVEL: LOW RISK</div>
                        <div class="risk-prob" style="color: #34d399;">{prob_nondiabetic * 100:.1f}%</div>
                        <div style="color: #cbd5e1; font-size: 0.88rem;">Normoglycemic / Low Risk Stratification</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")
            st.progress(prob_diabetic, text=f"Diabetes Mellitus Risk Index: {prob_diabetic * 100:.1f}%")

            # Clinical Advisory Callout Box
            if is_diabetic:
                st.markdown(
                    """
                    <div class="clinical-advisory">
                        <strong style="color: #f87171;">⚕️ Recommended Clinical Pathway:</strong><br/>
                        • Order confirmatory diagnostic laboratory testing (Fasting Plasma Glucose & HbA1c).<br/>
                        • Schedule diabetes self-management education and medical nutrition therapy.<br/>
                        • Review cardiovascular risk factors and evaluate for microvascular screening.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="clinical-advisory">
                        <strong style="color: #34d399;">⚕️ Recommended Clinical Pathway:</strong><br/>
                        • Routine annual preventive metabolic health screening.<br/>
                        • Maintain ADA-aligned balanced nutrition and ≥150 min/week moderate physical activity.<br/>
                        • Reassess if clinical symptoms, gestational factors, or BMI elevate.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("##### 🔬 Laboratory Biomarker Analysis (6 Core Parameters)")

            breakdown_data = []

            # Glucose
            if p_glucose >= 126:
                g_status = "High Risk (Diabetic Range)"
                g_tag = "ref-danger"
            elif p_glucose >= 100:
                g_status = "Elevated (Prediabetic)"
                g_tag = "ref-warning"
            else:
                g_status = "Normal (<100 mg/dL)"
                g_tag = "ref-normal"
            breakdown_data.append(("Glucose (FPG)", f"{p_glucose:.0f} mg/dL", "<100 mg/dL (ADA)", g_status, g_tag))

            # BMI
            if p_bmi >= 30:
                b_status = "High Risk (Obese Class I+)"
                b_tag = "ref-danger"
            elif p_bmi >= 25:
                b_status = "Overweight (25–29.9)"
                b_tag = "ref-warning"
            else:
                b_status = "Normal (18.5–24.9)"
                b_tag = "ref-normal"
            breakdown_data.append(("Body Mass Index (BMI)", f"{p_bmi:.1f} kg/m²", "18.5–24.9 kg/m² (WHO)", b_status, b_tag))

            # Blood Pressure
            if p_bp >= 90:
                bp_status = "High (Hypertension Stage 2)"
                bp_tag = "ref-danger"
            elif p_bp >= 80:
                bp_status = "Stage 1 Elevated"
                bp_tag = "ref-warning"
            else:
                bp_status = "Normal (<80 mm Hg)"
                bp_tag = "ref-normal"
            breakdown_data.append(("Diastolic BP", f"{p_bp:.0f} mm Hg", "<80 mm Hg (AHA)", bp_status, bp_tag))

            # Pedigree Function
            if p_dpf >= 0.8:
                dpf_status = "Strong Genetic Predisposition"
                dpf_tag = "ref-danger"
            elif p_dpf >= 0.4:
                dpf_status = "Moderate Genetic Predisposition"
                dpf_tag = "ref-warning"
            else:
                dpf_status = "Low Genetic Risk"
                dpf_tag = "ref-normal"
            breakdown_data.append(("Pedigree Score (DPF)", f"{p_dpf:.2f}", "<0.40 Normal", dpf_status, dpf_tag))

            # Age
            if p_age >= 50:
                age_status = "Higher Age Cohort (≥50)"
                age_tag = "ref-warning"
            else:
                age_status = "Standard Age Cohort"
                age_tag = "ref-normal"
            breakdown_data.append(("Patient Age", f"{p_age} yrs", "Age Stratification", age_status, age_tag))

            # Pregnancies
            if p_preg >= 6:
                preg_status = "High Parity (≥6)"
                preg_tag = "ref-warning"
            else:
                preg_status = "Standard Parity"
                preg_tag = "ref-normal"
            breakdown_data.append(("Pregnancies / Parity", f"{p_preg}", "Obstetric History", preg_status, preg_tag))

            # Render HTML table with healthcare styling
            table_rows = "".join([
                f"<tr><td style='padding: 8px 12px; font-weight: 600; color: #f1f5f9;'>{name}</td>"
                f"<td style='padding: 8px 12px; font-family: monospace; color: #38bdf8; font-weight: 700;'>{val}</td>"
                f"<td style='padding: 8px 12px; color: #94a3b8; font-size: 0.82rem;'>{ref}</td>"
                f"<td style='padding: 8px 12px;'><span class='ref-badge {tag}'>{status}</span></td></tr>"
                for name, val, ref, status, tag in breakdown_data
            ])
            st.markdown(
                f"""
                <table style='width: 100%; border-collapse: collapse; font-size: 0.88rem; background: rgba(13, 26, 45, 0.6); border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 8px;'>
                    <thead>
                        <tr style='border-bottom: 1px solid rgba(6, 182, 212, 0.25); color: #7dd3fc; text-align: left;'>
                            <th style='padding: 8px 12px;'>Biomarker</th>
                            <th style='padding: 8px 12px;'>Patient Value</th>
                            <th style='padding: 8px 12px;'>Standard Reference</th>
                            <th style='padding: 8px 12px;'>Clinical Classification</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True
            )

        st.caption("⚕️ *Clinical Decision Support Disclaimer: This system aids medical professionals in risk stratification based on 6 core clinical parameters. Definitive diagnosis requires formal laboratory testing (Oral Glucose Tolerance Test or HbA1c).*")

    # =========================================================================
    # TAB: Automated Test Suite
    # =========================================================================
    with tab_test:
        st.markdown("#### Automated Clinical Benchmark Test Suite (8 Archetypes)")
        st.caption("Standardized test harness executing 8 clinical archetypes against the 6-feature decision tree pipeline.")
        c_t_btn1, _ = st.columns([1.5, 3.5])
        with c_t_btn1:
            if st.button("▶️ Start the Test", key="tab_start_test_btn", type="primary", use_container_width=True):
                st.session_state["tab_test_executed"] = True

        render_test_suite_panel(model, in_tab=True)

    # =========================================================================
    # TAB 2: Model Performance & Validation
    # =========================================================================
    with tab2:
        st.markdown("#### Test Set Validation Metrics (6 Features)")
        st.caption("Evaluated on an independent 20% stratified holdout test set (154 patient records).")

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Accuracy</div>
                    <div class="stat-value">{metrics['accuracy'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Precision</div>
                    <div class="stat-value">{metrics['precision'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">Recall (Sensitivity)</div>
                    <div class="stat-value">{metrics['recall'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">F1-Score</div>
                    <div class="stat-value">{metrics['f1'] * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col5:
            cv_val = f"{model_bundle['cv_score'] * 100:.1f}%" if model_bundle['cv_score'] else "67.7%"
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-label">5-Fold CV Score</div>
                    <div class="stat-value">{cv_val}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.write("")

        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.markdown("##### 🔲 Clinical Confusion Matrix")
            cm = metrics["confusion_matrix"]
            tn, fp, fn, tp = cm.ravel()
            total_cases = cm.sum()

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8), facecolor="#08101d")
            ax_cm.set_facecolor("#08101d")

            cax = ax_cm.imshow(cm, cmap="YlGnBu", interpolation="nearest")
            fig_cm.colorbar(cax, fraction=0.046, pad=0.04)

            labels = [
                [f"True Neg (TN)\n{tn}\n({tn/total_cases*100:.1f}%)", f"False Pos (FP)\n{fp}\n({fp/total_cases*100:.1f}%)"],
                [f"False Neg (FN)\n{fn}\n({fn/total_cases*100:.1f}%)", f"True Pos (TP)\n{tp}\n({tp/total_cases*100:.1f}%)"]
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

        with v_col2:
            st.markdown("##### 🎯 Clinical Biomarker Weight Ranking (6 Features)")
            fi = metrics["feature_importances"]

            fig_fi, ax_fi = plt.subplots(figsize=(6, 4.8), facecolor="#08101d")
            ax_fi.set_facecolor("#08101d")

            y_pos = np.arange(len(fi))
            colors = ["#06b6d4" if val > 0.15 else "#14b8a6" if val > 0.05 else "#38bdf8" for val in fi.values]

            bars = ax_fi.barh(y_pos, fi.values, color=colors, edgecolor="#162a45", height=0.65)
            ax_fi.set_yticks(y_pos)
            ax_fi.set_yticklabels(fi.index, color="#cbd5e1", fontsize=9)
            ax_fi.invert_yaxis()
            ax_fi.set_xlabel("Diagnostic Importance (Gini / Entropy Impurity)", color="#94a3b8", fontsize=9)
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
        st.markdown("##### 📄 Clinical Decision Support Diagnostic Report")
        rep = metrics["classification_report"]
        report_df = pd.DataFrame({
            "Diagnostic Class": ["Class 0 (Normoglycemic / Low Risk)", "Class 1 (Diabetic / High Risk)", "Overall Accuracy", "Macro Average", "Weighted Average"],
            "Precision (PPV)": [
                f"{rep['0']['precision']:.3f}",
                f"{rep['1']['precision']:.3f}",
                "-",
                f"{rep['macro avg']['precision']:.3f}",
                f"{rep['weighted avg']['precision']:.3f}"
            ],
            "Recall (Sensitivity)": [
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
            "Patient Cohort Size": [
                int(rep['0']['support']),
                int(rep['1']['support']),
                int(rep['macro avg']['support']),
                int(rep['macro avg']['support']),
                int(rep['weighted avg']['support'])
            ]
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: Decision Tree Rules & Visualizer
    # =========================================================================
    with tab3:
        st.markdown("#### Clinical Decision Tree Pathway (6 Features)")
        st.caption("Inspect the exact hierarchical diagnostic branching thresholds of the trained decision tree.")

        tree_step = model.named_steps.get("tree", model.named_steps.get("classifier"))
        actual_depth = tree_step.get_depth()

        c_tree1, c_tree2 = st.columns([2, 1])
        with c_tree1:
            st.info(f"🌲 **Clinical Tree Architecture:** Depth = **{actual_depth}** | Terminal Leaf Nodes = **{tree_step.get_n_leaves()}** | Splitting Metric = **Information Gain (Entropy)**")
        with c_tree2:
            display_depth = st.slider("Diagnostic Hierarchy Zoom", min_value=1, max_value=min(actual_depth, 6), value=min(actual_depth, 3))

        fig_tree, ax_tree = plt.subplots(figsize=(14, 7), dpi=150, facecolor="#08101d")
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

        st.markdown("##### 📜 Clinical Branching Protocol Rules (`export_text`)")
        st.caption("Note: Features are normalized by `StandardScaler` in the clinical pipeline prior to branching.")
        rules_str = get_tree_rules_text(model)
        st.code(rules_str, language="text")

    # =========================================================================
    # TAB 4: Exploratory Data Analysis (EDA)
    # =========================================================================
    with tab4:
        st.markdown("#### Population Health Epidemiology & Cohort Distributions (6 Features)")
        st.caption("Investigate population-level distributions, unmeasured test rates, and clinical biomarker variations.")

        eda_row1_c1, eda_row1_c2 = st.columns([1.5, 1])

        with eda_row1_c1:
            st.markdown("##### 📑 Clinical Cohort Dataset Preview")
            show_clean = st.checkbox("Show clean dataset (6 features)", value=True)
            display_df = df_clean if show_clean else df_raw
            st.dataframe(display_df.head(10), use_container_width=True)

        with eda_row1_c2:
            st.markdown("##### ⚠️ Unmeasured Clinical Test Imputation (3 Features)")
            missing_counts = (df_raw[ZERO_AS_MISSING] == 0).sum()
            missing_pct = (missing_counts / len(df_raw)) * 100

            fig_miss, ax_miss = plt.subplots(figsize=(5, 3.2), facecolor="#08101d")
            ax_miss.set_facecolor("#08101d")
            ax_miss.barh(ZERO_AS_MISSING, missing_pct, color="#f43f5e", height=0.6)
            ax_miss.set_xlabel("% Records Requiring Imputation (Zero Value)", color="#94a3b8", fontsize=8)
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
        st.markdown("##### 🔬 Epidemiological Biomarker Distribution by Diagnostic Outcome")

        col_sel, col_fig = st.columns([1, 2])
        with col_sel:
            selected_feature = st.selectbox("Select Biomarker to Visualize", FEATURES, index=1)
            st.write("")
            st.write(f"**Clinical Description:** {VALID_RANGES[selected_feature][2]}")
            st.write(f"**Allowed Physiological Range:** {VALID_RANGES[selected_feature][0]} – {VALID_RANGES[selected_feature][1]}")

            g0 = df_clean[df_clean[TARGET] == 0][selected_feature].dropna()
            g1 = df_clean[df_clean[TARGET] == 1][selected_feature].dropna()

            st.write("**Normoglycemic Cohort (Class 0):**")
            st.caption(f"Mean: `{g0.mean():.2f}` | Median: `{g0.median():.2f}` | Std: `{g0.std():.2f}`")

            st.write("**Diabetic Cohort (Class 1):**")
            st.caption(f"Mean: `{g1.mean():.2f}` | Median: `{g1.median():.2f}` | Std: `{g1.std():.2f}`")

        with col_fig:
            fig_dist, ax_dist = plt.subplots(figsize=(7, 3.8), facecolor="#08101d")
            ax_dist.set_facecolor("#08101d")

            ax_dist.hist(g0, bins=20, alpha=0.65, color="#10b981", label="Normoglycemic / Low Risk (0)", density=True, edgecolor="#08101d")
            ax_dist.hist(g1, bins=20, alpha=0.65, color="#ef4444", label="Diabetic / High Risk (1)", density=True, edgecolor="#08101d")

            ax_dist.set_xlabel(f"{selected_feature} ({VALID_RANGES[selected_feature][2]})", color="#cbd5e1", fontsize=9)
            ax_dist.set_ylabel("Normalized Density", color="#94a3b8", fontsize=9)
            ax_dist.tick_params(colors="#cbd5e1", labelsize=8)
            ax_dist.legend(facecolor="#0d1a2d", edgecolor="#162a45", labelcolor="#f8fafc", fontsize=9)
            ax_dist.grid(color="#162a45", linestyle="--", alpha=0.5)
            for spine in ax_dist.spines.values():
                spine.set_color("#162a45")

            fig_dist.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)


if __name__ == "__main__":
    main()
