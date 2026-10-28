import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RANDOM_STATE = 42
DATA_FILE = "diabetes.csv"
DATA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
            "master/pima-indians-diabetes.data.csv")
MODEL_FILE = "model/diabetes_6feature_model.joblib"

# Raw columns from CSV
ALL_RAW_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# 6 features retained after removing SkinThickness and Insulin
TRAIN_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "BMI",
    "DiabetesPedigreeFunction", "Age"
]
FEATURES = TRAIN_FEATURES
TARGET = "Outcome"

# 0 is biologically impossible for these 3 columns (means unmeasured)
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "BMI"]

# Allowed input ranges for the 6 features
VALID_RANGES = {
    "Pregnancies":              (0, 20,       "number of pregnancies"),
    "Glucose":                  (40, 300,     "mg/dL, plasma glucose"),
    "BloodPressure":            (30, 140,     "mm Hg, diastolic"),
    "BMI":                      (10.0, 70.0,  "kg/m^2"),
    "DiabetesPedigreeFunction": (0.05, 2.50,  "genetic risk score"),
    "Age":                      (1, 120,      "years"),
}


def download_data_if_needed(file_path=DATA_FILE, url=DATA_URL):
    """Download the dataset if it does not already exist locally."""
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    return file_path


def load_dataset(file_path=DATA_FILE):
    """
    Load the Pima Indians Diabetes dataset.
    Converts 0s to NaN for ZERO_AS_MISSING and filters to the 6 retained features.
    """
    download_data_if_needed(file_path)
    df_raw = pd.read_csv(file_path, header=None, names=ALL_RAW_FEATURES + [TARGET])
    df_clean = df_raw.copy()
    df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)
    df_clean = df_clean[TRAIN_FEATURES + [TARGET]].copy()
    return df_raw, df_clean


def build_pipeline(classifier_params=None):
    """
    Construct a scikit-learn Pipeline with median imputation, standardization,
    and a balanced DecisionTreeClassifier.
    """
    if classifier_params is None:
        classifier_params = {"random_state": RANDOM_STATE, "class_weight": "balanced"}
    else:
        if "random_state" not in classifier_params:
            classifier_params["random_state"] = RANDOM_STATE
        if "class_weight" not in classifier_params:
            classifier_params["class_weight"] = "balanced"

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(**classifier_params))
    ])
    return pipeline


def train_and_tune(X_train, y_train, use_grid_search=True, param_grid=None, cv=5):
    """
    Train pipeline with either GridSearchCV or default hyperparameters.
    """
    pipeline = build_pipeline({"class_weight": "balanced"})

    if not use_grid_search:
        pipeline.fit(X_train, y_train)
        return pipeline, None

    if param_grid is None:
        param_grid = {
            "tree__criterion": ["gini", "entropy"],
            "tree__max_depth": [2, 3, 4, 5, 6, 8, None],
            "tree__min_samples_leaf": [1, 5, 10, 20],
            "tree__min_samples_split": [2, 10, 20]
        }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained pipeline on test set and extract metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    tree_step = model.named_steps.get("tree", model.named_steps.get("classifier"))
    feature_importances = pd.Series(
        tree_step.feature_importances_,
        index=TRAIN_FEATURES
    ).sort_values(ascending=False)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": feature_importances,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def save_model(model, file_path=MODEL_FILE):
    """Save pipeline to disk via joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    joblib.dump(model, file_path)
    # Also save to root fallback paths for seamless loading
    fallback_paths = ["diabetes_6feature_model.joblib", "diabetes_tree_model.joblib"]
    for fp in fallback_paths:
        try:
            joblib.dump(model, fp)
        except Exception:
            pass
    return file_path


def load_model(file_path=MODEL_FILE):
    """Load trained pipeline from disk."""
    candidates = [
        file_path,
        "model/diabetes_6feature_model.joblib",
        "diabetes_6feature_model.joblib",
        "diabetes_tree_model.joblib",
        "diabetes_model.joblib"
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    return None


def predict_single(model, feature_values):
    """
    Predict diabetes outcome and probability for a single set of feature inputs (dict or list).
    """
    if isinstance(feature_values, dict):
        row = [feature_values.get(f, np.nan) for f in TRAIN_FEATURES]
    else:
        row = list(feature_values)

    row_dict = dict(zip(TRAIN_FEATURES, row))
    for col in ZERO_AS_MISSING:
        if row_dict.get(col) == 0:
            row_dict[col] = np.nan

    input_df = pd.DataFrame([row_dict])[TRAIN_FEATURES]

    pred_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    prob_negative = float(probabilities[0])
    prob_positive = float(probabilities[1])

    return {
        "prediction": pred_class,
        "label": "Diabetic (High Risk)" if pred_class == 1 else "Non-Diabetic (Low Risk)",
        "probability_diabetic": prob_positive,
        "probability_nondiabetic": prob_negative,
        "confidence": prob_positive if pred_class == 1 else prob_negative
    }


def predict_risk(model, values):
    """values is a dict {feature: number}. Returns (label, confidence %)."""
    row = pd.DataFrame([values], columns=TRAIN_FEATURES).astype(float)
    row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
    proba = model.predict_proba(row)[0]
    pred = int(np.argmax(proba))
    label = "High Risk" if pred == 1 else "Low Risk"
    return label, proba[pred] * 100


def read_value(name):
    lo, hi, desc = VALID_RANGES[name]
    while True:
        raw = input(f"  {name} ({desc}) [{lo}-{hi}]: ").strip()
        try:
            val = float(raw)
        except ValueError:
            print("   ! Please enter a number.")
            continue
        if lo <= val <= hi:
            return val
        print(f"   ! Out of range. Enter a value between {lo} and {hi}.")


def cli(model):
    print("=" * 55)
    print("  DIABETES RISK PREDICTOR (6 Features, Educational Use)")
    print("=" * 55)
    while True:
        print("\nEnter patient details:")
        values = {f: read_value(f) for f in TRAIN_FEATURES}
        label, conf = predict_risk(model, values)
        print("\n" + "-" * 40)
        print(f"  Result     : {label.upper()}")
        print(f"  Confidence : {conf:.1f}%")
        print("-" * 40)
        print("  Note: screening aid only, not a medical diagnosis.")
        if input("\nPredict another patient? (y/n): ").strip().lower() != "y":
            print("Goodbye!")
            break


def get_tree_rules_text(model):
    """Extract human-readable decision rules from the tree classifier."""
    tree_clf = model.named_steps.get("tree", model.named_steps.get("classifier"))
    return export_text(tree_clf, feature_names=TRAIN_FEATURES)


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    df_raw, df = load_dataset()

    print("Shape (6 Features + Target):", df.shape)
    print(df.head())
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    print("\nMissing values per column:")
    print(df.isna().sum())

    X = df[TRAIN_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    print("\nTrain:", X_train.shape, "| Test:", X_test.shape)

    pipe = build_pipeline()

    param_grid = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [2, 3, 4, 5, 6, 8, None],
        "tree__min_samples_leaf": [1, 5, 10, 20],
        "tree__min_samples_split": [2, 10, 20],
    }

    search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)

    model = search.best_estimator_
    save_model(model, MODEL_FILE)

    print("Best parameters :", search.best_params_)
    print(f"Best CV F1-score: {search.best_score_:.4f}")

    y_pred = model.predict(X_test)

    print(f"\nTrain Accuracy : {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score       : {f1_score(y_test, y_pred):.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("                 Low Risk   High Risk")
    print(f"Actual Low Risk  {tn:^8d}   {fp:^9d}   (TN, FP)")
    print(f"Actual High Risk {fn:^8d}   {tp:^9d}   (FN, TP)")

    tree = model.named_steps["tree"]
    importance = pd.Series(tree.feature_importances_, index=TRAIN_FEATURES)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False).round(3))

    # Quick check with one patient (6 features)
    print("\nSample Patient Risk Assessment:")
    sample_patient = dict(Pregnancies=6, Glucose=180, BloodPressure=85,
                          BMI=38.5, DiabetesPedigreeFunction=0.9, Age=52)
    print(predict_risk(model, sample_patient))

    # Interactive CLI
    cli(model)
