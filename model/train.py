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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_FILE = os.path.join(MODEL_DIR, "diabetes_6feature_model.joblib")

# Raw 8 features present in the original dataset
ALL_RAW_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# 6 retained features after permanently dropping SkinThickness and Insulin
TRAIN_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "BMI",
    "DiabetesPedigreeFunction", "Age"
]
FEATURES = TRAIN_FEATURES
TARGET = "Outcome"

# 0 is impossible for these 3 retained columns, indicating an unmeasured value
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "BMI"]

VALID_RANGES = {
    "Pregnancies":              (0, 20,     "number of pregnancies"),
    "Glucose":                  (40, 300,   "mg/dL, plasma glucose"),
    "BloodPressure":            (30, 140,   "mm Hg, diastolic"),
    "BMI":                      (10.0, 70.0, "kg/m^2"),
    "DiabetesPedigreeFunction": (0.05, 2.50, "genetic risk score"),
    "Age":                      (1, 120,    "years"),
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
    Load dataset, mark 0 as NaN for ZERO_AS_MISSING, and slice to 6 features + target.
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
    Train pipeline with GridSearchCV over decision tree hyperparameters.
    """
    pipeline = build_pipeline()

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
    """Save trained pipeline to disk via joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    joblib.dump(model, file_path)
    # Also save to project root for top-level references
    root_file = os.path.join(os.path.dirname(os.path.abspath(file_path)), "..", "diabetes_6feature_model.joblib")
    if os.path.abspath(file_path) != os.path.abspath(root_file):
        try:
            joblib.dump(model, root_file)
        except Exception:
            pass
    return file_path


def load_model(file_path=MODEL_FILE):
    """Load trained pipeline from disk, checking candidate paths."""
    candidates = [
        file_path,
        os.path.join(os.path.dirname(__file__), "diabetes_6feature_model.joblib"),
        "model/diabetes_6feature_model.joblib",
        "diabetes_6feature_model.joblib",
        "diabetes_tree_model.joblib",
        "diabetes_model.joblib"
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    return None


def predict_risk(model, values):
    """values is a dict {feature: number}. Returns (label, confidence %)."""
    row = pd.DataFrame([values], columns=TRAIN_FEATURES).astype(float)
    row[ZERO_AS_MISSING] = row[ZERO_AS_MISSING].replace(0, np.nan)
    proba = model.predict_proba(row)[0]
    pred = int(np.argmax(proba))
    label = "High Risk" if pred == 1 else "Low Risk"
    return label, proba[pred] * 100


def get_tree_rules_text(model):
    """Extract human-readable decision rules from the tree classifier."""
    tree_clf = model.named_steps.get("tree", model.named_steps.get("classifier"))
    return export_text(tree_clf, feature_names=TRAIN_FEATURES)


def train_and_save():
    """Main training routine for 6-feature pipeline."""
    _, df_clean = load_dataset()
    X = df_clean[TRAIN_FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model, search = train_and_tune(X_train, y_train, use_grid_search=True)
    save_model(model, MODEL_FILE)
    # Also save as diabetes_tree_model.joblib in root for compatibility
    joblib.dump(model, "diabetes_tree_model.joblib")

    print("=== 6-Feature Model Training Complete ===")
    if search:
        print("Best parameters :", search.best_params_)
        print(f"Best CV F1-score: {search.best_score_:.4f}")

    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nTrain Accuracy : {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-Score       : {metrics['f1']:.4f}")
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])
    print("\nFeature Importance:\n", metrics["feature_importances"].round(3))
    return model


if __name__ == "__main__":
    train_and_save()
