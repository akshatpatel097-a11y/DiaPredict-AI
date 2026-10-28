#!/usr/bin/env python3
"""
Interactive CLI for Diabetes Risk Prediction using DecisionTreeClassifier (6 Features).
Uses validated input ranges and pipeline imputation for unmeasured features.
"""

import argparse
import sys
from sklearn.model_selection import train_test_split

from diabetes_model import (
    FEATURES,
    RANDOM_STATE,
    TARGET,
    VALID_RANGES,
    ZERO_AS_MISSING,
    evaluate_model,
    get_tree_rules_text,
    load_dataset,
    load_model,
    predict_single,
    save_model,
    train_and_tune,
)


def prompt_float(prompt_text, min_val, max_val, default=None):
    """Prompt user for a float value within [min_val, max_val]."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        try:
            user_input = input(f"{prompt_text}{default_str}: ").strip()
            if not user_input and default is not None:
                return float(default)
            val = float(user_input)
            if min_val <= val <= max_val:
                return val
            print(f"  --> Value must be between {min_val} and {max_val}. Please re-enter.")
        except ValueError:
            print("  --> Invalid input. Please enter a numerical value.")


def ensure_model(retrain=False):
    """Ensure a trained 6-feature model exists, training and saving one if necessary."""
    model = load_model()
    if model is not None and not retrain:
        return model

    print("Training new 6-Feature Decision Tree pipeline with GridSearchCV...")
    _, df_clean = load_dataset()
    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model, grid = train_and_tune(X_train, y_train, use_grid_search=True)
    save_model(model)

    print("\n=== Model Training Completed ===")
    if grid:
        print(f"Best CV Score: {grid.best_score_:.4f}")
        print(f"Best Hyperparameters: {grid.best_params_}")

    metrics = evaluate_model(model, X_test, y_test)
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    return model


def run_interactive_prediction(model):
    """Prompt user interactively for each of the 6 features and output prediction."""
    print("\n" + "=" * 60)
    print(" PIMA INDIANS DIABETES RISK ASSESSMENT CLI (6 Features)")
    print(" Enter patient values below (press Enter for default/example):")
    print("=" * 60)

    sample_defaults = {
        "Pregnancies": 2,
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 33
    }

    patient_data = {}
    for feature in FEATURES:
        min_v, max_v, desc = VALID_RANGES[feature]
        prompt_label = f"• {feature:<24} ({desc}, range {min_v}-{max_v})"
        patient_data[feature] = prompt_float(
            prompt_label,
            min_val=min_v,
            max_val=max_v,
            default=sample_defaults.get(feature)
        )

    result = predict_single(model, patient_data)

    print("\n" + "=" * 60)
    print(" ASSESSMENT RESULTS")
    print("=" * 60)
    print(f" Prediction             : {result['label']}")
    print(f" Diabetes Probability   : {result['probability_diabetic'] * 100:.1f}%")
    print(f" Non-Diabetic Probability: {result['probability_nondiabetic'] * 100:.1f}%")
    print(f" Model Confidence       : {result['confidence'] * 100:.1f}%")

    # Biomarker contextual flags
    print("\nKey Biomarker Flags:")
    if patient_data["Glucose"] >= 126:
        print(" [!] High Glucose (≥126 mg/dL indicates diabetic range fasting glucose)")
    elif patient_data["Glucose"] >= 100:
        print(" [!] Impaired Glucose (100-125 mg/dL indicates prediabetes range)")
    else:
        print(" [✓] Normal Fasting Glucose (<100 mg/dL)")

    if patient_data["BMI"] >= 30:
        print(" [!] Elevated BMI (≥30 kg/m² indicates obesity)")
    elif patient_data["BMI"] >= 25:
        print(" [!] Overweight BMI (25-29.9 kg/m²)")
    else:
        print(" [✓] Normal BMI (<25 kg/m²)")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pima Indians Diabetes Decision Tree CLI (6 Features)")
    parser.add_argument("--train", action="store_true", help="Force model retraining and print metrics")
    parser.add_argument("--rules", action="store_true", help="Display text rules of the decision tree")
    args = parser.parse_args()

    model = ensure_model(retrain=args.train)

    if args.rules:
        print("\n=== DECISION TREE RULES ===")
        print(get_tree_rules_text(model))

    if not args.rules and not args.train:
        from diabetes_model import cli as run_cli
        run_cli(model)


if __name__ == "__main__":
    main()
