#!/usr/bin/env python3
"""
Validation test suite for 6-feature diabetes risk prediction model.
Compatible with direct execution (python test_cases.py) and pytest (pytest test_cases.py).
SkinThickness and Insulin are permanently removed.
"""

import pytest
from diabetes_model import load_model, predict_risk

TEST_CASES = [
    ("TC1", dict(Pregnancies=0, Glucose=85, BloodPressure=66, BMI=22.0, DiabetesPedigreeFunction=0.20, Age=23), "Low Risk"),
    ("TC2", dict(Pregnancies=6, Glucose=180, BloodPressure=85, BMI=38.5, DiabetesPedigreeFunction=0.90, Age=52), "High Risk"),
    ("TC3", dict(Pregnancies=2, Glucose=126, BloodPressure=72, BMI=30.0, DiabetesPedigreeFunction=0.40, Age=35), "High Risk"),
    ("TC4", dict(Pregnancies=1, Glucose=95, BloodPressure=70, BMI=30.0, DiabetesPedigreeFunction=0.30, Age=28), "Low Risk"),
    ("TC5", dict(Pregnancies=1, Glucose=190, BloodPressure=70, BMI=26.0, DiabetesPedigreeFunction=0.35, Age=30), "High Risk"),
    ("TC6", dict(Pregnancies=10, Glucose=150, BloodPressure=90, BMI=42.0, DiabetesPedigreeFunction=1.20, Age=60), "High Risk"),
    ("TC7", dict(Pregnancies=3, Glucose=100, BloodPressure=74, BMI=24.5, DiabetesPedigreeFunction=0.25, Age=29), "Low Risk"),
    ("TC8", dict(Pregnancies=0, Glucose=70, BloodPressure=60, BMI=17.5, DiabetesPedigreeFunction=0.15, Age=21), "Low Risk"),
]


@pytest.fixture(scope="module")
def trained_model():
    model = load_model()
    assert model is not None, "Model file not found. Ensure diabetes_6feature_model.joblib exists."
    return model


@pytest.mark.parametrize("tid, vals, expected", TEST_CASES)
def test_prediction_cases(trained_model, tid, vals, expected):
    label, conf = predict_risk(trained_model, vals)
    assert label == expected, f"Failed on {tid}: expected {expected}, got {label} ({conf:.1f}%)"


def execute_test_suite(model=None):
    """Execute all test cases and return (passed_count, total_count, results_list)."""
    if model is None:
        model = load_model()
    if model is None:
        raise RuntimeError("Model file not found. Please train and save the model first.")

    results = []
    passed = 0
    for tid, vals, expected in TEST_CASES:
        label, conf = predict_risk(model, vals)
        ok = (label == expected)
        passed += ok
        results.append({
            "Test ID": tid,
            "Expected": expected,
            "Predicted": label,
            "Confidence": f"{conf:.1f}%",
            "Result": "PASS" if ok else "FAIL",
            "Biomarkers": vals
        })
    return passed, len(TEST_CASES), results


def run_tests():
    passed, total, results = execute_test_suite()
    print("=" * 65)
    print("  RUNNING 6-FEATURE DIABETES RISK PREDICTION TEST SUITE")
    print("=" * 65)

    for r in results:
        status = r["Result"]
        print(f" {r['Test ID']} | expected: {r['Expected']:<9} | actual: {r['Predicted']:<9} ({r['Confidence']:>6}) | [{status}]")

    print("-" * 65)
    print(f" Summary: Passed {passed}/{total} tests ({passed / total * 100:.0f}%)")
    print("=" * 65)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
