import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Add project root to path for imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib

from diabetes_model import (
    FEATURES,
    VALID_RANGES,
    ZERO_AS_MISSING,
    evaluate_model,
    load_dataset,
    load_model,
    predict_single,
)
from test_cases import execute_test_suite

# Cache loaded model in serverless runtime
_CACHED_MODEL = None


def get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        root_dir = Path(__file__).resolve().parent.parent
        api_dir = Path(__file__).resolve().parent
        possible_paths = [
            api_dir / "diabetes_6feature_model.joblib",
            root_dir / "model" / "diabetes_6feature_model.joblib",
            root_dir / "diabetes_6feature_model.joblib",
            root_dir / "diabetes_tree_model.joblib",
            root_dir / "diabetes_model.joblib",
            Path("diabetes_6feature_model.joblib"),
            Path("model/diabetes_6feature_model.joblib"),
        ]
        for p in possible_paths:
            if p.exists():
                try:
                    _CACHED_MODEL = joblib.load(str(p))
                    break
                except Exception:
                    pass
        if _CACHED_MODEL is None:
            _CACHED_MODEL = load_model()
    return _CACHED_MODEL


def classify_biomarkers(vals):
    p_glucose = float(vals.get("Glucose", 0))
    p_bmi = float(vals.get("BMI", 0))
    p_bp = float(vals.get("BloodPressure", 0))
    p_dpf = float(vals.get("DiabetesPedigreeFunction", 0))
    p_age = float(vals.get("Age", 0))
    p_preg = float(vals.get("Pregnancies", 0))

    breakdown = []
    # Glucose
    if p_glucose >= 126:
        g_status, g_tag = "High Risk (Diabetic Range)", "ref-danger"
    elif p_glucose >= 100:
        g_status, g_tag = "Elevated (Prediabetic)", "ref-warning"
    else:
        g_status, g_tag = "Normal (<100 mg/dL)", "ref-normal"
    breakdown.append({"name": "Glucose (FPG)", "val": f"{p_glucose:.0f} mg/dL", "ref": "<100 mg/dL (ADA)", "status": g_status, "tag": g_tag})

    # BMI
    if p_bmi >= 30:
        b_status, b_tag = "High Risk (Obese Class I+)", "ref-danger"
    elif p_bmi >= 25:
        b_status, b_tag = "Overweight (25–29.9)", "ref-warning"
    else:
        b_status, b_tag = "Normal (18.5–24.9)", "ref-normal"
    breakdown.append({"name": "Body Mass Index (BMI)", "val": f"{p_bmi:.1f} kg/m²", "ref": "18.5–24.9 kg/m² (WHO)", "status": b_status, "tag": b_tag})

    # Blood Pressure
    if p_bp >= 90:
        bp_status, bp_tag = "High (Hypertension Stage 2)", "ref-danger"
    elif p_bp >= 80:
        bp_status, bp_tag = "Stage 1 Elevated", "ref-warning"
    else:
        bp_status, bp_tag = "Normal (<80 mm Hg)", "ref-normal"
    breakdown.append({"name": "Diastolic BP", "val": f"{p_bp:.0f} mm Hg", "ref": "<80 mm Hg (AHA)", "status": bp_status, "tag": bp_tag})

    # Pedigree Function
    if p_dpf >= 0.8:
        dpf_status, dpf_tag = "Strong Genetic Predisposition", "ref-danger"
    elif p_dpf >= 0.4:
        dpf_status, dpf_tag = "Moderate Genetic Predisposition", "ref-warning"
    else:
        dpf_status, dpf_tag = "Low Genetic Risk", "ref-normal"
    breakdown.append({"name": "Pedigree Score (DPF)", "val": f"{p_dpf:.2f}", "ref": "<0.40 Normal", "status": dpf_status, "tag": dpf_tag})

    # Age
    if p_age >= 50:
        age_status, age_tag = "Higher Age Cohort (≥50)", "ref-warning"
    else:
        age_status, age_tag = "Standard Age Cohort", "ref-normal"
    breakdown.append({"name": "Patient Age", "val": f"{p_age:.0f} yrs", "ref": "Age Stratification", "status": age_status, "tag": age_tag})

    # Pregnancies
    if p_preg >= 6:
        preg_status, preg_tag = "High Parity (≥6)", "ref-warning"
    else:
        preg_status, preg_tag = "Standard Parity", "ref-normal"
    breakdown.append({"name": "Pregnancies / Parity", "val": f"{p_preg:.0f}", "ref": "Obstetric History", "status": preg_status, "tag": preg_tag})

    return breakdown


class handler(BaseHTTPRequestHandler):
    def send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, content, status_code=200):
        body = content.encode("utf-8") if isinstance(content, str) else content
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        model = get_model()

        # If root or index is requested, serve the frontend HTML application
        if path in ("", "/", "/index", "/index.html", "/app") or not path.startswith("/api"):
            html_candidates = [
                Path(__file__).resolve().parent / "index.html",
                Path(__file__).resolve().parent.parent / "public" / "index.html",
                Path(__file__).resolve().parent.parent / "index.html",
                Path("public/index.html"),
                Path("index.html"),
            ]
            for hp in html_candidates:
                if hp.exists():
                    try:
                        with open(hp, "r", encoding="utf-8") as f:
                            self.send_html(f.read())
                            return
                    except Exception:
                        pass

        if path in ("/api/test", "/api/run-tests") or path.endswith("/test") or path.endswith("/run-tests"):
            try:
                passed, total, results = execute_test_suite(model)
                self.send_json({
                    "status": "success",
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": round((passed / total) * 100, 1),
                    "results": results
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, 500)
            return

        if path in ("/api/health", "/api/status", "/api", "/health") or path.endswith("/health") or path.endswith("/status"):
            self.send_json({
                "status": "healthy",
                "service": "DiaPredict AI CDSS Vercel Serverless API",
                "model_loaded": model is not None,
                "features": FEATURES
            })
            return

        self.send_json({"status": "error", "message": f"Endpoint not found: {path}"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        model = get_model()

        if model is None:
            self.send_json({"status": "error", "message": "Model not loaded"}, 500)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_json({"status": "error", "message": "Invalid JSON body"}, 400)
            return

        if path in ("/api/predict", "/api/assess") or path.endswith("/predict") or path.endswith("/assess"):
            try:
                # Extract and cast the 6 features
                vals = {
                    "Pregnancies": float(body.get("Pregnancies", 0)),
                    "Glucose": float(body.get("Glucose", 120)),
                    "BloodPressure": float(body.get("BloodPressure", 70)),
                    "BMI": float(body.get("BMI", 25.0)),
                    "DiabetesPedigreeFunction": float(body.get("DiabetesPedigreeFunction", 0.5)),
                    "Age": float(body.get("Age", 30))
                }

                pred = predict_single(model, vals)
                is_diabetic = (pred["prediction"] == 1)
                prob_diabetic = float(pred["probability_diabetic"])
                prob_nondiabetic = float(pred["probability_nondiabetic"])
                confidence = float(pred["confidence"])

                breakdown = classify_biomarkers(vals)

                self.send_json({
                    "status": "success",
                    "prediction": 1 if is_diabetic else 0,
                    "label": "High Risk" if is_diabetic else "Low Risk",
                    "triage_title": "TRIAGE LEVEL: HIGH RISK" if is_diabetic else "TRIAGE LEVEL: LOW RISK",
                    "probability_diabetic": round(prob_diabetic * 100, 1),
                    "probability_nondiabetic": round(prob_nondiabetic * 100, 1),
                    "confidence": round(confidence, 1),
                    "is_diabetic": is_diabetic,
                    "breakdown": breakdown,
                    "biomarkers": vals
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, 500)
            return

        if path in ("/api/test", "/api/run-tests") or path.endswith("/test") or path.endswith("/run-tests"):
            try:
                passed, total, results = execute_test_suite(model)
                self.send_json({
                    "status": "success",
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_rate": round((passed / total) * 100, 1),
                    "results": results
                })
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, 500)
            return

        self.send_json({"status": "error", "message": f"Endpoint not found: {path}"}, 404)
