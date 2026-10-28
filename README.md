---
title: DiaPredict AI - Clinical Decision Support
emoji: 🩺
colorFrom: cyan
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🩺 DiaPredict AI — Clinical Decision Support System (CDSS)

An evidence-based clinical machine learning application for diabetes risk stratification based on 6 core physiological biomarkers (with unmeasured-test imputation).

---

## 🚀 Instant Deployment Options (Free & Live on the Web)

### Option 1: Hugging Face Spaces (Recommended — 100% Free & Fast)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set a **Space name** (e.g., `diapredict-ai`).
3. Select **Docker** as the Space SDK (Blank template).
4. Set Space hardware to **Free (CPU)** and visibility to **Public**.
5. Push this repository to your Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push --set-upstream space main
   ```
6. Hugging Face will automatically detect the [Dockerfile](file:///Users/apple/antigravity/Dockerfile) and launch your live application URL (`https://huggingface.co/spaces/<your-username>/<your-space-name>`)!

---

### Option 2: Render (Free Web Service)

1. Push this repository to **GitHub**.
2. Go to [dashboard.render.com](https://dashboard.render.com/) and click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Render automatically detects the [render.yaml](file:///Users/apple/antigravity/render.yaml) and [Dockerfile](file:///Users/apple/antigravity/Dockerfile).
5. Select the **Free** instance type and click **Create Web Service**.
6. Your live URL will be active at `https://diapredict-ai-cdss.onrender.com`.

---

### Option 3: Streamlit Community Cloud (Free Native Hosting)

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app** and select your repository.
4. Set **Main file path** to `app.py`.
5. Click **Deploy!**

---

## 🐳 Local Docker Testing

To build and run the container locally:

```bash
# 1. Build the Docker container
docker build -t diapredict-ai .

# 2. Run container on port 7860
docker run -p 7860:7860 diapredict-ai

# 3. Open in your browser:
# http://localhost:7860
```

---

## 🔬 Clinical Feature Architecture (6 Features)

| Biomarker | Clinical Parameter | Physiological Range | Clinical Reference |
| :--- | :--- | :--- | :--- |
| **Glucose** | Fasting Plasma Glucose (FPG) | 40–300 mg/dL | ADA Fasting (<100 mg/dL Normal) |
| **BloodPressure** | Diastolic Blood Pressure | 30–140 mm Hg | AHA (<80 mm Hg Normal) |
| **BMI** | Body Mass Index | 10.0–70.0 kg/m² | WHO (18.5–24.9 kg/m² Normal) |
| **DiabetesPedigreeFunction** | Family Genetic Risk Score | 0.05–2.50 | Genetic Predisposition Score |
| **Age** | Patient Age | 1–120 years | Demographic Stratification |
| **Pregnancies** | Obstetric History / Parity | 0–20 | Gestational Parity Count |

*Note: `SkinThickness` and `Insulin` were permanently dropped due to high clinical missingness rates.*

---

## 🧪 Automated Clinical Test Suite

Run the 8 standardized clinical benchmark archetypes locally:

```bash
# Via pytest
pytest -v test_cases.py

# Direct execution
python test_cases.py
```
