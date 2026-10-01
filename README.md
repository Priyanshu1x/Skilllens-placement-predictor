# 🎓 AI Placement Assistant

A complete end-to-end machine learning pipeline and interactive web dashboard designed to predict student placement readiness. This project prioritizes safe, explainable AI by incorporating an "Abstention Threshold" to route uncertain, borderline predictions to human Placement Officers.

## 🏗️ Architecture & Design Decisions
1. **Custom NumPy Baseline:** Built a Logistic Regression model entirely from scratch using pure Python and NumPy to demonstrate a deep understanding of gradient descent and cross-entropy loss.
2. **Model Comparison:** Evaluated the custom baseline against industry-standard ensemble models (Random Forest, XGBoost) using Precision, Recall, F1, and PR-AUC.
3. **Abstention Threshold (Human-in-the-Loop):** Tabular human-performance data is inherently noisy. Instead of forcing the model to guess on borderline cases (resulting in high False Positives), the system uses a calibrated `0.45 - 0.55` confidence threshold. Predictions falling in this "gray area" are flagged for manual human review, significantly increasing the automated precision of the system.
4. **Data Drift Detection:** Includes a Kolmogorov-Smirnov (KS) test script to compare live incoming data against the training distribution to warn administrators of statistical drift.
5. **Interactive UI:** A Streamlit dashboard supporting both single-student "what-if" analysis (with feature importance visualization) and batch CSV uploads.

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt