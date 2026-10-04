# 🎓 AI Placement Assistant

A complete end-to-end machine learning pipeline and interactive web dashboard designed to predict student placement readiness. This project prioritizes safe, explainable AI by incorporating an "Abstention Threshold" to route uncertain, borderline predictions to human Placement Officers.

🚀 **Live Deployment:** [AI Placement Assistant](https://skilllens-placement-predictor.streamlit.app/)  
*Test the custom ML pipeline, run what-if scenarios with real-time explainability, and see the Human-in-the-Loop abstention threshold in action.*

## 🏗️ Architecture & Design Decisions
1. **Custom NumPy Baseline:** Built a Logistic Regression model entirely from scratch using pure Python and NumPy to demonstrate a deep understanding of gradient descent and cross-entropy loss.
2. **Model Comparison:** Evaluated the custom baseline against industry-standard ensemble models (Random Forest, XGBoost) using Precision, Recall, F1, and PR-AUC.
3. **Abstention Threshold (Human-in-the-Loop):** Tabular human-performance data is inherently noisy. Instead of forcing the model to guess on borderline cases (resulting in high False Positives), the system uses a calibrated `0.45 - 0.55` confidence threshold. Predictions falling in this "gray area" are flagged for manual human review, significantly increasing the automated precision of the system.
4. **Data Drift Detection:** Includes a Kolmogorov-Smirnov (KS) test script to compare live incoming data against the training distribution to warn administrators of statistical drift.
5. **Interactive UI:** A Streamlit dashboard supporting both single-student "what-if" analysis (with feature importance visualization) and batch CSV uploads.

### System Architecture Diagram
```mermaid
graph TD
    A[Raw Tabular Data] --> B[Data Preprocessing Pipeline]
    B --> C[Imputation & Scaling]
    C --> D[Feature Engineering]
    D --> E{Model Comparison}
    E --> F[XGBoost / Random Forest]
    E --> G[Custom NumPy Logistic Regression]
    G --> H[Streamlit UI Interface]
    H --> I{Abstention Logic}
    I -- >0.55 --> J[Automated: Placed]
    I -- <0.45 --> K[Automated: Not Placed]
    I -- 0.45 to 0.55 --> L[Manual: Human Review]
