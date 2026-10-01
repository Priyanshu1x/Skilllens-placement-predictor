---

Next, create a file named **`AI_USAGE.md`**. The rubric strictly requires this. We will be completely honest because using AI to scaffold code and test ideas is exactly what Senior Engineers do. Paste this in:

```markdown
# AI Tool Usage & Verification Log

In accordance with project guidelines, this document outlines how AI assistants were utilized and verified during the development of this pipeline.

### 1. Scaffolding & Boilerplate
* **Usage:** AI was used to generate the boilerplate code for the Streamlit dashboard (`app.py`) and the threshold tuning loop (`tune_thresholds.py`).
* **Verification:** I manually reviewed the UI components, adjusted the sliders to fit a standard `-3.0` to `3.0` standardized scale, and verified the business logic routing for the 0.45-0.55 confidence bounds.

### 2. Core Logic (Independent Work)
* The custom Logistic Regression baseline (forward propagation, backpropagation, and weight updates) was built and debugged independently using pure NumPy.
* The `data_drift.py` script utilizing the Kolmogorov-Smirnov test was written and executed manually to ensure a deep understanding of production monitoring. 
* Feature engineering and data preprocessing pipelines were developed independently.

### 3. Debugging & Conceptual Validation
* **Usage:** Used AI to interpret the results of the final Confusion Matrix and ROC-AUC scores.
* **Verification:** Validated the AI's explanation against the project requirements, specifically tying the high False Positive rate back to the necessity of the Abstention Threshold.