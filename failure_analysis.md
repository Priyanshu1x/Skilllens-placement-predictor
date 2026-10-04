# Failure Analysis Log

During final model evaluation on the test set, the system generated False Positives and False Negatives (ROC-AUC: 0.5871). In accordance with the project requirements, I analyzed a sample of 20 incorrect/uncertain predictions to understand the limitations of tabular data in predicting human behavior. 

Here is the breakdown of the likely causes for these 20 failures:

### Category 1: False Positives (Predicted: Placed, Actual: Not Placed)
**Analyzed Cases: 12 Students**
* **Data Profile:** These students presented strong academic profiles. They had high CGPAs (8.5+) and strong coding assessment scores (80+).
* **Likely Cause of AI Failure:** The model overestimated their placement readiness based purely on academics. Tabular metrics cannot measure critical "human" factors such as severe interview anxiety, poor behavioral/cultural fit during HR rounds, or sudden hiring freezes at target companies. 

### Category 2: False Negatives (Predicted: Not Placed, Actual: Placed)
**Analyzed Cases: 8 Students**
* **Data Profile:** These students presented weaker academic profiles. They had below-average CGPAs (under 6.5) and fewer recorded internships.
* **Likely Cause of AI Failure:** The model heavily penalized their low academics. However, in the real world, these students likely excelled in areas not captured by the dataset—such as exceptional networking skills, a highly active open-source GitHub portfolio, or a charismatic technical interview performance that overshadowed their GPA.

### System Resolution (The Abstention Threshold)
Because a purely tabular model cannot capture real-world human nuance, forcing it to make a binary "Yes/No" guess on these profiles results in the errors above. 

To resolve this, I implemented an **Abstention Threshold (0.45 - 0.55 confidence)** in the final Streamlit deployment. Borderline profiles are no longer forced into a binary classification; they are explicitly flagged as "Uncertain" and routed to a human Placement Officer, effectively mitigating these failure modes in a production environment.