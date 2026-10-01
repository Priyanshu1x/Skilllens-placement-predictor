import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# You must import your custom model class!
from custom_model import customlogisticregression

if __name__ == "__main__":
    # 1. Load the Test Data (This is the final, frozen test set)
    _, _, _, _, X_test, y_test = joblib.load('data/processed_data.pkl')
    
    # Convert 'Placed'/'Not Placed' string labels to 1s and 0s
    y_test_binary = np.where(y_test == 'Placed', 1, 0)
    
    # 2. Load your winning Custom Model
    model = joblib.load('data/custom_baseline_model.pkl')
    
    # 3. Get predictions and probabilities
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # --- RUBRIC REQUIREMENT: ROC-AUC ---
    roc_auc = roc_auc_score(y_test_binary, probs)
    print(f"\nFINAL TEST SET METRICS:")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # --- RUBRIC REQUIREMENT: CONFUSION MATRIX ---
    print("\nConfusion Matrix (Raw Output):")
    cm = confusion_matrix(y_test_binary, preds)
    print(cm)
    
    # Optional: Pop open a nice visual plot of the matrix
    print("\nOpening visual plot of Confusion Matrix...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
    disp.plot(cmap='Blues')
    plt.title("Final Test Set Confusion Matrix")
    plt.show()