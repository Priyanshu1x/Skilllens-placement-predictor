import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from custom_model import customlogisticregression

if __name__ == "__main__":
    _, _, _, _, X_test, y_test = joblib.load('data/processed_data.pkl')
    
    y_test_binary = np.where(y_test == 'Placed', 1, 0)
    
    model = joblib.load('data/custom_baseline_model.pkl')
    
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test_binary, probs)
    print(f"\nFINAL TEST SET METRICS:")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix (Raw Output):")
    cm = confusion_matrix(y_test_binary, preds)
    print(cm)
    
    print("\nOpening visual plot of Confusion Matrix...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
    disp.plot(cmap='Blues')
    plt.title("Final Test Set Confusion Matrix")
    plt.show()