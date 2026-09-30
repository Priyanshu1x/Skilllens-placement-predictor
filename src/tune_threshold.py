import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score
from custom_model import customlogisticregression

if __name__ == "__main__":
    _, _, X_val, y_val, _, _ = joblib.load('data/processed_data.pkl')
    y_val_binary = np.where(y_val == 'Placed', 1, 0)
    
    model = joblib.load('data/custom_baseline_model.pkl')
    probs = model.predict_proba(X_val)
    
    for lower, upper in [(0.50, 0.50), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65)]:
        preds = np.full(probs.shape, -1) 
        preds[probs >= upper] = 1        
        preds[probs <= lower] = 0        
        
        auto_mask = preds != -1
        humans = np.sum(~auto_mask)
        
        prec = precision_score(y_val_binary[auto_mask], preds[auto_mask], zero_division=0)
        rec = recall_score(y_val_binary[auto_mask], preds[auto_mask], zero_division=0)

        print(f"Bounds: {lower:.2f}-{upper:.2f} | Sent to Human: {humans}/{len(probs)} ({(humans/len(probs))*100:.1f}%) | Auto-Precision: {prec:.4f}")