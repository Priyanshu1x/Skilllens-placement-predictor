import numpy as np
import joblib
from scipy.stats import ks_2samp

if __name__ == "__main__":
    X_train, _, _, _, x_test, _ = joblib.load('data/processed_data.pkl')
    
    print("--- Running Data Drift Check (KS Test) ---")
    drift_detected = False
    num_features = X_train.shape[1]

    for i in range(num_features):
        train_col = X_train[:, i]
        test_col = x_test[:, i]
        
        statistic, p_value = ks_2samp(train_col, test_col)
        
        if p_value < 0.05:
            print(f" Warning: Data drift detected in feature column {i} (p-value: {p_value:.4f})")
            drift_detected = True

    
    if not drift_detected:
        print("No data drift detected. The new data matches the training data.")
    else:
        print("Drift found. Model may need retraining or feature investigation.")