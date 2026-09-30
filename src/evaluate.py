import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

from custom_model import customlogisticregression
logistic_reg_model=joblib.load('data/custom_baseline_model.pkl')
rf_model=joblib.load('data/rf_model.pkl')
xgb_model=joblib.load('data/xgb_model.pkl')

_,_,_,_,x_test,y_test=joblib.load('data/processed_data.pkl')

y_test_binary=np.where(y_test=='Placed',1,0)

predict_proba=logistic_reg_model.predict_proba(x_test)
prediction=logistic_reg_model.predict(x_test)

print("precision score = ", precision_score(y_test_binary,prediction))
print("recall score = ",recall_score(y_test_binary,prediction))
print("f1 score = ",f1_score(y_test_binary,prediction))
print("PR_AUC = ",average_precision_score(y_test_binary,predict_proba))


rf_predict_proba = rf_model.predict_proba(x_test)[:, 1]
rf_prediction = rf_model.predict(x_test)

print("Precision: ", precision_score(y_test_binary, rf_prediction))
print("Recall:    ", recall_score(y_test_binary, rf_prediction))
print("F1 score:  ", f1_score(y_test_binary, rf_prediction))
print("PR-AUC:    ", average_precision_score(y_test_binary, rf_predict_proba))

print("\n--- 3. XGBoost ---")
xgb_predict_proba = xgb_model.predict_proba(x_test)[:, 1]
xgb_prediction = xgb_model.predict(x_test)

print("Precision: ", precision_score(y_test_binary, xgb_prediction))
print("Recall:    ", recall_score(y_test_binary, xgb_prediction))
print("F1 score:  ", f1_score(y_test_binary, xgb_prediction))
print("PR-AUC:    ", average_precision_score(y_test_binary, xgb_predict_proba))