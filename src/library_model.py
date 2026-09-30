import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

if __name__=="__main__":
    X_train, y_train,_, _, _, _=joblib.load('data/processed_data.pkl')
    y_train_binary=np.where(y_train=='Placed',1,0)

    xgb_model=xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        random_state=42,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train,y_train_binary)
    joblib.dump(xgb_model, 'data/xgb_model.pkl')


    rf_model=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)

    rf_model.fit(X_train,y_train_binary)
    joblib.dump(rf_model,'data/rf_model.pkl')