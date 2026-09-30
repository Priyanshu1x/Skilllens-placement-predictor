import numpy as np
import joblib

class customlogisticregression:
    def __init__(self,epochs=1000,learning_rate=0.01):
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.weight=None
        self.bias=None

    def sig(self,z):
        z=np.clip(z,-250,250)
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        no_rows,no_feat=X.shape
        self.weight=np.zeros(shape=(no_feat))
        self.bias=0
        for epoch in range (self.epochs):
            linear_func=np.dot(X,self.weight)+self.bias
            y_pred=self.sig(linear_func)

            dw=(1/no_rows)*np.dot(X.T,(y_pred-y))
            db=(1/no_rows)*np.sum(y_pred-y)

            self.weight-=(self.learning_rate*dw)
            self.bias-=(self.learning_rate*db)

    def predict_proba(self,X):
        linear=np.dot(X,self.weight)+self.bias
        return self.sig(linear)

    def predict(self, X, threshold=0.5):
        y_predicted_proba = self.predict_proba(X)
        return (y_predicted_proba >= threshold).astype(int)

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = joblib.load('data/processed_data.pkl')
    y_train_binary = np.where(y_train == 'Placed', 1, 0)
    y_test_binary = np.where(y_test == 'Placed', 1, 0)

    model = customlogisticregression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train_binary)
        
    joblib.dump(model, 'data/custom_baseline_model.pkl')
    print("Custom baseline trained and saved to data/custom_baseline_model.pkl")

        
    

