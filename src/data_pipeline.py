import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


data = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\noisy_student_placement_data_v4.csv")
df = data.drop(columns=['salary_package_lpa', 'student_id'], errors='ignore')


df = df.sort_values('Application_date').reset_index(drop=True)


n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]


X_train = train_df.drop(columns=['placement_status', 'Application_date'])
y_train = train_df['placement_status'].values

X_val = val_df.drop(columns=['placement_status', 'Application_date'])
y_val = val_df['placement_status'].values

X_test = test_df.drop(columns=['placement_status', 'Application_date'])
y_test = test_df['placement_status'].values


cat_col = X_train.select_dtypes(include=['object']).columns.tolist()
num_col = X_train.select_dtypes(exclude=['object']).columns.tolist()


cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipeline, cat_col),
    ('num', num_pipeline, num_col)
])


X_train_clean = preprocessor.fit_transform(X_train)
X_val_clean = preprocessor.transform(X_val)
X_test_clean = preprocessor.transform(X_test)


joblib.dump(preprocessor, 'data/preprocessor.pkl')
joblib.dump((X_train_clean, y_train, X_val_clean, y_val, X_test_clean, y_test), 'data/processed_data.pkl')

print("Data pipeline completed successfully!")