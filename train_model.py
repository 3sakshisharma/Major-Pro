# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

BASE = os.path.dirname(__file__) or "."
DATA_PATH = os.path.join(BASE, "house_data.csv")
MODEL_PATH = os.path.join(BASE, "house_price_model.pkl")

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# Required columns
required = ['City','AreaName','AreaSqft','Bedrooms','PropertyType','Price']
for c in required:
    if c not in df.columns:
        raise Exception(f"Missing column in CSV: {c}")

# Clean & convert
df = df.dropna(subset=required).reset_index(drop=True)
df['AreaSqft'] = pd.to_numeric(df['AreaSqft'], errors='coerce')
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['AreaSqft','Bedrooms','Price']).reset_index(drop=True)

# Features & target
X = df[['City','AreaName','AreaSqft','Bedrooms','PropertyType']]
y = df['Price']

# Preprocessing: one-hot categorical, passthrough numeric
cat_cols = ['City','AreaName','PropertyType']
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("Training model... (may take a while)")
pipeline.fit(X_train, y_train)

# Eval
preds = pipeline.predict(X_test)
print("R2 score:", r2_score(y_test, preds))
print("RMSE   :", np.sqrt(mean_squared_error(y_test, preds)))

# Save model
joblib.dump(pipeline, MODEL_PATH)
print("Saved model as:", MODEL_PATH)