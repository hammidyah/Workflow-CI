# -*- coding: utf-8 -*-
"""modelling

Converted from Colab to a local script: removes Colab magic and uses local files.
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset (local file in repo)
df = pd.read_csv('titanic_dataset.csv')

# Basic preprocessing (tolerant to missing columns)
df = df.drop(columns=["alive", "class", "embark_town"], errors='ignore')
if "adult_male" in df.columns:
    df["adult_male"] = df["adult_male"].astype(int)
if "alone" in df.columns:
    df["alone"] = df["alone"].astype(int)

df = pd.get_dummies(
    df,
    columns=[col for col in ["sex", "embarked", "who", "deck"] if col in df.columns],
    drop_first=True
)

X = df.drop("survived", axis=1)
y = df["survived"]

# Convert any boolean columns to int
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MLflow experiment (uses default tracking uri unless you change it)
mlflow.set_experiment("Basic_Model")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    mlflow.log_metric("train_accuracy", float(acc))

    print(f"Model training complete. train_accuracy={acc:.4f}. Check MLflow UI.")

# Note: install packages in terminal (do NOT use `!pip` inside this file).