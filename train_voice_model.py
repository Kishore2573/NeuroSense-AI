import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print("Loading voice dataset...")

data = pd.read_csv("datasets/voice_dataset.csv")

# Remove name column
if "name" in data.columns:
    data = data.drop(columns=["name"])

X = data.drop("status", axis=1)
y = data["status"]

print("Dataset shape:", X.shape)

# Train split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Voice model trained successfully")