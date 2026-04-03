import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
import pickle

# ── USER DETAILS ──────────────────────────────────────────
ROLL_NO = "2022bcs0002"
NAME    = "Abhishek Akash"
# ──────────────────────────────────────────────────────────

# Set MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment
mlflow.set_experiment(f"{ROLL_NO}_experiment")

# Load dataset
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

features = list(X.columns)
X_sel = X[features]

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="Run1_RandomForest_v1_allfeatures"):

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # ── Log parameters ─────────────────────────────────────
    mlflow.log_param("run_number", 1)
    mlflow.log_param("dataset_version", "v1")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("features_used", features)
    mlflow.log_param("num_features", len(features))
    mlflow.log_param("name", NAME)
    mlflow.log_param("roll_no", ROLL_NO)

    # ── Log metrics ────────────────────────────────────────
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # ── Log model ──────────────────────────────────────────
    mlflow.sklearn.log_model(model, "model")

    print(f"Run1 | RandomForest | accuracy={accuracy:.4f} | f1={f1:.4f}")

# ── Save model locally ─────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ── Save metrics.json ──────────────────────────────────────
metrics = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "run": "Run1",
    "model": "RandomForestClassifier",
    "dataset_version": "v1",
    "features": features,
    "accuracy": round(accuracy, 4),
    "f1_score": round(f1, 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Done. metrics.json and model saved.")