import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ════════════════════════════════════════════════════════════
ROLL_NO = "2022bcs0002"
NAME    = "Abhishek Akash"
# ════════════════════════════════════════════════════════════

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(f"{ROLL_NO}_experiment")

# ── Load dataset ─────────────────────────────────────────────
iris     = load_iris(as_frame=True)
X, y     = iris.data, iris.target
features = list(X.columns)   # all 4 features
X_sel    = X[features]

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42
)

# ── Train ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="Run1_RandomForest_v1_allfeatures"):

    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = None,
        random_state = 42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1       = f1_score(y_test, preds, average="weighted")

    # Log parameters
    mlflow.log_param("run_number",        1)
    mlflow.log_param("dataset_version",   "v1")
    mlflow.log_param("model_type",        "RandomForestClassifier")
    mlflow.log_param("n_estimators",      100)
    mlflow.log_param("max_depth",         "None")
    mlflow.log_param("features_used",     features)
    mlflow.log_param("num_features",      len(features))
    mlflow.log_param("feature_selection", "No")
    mlflow.log_param("name",              NAME)
    mlflow.log_param("roll_no",           ROLL_NO)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run1 | RandomForest | accuracy={accuracy:.4f} | f1={f1:.4f}")

# ── Save model for FastAPI ─────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ── Save metrics.json ─────────────────────────────────────────
metrics = {
    "name"             : NAME,
    "roll_no"          : ROLL_NO,
    "run_number"       : 1,
    "run_name"         : "Run1_RandomForest_v1_allfeatures",
    "model"            : "RandomForestClassifier",
    "dataset_version"  : "v1",
    "features"         : features,
    "feature_selection": "No",
    "n_estimators"     : 100,
    "max_depth"        : "None",
    "accuracy"         : round(accuracy, 4),
    "f1_score"         : round(f1, 4)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. best_model.pkl and metrics.json saved.")