import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ════════════════════════════════════════════════════════════
ROLL_NO = "2022bcs0002"
NAME    = "Abhishek Akash"
# ════════════════════════════════════════════════════════════

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(f"{ROLL_NO}_experiment")

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("housing.csv")

# ── Preprocessing ─────────────────────────────────────────────
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

le = LabelEncoder()
df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"])

# Feature selection — only most important features
features = [
    "median_income",
    "housing_median_age",
    "latitude",
    "longitude"
]
target = "median_house_value"

X = df[features]
y = df[target]

# v2 dataset — full dataset
# SVR is slow on large data — use 30% sample for speed
X = X.sample(frac=0.3, random_state=42)
y = y[X.index]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVR needs scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="Run4_SVR_v2_featureselection"):

    model = SVR(
        kernel  = "rbf",
        C       = 100,
        epsilon = 0.1,
        gamma   = "scale"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Log parameters
    mlflow.log_param("run_number",        4)
    mlflow.log_param("dataset_version",   "v2")
    mlflow.log_param("dataset_size",      len(X))
    mlflow.log_param("model_type",        "SVR")
    mlflow.log_param("kernel",            "rbf")
    mlflow.log_param("C",                 100)
    mlflow.log_param("epsilon",           0.1)
    mlflow.log_param("gamma",             "scale")
    mlflow.log_param("scaling",           "StandardScaler")
    mlflow.log_param("features_used",     features)
    mlflow.log_param("num_features",      len(features))
    mlflow.log_param("feature_selection", "Yes - top 4 features only")
    mlflow.log_param("dropped_features",  "total_rooms, total_bedrooms, population, households, ocean_proximity")
    mlflow.log_param("name",              NAME)
    mlflow.log_param("roll_no",           ROLL_NO)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run4 | SVR | MAE={mae:.2f} | R2={r2:.4f}")

# ── Save model for FastAPI ─────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ── Save metrics.json ─────────────────────────────────────────
metrics = {
    "name"             : NAME,
    "roll_no"          : ROLL_NO,
    "run_number"       : 4,
    "run_name"         : "Run4_SVR_v2_featureselection",
    "model"            : "SVR",
    "dataset_version"  : "v2",
    "dataset_size"     : len(X),
    "features"         : features,
    "feature_selection": "Yes - top 4 features only",
    "dropped_features" : "total_rooms, total_bedrooms, population, households, ocean_proximity",
    "kernel"           : "rbf",
    "C"                : 100,
    "epsilon"          : 0.1,
    "mae"              : round(mae, 2),
    "r2_score"         : round(r2, 4)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. best_model.pkl and metrics.json saved.")