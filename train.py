import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ════════════════════════════════════════════════════════════
ROLL_NO = "2022bcs0002"
NAME    = "Abhishek Akash"
# ════════════════════════════════════════════════════════════

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(f"{ROLL_NO}_experiment")

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("data/housing.csv")

# ── Preprocessing ─────────────────────────────────────────────
# Fill missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Encode ocean_proximity (text → number)
le = LabelEncoder()
df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"])

# All features
features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income", "ocean_proximity"
]
target = "median_house_value"

X = df[features]
y = df[target]

# v1 dataset — use 60% of data (partial dataset)
X = X.sample(frac=0.6, random_state=42)
y = y[X.index]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="Run1_RandomForest_v1_allfeatures"):

    model = RandomForestRegressor(
        n_estimators = 100,
        max_depth    = None,
        random_state = 42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Log parameters
    mlflow.log_param("run_number",        1)
    mlflow.log_param("dataset_version",   "v1")
    mlflow.log_param("dataset_size",      len(X))
    mlflow.log_param("model_type",        "RandomForestRegressor")
    mlflow.log_param("n_estimators",      100)
    mlflow.log_param("max_depth",         "None")
    mlflow.log_param("features_used",     features)
    mlflow.log_param("num_features",      len(features))
    mlflow.log_param("feature_selection", "No")
    mlflow.log_param("name",              NAME)
    mlflow.log_param("roll_no",           ROLL_NO)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run1 | RandomForest | MAE={mae:.2f} | R2={r2:.4f}")

# ── Save model for FastAPI ─────────────────────────────────────
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder too
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ── Save metrics.json ─────────────────────────────────────────
metrics = {
    "name"             : NAME,
    "roll_no"          : ROLL_NO,
    "run_number"       : 1,
    "run_name"         : "Run1_RandomForest_v1_allfeatures",
    "model"            : "RandomForestRegressor",
    "dataset_version"  : "v1",
    "dataset_size"     : len(X),
    "features"         : features,
    "feature_selection": "No",
    "n_estimators"     : 100,
    "max_depth"        : "None",
    "mae"              : round(mae, 2),
    "r2_score"         : round(r2, 4)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. best_model.pkl and metrics.json saved.")