import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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

# All features
features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income", "ocean_proximity"
]
target = "median_house_value"

X = df[features]
y = df[target]

# v2 dataset — full dataset
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN needs scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────
with mlflow.start_run(run_name="Run3_KNN_v2_allfeatures"):

    model = KNeighborsRegressor(
        n_neighbors = 5,
        weights     = "distance",
        metric      = "euclidean"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Log parameters
    mlflow.log_param("run_number",        3)
    mlflow.log_param("dataset_version",   "v2")
    mlflow.log_param("dataset_size",      len(X))
    mlflow.log_param("model_type",        "KNeighborsRegressor")
    mlflow.log_param("n_neighbors",       5)
    mlflow.log_param("weights",           "distance")
    mlflow.log_param("metric",            "euclidean")
    mlflow.log_param("scaling",           "StandardScaler")
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

    print(f"Run3 | KNN | MAE={mae:.2f} | R2={r2:.4f}")

# ── Save model + scaler for FastAPI ───────────────────────────
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
    "run_number"       : 3,
    "run_name"         : "Run3_KNN_v2_allfeatures",
    "model"            : "KNeighborsRegressor",
    "dataset_version"  : "v2",
    "dataset_size"     : len(X),
    "features"         : features,
    "feature_selection": "No",
    "n_neighbors"      : 5,
    "weights"          : "distance",
    "mae"              : round(mae, 2),
    "r2_score"         : round(r2, 4)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done. best_model.pkl and metrics.json saved.")