from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

# ── CHANGE THESE ──────────────────────────────────────────
ROLL_NO = "2022bcs0002"
NAME    = "Abhishek Akash"
# ──────────────────────────────────────────────────────────

app = FastAPI()

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Health check endpoint
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "name": NAME,
        "roll_no": ROLL_NO,
        "message": "MLOps Assignment API is running"
    }

# Define input format
class PredictionInput(BaseModel):
    features: list[float]   # e.g. [5.1, 3.5, 1.4, 0.2]

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    class_names = ["setosa", "versicolor", "virginica"]
    return {
        "prediction": prediction,
        "class_name": class_names[prediction],
        "name": NAME,
        "roll_no": ROLL_NO
    }