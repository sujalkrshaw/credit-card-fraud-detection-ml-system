from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model safely
bundle = joblib.load("models/fraud_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# Input schema
class Transaction(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(tx: Transaction):
    try:
        X = np.array(tx.data)

        # Ensure 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # FIX: auto match feature size
        required_features = scaler.n_features_in_

        if X.shape[1] < required_features:
            padding = required_features - X.shape[1]
            X = np.hstack([X, np.zeros((X.shape[0], padding))])
        elif X.shape[1] > required_features:
            X = X[:, :required_features]

        X = scaler.transform(X)

        probs = model.predict_proba(X)[:, 1]
        preds = ["FRAUD" if p > 0.8 else "NORMAL" for p in probs]

        return [
            {"probability": float(p), "prediction": pred}
            for p, pred in zip(probs, preds)
        ]

    except Exception as e:
        return {"error": str(e)}