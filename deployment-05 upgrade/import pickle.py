import pickle

from typing import Any
from fastapi import Body, FastAPI
import uvicorn

app = FastAPI(title="customer-churn-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)



def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)
@app.post("/predict")
def predict(customer: dict[str, Any] = Body(...)) -> dict[str, Any]:
    churn_probability = predict_single(customer)
    return {
        "churn_probability": churn_probability,
        "churn": churn_probability >= 0.5,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
