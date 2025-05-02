from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model import load_model

app = FastAPI()
model = load_model()

class IrisRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: IrisRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}