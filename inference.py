from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"status": "API berjalan"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    pred = model.predict(input_df)

    return {
        "prediction": int(pred[0])
    }