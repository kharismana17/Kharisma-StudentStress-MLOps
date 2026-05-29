
import mlflow.pyfunc
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

model = mlflow.pyfunc.load_model(
    "/content/mlruns/159810739198386782/d80e3bd3d1e94e99bdf76f5e1b492345/artifacts/model"
)

@app.get("/")
def home():
    return {"status":"API berjalan"}

@app.post("/predict")
def predict(data: dict):

    input_df = pd.DataFrame([data])

    pred = model.predict(input_df)

    return {
        "prediction": int(pred[0])
    }
