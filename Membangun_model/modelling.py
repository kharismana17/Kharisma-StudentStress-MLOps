import os
import pandas as pd
import mlflow
import dagshub
import joblib


dagshub.init(
    repo_owner='kharismana17',
    repo_name='Kharisma-StudentStress-MLOps',
    mlflow=True
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    BASE_DIR,
    "Eksperimen_SML_Kharisma",
    "dataset_preprocessing",
    "stress_clean.csv"
)

df = pd.read_csv(file_path)


X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("StressPrediction")


with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "model.pkl")
    print("Model berhasil disimpan")

    print("Accuracy:", acc)