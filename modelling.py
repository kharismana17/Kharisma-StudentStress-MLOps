
import mlflow
import pandas as pd
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    "dataset_preprocessing/stress_clean.csv"
)

# Pisah fitur dan target
X = df.drop(
    "Stress_Level",
    axis=1
)

y = df["Stress_Level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Set experiment MLflow
mlflow.set_experiment(
    "Kharisma_Nur_AuliaStressPrediction"
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        pred
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example,
    pip_requirements=[
        "scikit-learn",
        "pandas",
        "numpy",
        "mlflow"
    ]
)

    print("Accuracy:", acc)
