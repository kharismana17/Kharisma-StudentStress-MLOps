
import pandas as pd
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv("student_stress.csv")


imputer = SimpleImputer(
    strategy="mean"
)

num_cols = df.select_dtypes(
    include=["number"]
)

df[num_cols.columns] = imputer.fit_transform(
    num_cols
)


encoders = {}

for col in df.select_dtypes(
    include=["object"]
).columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col]
    )

    encoders[col] = encoder


os.makedirs(
    "dataset_preprocessing",
    exist_ok=True
)

os.makedirs(
    "artifacts",
    exist_ok=True
)

df.to_csv(
    "dataset_preprocessing/stress_clean.csv",
    index=False
)


joblib.dump(
    imputer,
    "artifacts/imputer.pkl"
)

joblib.dump(
    encoders,
    "artifacts/encoders.pkl"
)

print("Preprocessing selesai")
print("Dataset:", df.shape)
print("File tersimpan:")
print("- dataset_preprocessing/stress_clean.csv")
print("- artifacts/imputer.pkl")
print("- artifacts/encoders.pkl")
