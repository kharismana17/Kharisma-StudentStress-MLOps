import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

input_path = os.path.join(
    BASE_DIR,
    "dataset_raw",
    "student_stress.csv"
)

output_dir = os.path.join(
    BASE_DIR,
    "dataset_preprocessing"
)

os.makedirs(output_dir, exist_ok=True)

print("BASE_DIR =", BASE_DIR)
print("INPUT_PATH =", input_path)
print("EXISTS =", os.path.exists(input_path))

df = pd.read_csv(input_path)

num_cols = df.select_dtypes(include="number")

imputer = SimpleImputer(strategy="mean")
df[num_cols.columns] = imputer.fit_transform(num_cols)

encoder = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str)
    df[col] = encoder.fit_transform(df[col])

output_file = os.path.join(
    output_dir,
    "stress_clean.csv"
)

df.to_csv(output_file, index=False)

print("Output =", output_file)
print("Preprocessing selesai")