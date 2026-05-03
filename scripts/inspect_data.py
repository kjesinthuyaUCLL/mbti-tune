import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed")

# Check one file to see all columns
sample_file = PROCESSED_PATH / "ENFJ.csv"
df = pd.read_csv(sample_file)

print("Columns available in your data:")
print("-" * 40)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

print("\n" + "="*40)
print("First 3 rows sample:")
print(df.head(3).to_string())

print("\n" + "="*40)
print("Data types:")
print(df.dtypes)

print("\n" + "="*40)
print("Missing values:")
print(df.isnull().sum())