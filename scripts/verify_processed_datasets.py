import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

print("\n================ DATASET CHECK ================\n")

for file in PROC.glob("*.csv"):
    df = pd.read_csv(file)

    print(f"\n📄 {file.name}")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("Nulls:", df.isna().sum().sum())

print("\n================ DONE ================")