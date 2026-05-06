import pandas as pd

df = pd.read_csv("data/processed/mbti_classifier_dataset.csv")

print("Columns:", len(df.columns))
print(df.columns.tolist())

print("\nMissing expected MBTI raw features?")

expected = [
    "danceability_mean",
    "energy_mean",
    "valence_mean",
    "tempo_mean",
    "Cminor_count",
    "CMajor_count"
]

for col in expected:
    print(col, "→", col in df.columns)