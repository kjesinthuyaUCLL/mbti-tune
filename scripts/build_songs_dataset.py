from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler

RAW = Path("data/raw/raw_playlists")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

rows = []

# 1. Load all track-level playlists
for mbti_dir in RAW.iterdir():
    if not mbti_dir.is_dir():
        continue
    mbti = mbti_dir.name

    for file in mbti_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df["mbti"] = mbti
        rows.append(df)

df = pd.concat(rows, ignore_index=True)
print("Loaded:", df.shape)

# 2. Standardize audio feature names
rename_map = {
    "Dance": "danceability",
    "Energy": "energy",
    "Loud (Db)": "loudness",
    "Speech": "speechiness",
    "Acoustic": "acousticness",
    "Instrumental": "instrumentalness",
    "Live": "liveness",
    "Valence": "valence",
    "BPM": "tempo",
}

df = df.rename(columns=rename_map)

# 3. Select song features
song_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo"
]

# 4. Add MBTI letter targets
def mbti_to_letters(m):
    m = m.upper()
    return {
        "E": 1.0 if m[0] == "E" else 0.0,
        "S": 1.0 if m[1] == "S" else 0.0,
        "T": 1.0 if m[2] == "T" else 0.0,
        "J": 1.0 if m[3] == "J" else 0.0,
    }

for idx, row in df.iterrows():
    letters = mbti_to_letters(row["mbti"])
    for k, v in letters.items():
        df.loc[idx, k] = v

# 5. Fit scaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[song_features])

# Save scaler
with open(PROCESSED / "song_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature list
with open(PROCESSED / "song_features.json", "w") as f:
    json.dump(song_features, f, indent=2)

# 6. Save processed dataset
df.to_csv(PROCESSED / "song_level_mbti.csv", index=False)

print("Saved song_level_mbti.csv")
