from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler

RAW = Path("data/raw/mbti_playlists")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)

# 1. Load all MBTI playlist summary CSVs
df_list = []
for file in RAW.glob("*.csv"):
    df_list.append(pd.read_csv(file))

df = pd.concat(df_list, ignore_index=True)

print("Loaded:", df.shape)

# 2. Define playlist features (Dataset 2 schema)
playlist_features = [
    'danceability_mean','danceability_stdev',
    'energy_mean','energy_stdev',
    'loudness_mean','loudness_stdev',
    'mode_mean','mode_stdev',
    'speechiness_mean','speechiness_stdev',
    'acousticness_mean','acousticness_stdev',
    'liveness_mean','liveness_stdev',
    'valence_mean','valence_stdev',
    'tempo_mean','tempo_stdev',
    'instrumentalness_mean','instrumentalness_stdev',
    # 24 key counts
    'Cminor_count','CMajor_count','C#/Dbminor_count','C#/DbMajor_count',
    'Dminor_count','DMajor_count','D#_Ebminor_count','D#_EbMajor_count',
    'Eminor_count','EMajor_count','Fminor_count','FMajor_count',
    'F#/Gbminor_count','F#/GbMajor_count','Gminor_count','GMajor_count',
    'G#/Abminor_count','G#/AbMajor_count','Aminor_count','AMajor_count',
    'A#/Bbminor_count','A#/BbMajor_count','Bminor_count','BMajor_count'
]

# 3. Add MBTI letter targets
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

# 4. Fit scaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[playlist_features])

# Save scaler
with open(PROCESSED / "playlist_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature list
with open(PROCESSED / "playlist_features.json", "w") as f:
    json.dump(playlist_features, f, indent=2)

# 5. Save processed dataset
df.to_csv(PROCESSED / "playlist_data_from_mbti_playlists.csv", index=False)

print("Saved playlist_data_from_mbti_playlists.csv")
