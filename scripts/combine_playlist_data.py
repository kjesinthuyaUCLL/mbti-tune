import pandas as pd
from pathlib import Path

# Update path - files are directly in data/raw/
RAW_PATH = Path("data/raw")
MERGED_PATH = Path("data/merged")
MERGED_PATH.mkdir(exist_ok=True)

all_data = []

# Find all CSV files directly in data/raw/
csv_files = list(RAW_PATH.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in {RAW_PATH}")

if len(csv_files) == 0:
    print("\nChecking alternative paths...")
    # Try data/raw_playlists/
    RAW_PATH = Path("data/raw_playlists")
    csv_files = list(RAW_PATH.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {RAW_PATH}")
    
    if len(csv_files) == 0:
        # Try each subfolder
        for folder in RAW_PATH.iterdir():
            if folder.is_dir():
                csv_files.extend(list(folder.glob("*.csv")))
        print(f"Found {len(csv_files)} CSV files in subfolders")

for csv_file in csv_files:
    # Extract personality from filename (e.g., ENFJ.csv -> ENFJ)
    personality = csv_file.stem
    
    # Also check if filename contains personality pattern
    if len(personality) == 4 and personality.isupper():
        # Valid MBTI type
        pass
    elif '_' in personality:
        # Handle cases like "ENFJ_playlists.csv"
        personality = personality.split('_')[0]
    
    try:
        df = pd.read_csv(csv_file)
        df['personality'] = personality
        all_data.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} rows")
    except Exception as e:
        print(f"  Error loading {csv_file.name}: {e}")

if len(all_data) == 0:
    print("\n❌ No data files found. Please check your file structure.")
    print("\nYour data should be in one of these locations:")
    print("  - data/raw/ENFJ.csv")
    print("  - data/raw_playlists/ENFJ.csv")
    print("  - data/raw_playlists/ENFJ/playlist1.csv")
    exit()

combined = pd.concat(all_data, ignore_index=True)

# Display basic info
print(f"\n📊 Combined data:")
print(f"   Total rows: {len(combined)}")
print(f"   Personality distribution:")
print(combined['personality'].value_counts())

# Add MBTI dimension columns
mbti_map = {
    'E': ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'],
    'I': ['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
    'N': ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'],
    'S': ['ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
    'T': ['ENTJ', 'ENTP', 'ESTJ', 'ESTP', 'INTJ', 'INTP', 'ISTJ', 'ISTP'],
    'F': ['ENFJ', 'ENFP', 'ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'],
    'J': ['ENFJ', 'ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'],
    'P': ['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP']
}

def get_dimensions(personality):
    return [
        1.0 if personality in mbti_map['E'] else 0.0,
        1.0 if personality in mbti_map['N'] else 0.0,
        1.0 if personality in mbti_map['T'] else 0.0,
        1.0 if personality in mbti_map['J'] else 0.0,
    ]

combined['E'] = combined['personality'].apply(lambda x: get_dimensions(x)[0])
combined['N'] = combined['personality'].apply(lambda x: get_dimensions(x)[1])
combined['T'] = combined['personality'].apply(lambda x: get_dimensions(x)[2])
combined['J'] = combined['personality'].apply(lambda x: get_dimensions(x)[3])

# Identify feature columns (audio features with mean/stdev)
feature_cols = [col for col in combined.columns if any(x in col.lower() for x in 
    ['mean', 'stdev', 'dance', 'energy', 'valence', 'acoustic', 
     'speech', 'liveness', 'instrumental', 'tempo', 'loudness', 'mode'])]

# Also include key counts
key_cols = [col for col in combined.columns if any(x in col.lower() for x in 
    ['minor_count', 'major_count', 'key', 'count'])]
feature_cols = list(set(feature_cols + key_cols))

# Remove non-feature columns
feature_cols = [col for col in feature_cols if col not in ['personality', 'E', 'N', 'T', 'J']]

print(f"\n📊 Features: {len(feature_cols)} columns")
print(f"   Sample features: {feature_cols[:10]}")

# Save to CSV
combined.to_csv(MERGED_PATH / 'playlist_data.csv', index=False)
print(f"\n✅ Saved to {MERGED_PATH}/playlist_data.csv")

# Also save feature list
import json
with open(MERGED_PATH / 'features.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print(f"✅ Saved feature list to {MERGED_PATH}/features.json")

# Display dimension balance
print(f"\n📊 MBTI Dimension Balance:")
for dim in ['E', 'N', 'T', 'J']:
    balance = combined[dim].mean() * 100
    print(f"   {dim}: {balance:.1f}%")