import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_TRACKS = ROOT / "data/raw/pretrain/spotify_tracks.csv"
MBTI_DIR = ROOT / "data/raw/raw_playlists"
MBTI_SIMPLE = ROOT / "data/raw/mbti_playlists"

# =========================
# 1. CHECK PRETRAIN DATASET
# =========================

print("\n================ PRETRAIN DATA ================\n")

df = pd.read_csv(RAW_TRACKS, nrows=1000)

print("Shape sample:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types (important only):")
print(df.dtypes.value_counts())

print("\nMissing values (% top 10):")
print((df.isna().mean().sort_values(ascending=False).head(10) * 100).round(2))


# =========================
# 2. CHECK MBTI SIMPLE DATASET
# =========================

print("\n================ MBTI PLAYLIST (CLEAN) ================\n")

if MBTI_SIMPLE.exists():
    files = list(MBTI_SIMPLE.glob("*.csv"))
    print("Files:", len(files))

    sample = pd.read_csv(files[0])
    print("\nColumns:")
    print(sample.columns.tolist())

    print("\nSample rows:", len(sample))
    print("\nColumn types:")
    print(sample.dtypes.value_counts())

# =========================
# 3. CHECK RAW PLAYLISTS (STRUCTURED)
# =========================

print("\n================ RAW PLAYLIST FOLDERS ================\n")

folders = [f for f in MBTI_DIR.iterdir() if f.is_dir()]
print("MBTI folders:", len(folders))

for folder in folders:
    files = list(folder.glob("*.csv"))

    # skip empty folders
    if len(files) == 0:
        continue

    sample_file = files[0]

    try:
        df = pd.read_csv(sample_file, nrows=5)

        print(f"\n{folder.name}:")
        print(" files:", len(files))
        print(" cols:", len(df.columns))
        print(" columns sample:", df.columns.tolist()[:8])

    except Exception as e:
        print(f"Error in {folder.name}: {e}")

# =========================
# 4. QUICK CONSISTENCY CHECK
# =========================

print("\n================ CONSISTENCY CHECK ================\n")

all_cols = None
consistent = True

for folder in folders:
    files = list(folder.glob("*.csv"))
    if not files:
        continue

    df = pd.read_csv(files[0], nrows=5)

    if all_cols is None:
        all_cols = set(df.columns)
    else:
        if set(df.columns) != all_cols:
            consistent = False
            print(f"❌ Column mismatch in {folder.name}")

if consistent:
    print("✅ All MBTI playlist folders have consistent schema")
else:
    print("⚠️ Some schema mismatches detected")