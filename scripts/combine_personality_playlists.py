import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np

RAW_PATH = Path("data/raw_playlists")  # Your 16 folders
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

def safe_lower(text):
    """Safely convert to lowercase, handling NaN and None"""
    if pd.isna(text) or text is None:
        return "unknown"
    return str(text).lower().strip()

def combine_playlists_for_personality(personality_dir):
    """Combine all CSV files in a personality folder into one DataFrame with song frequencies"""
    all_dfs = []
    
    for csv_file in personality_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Ensure required columns exist
            if "Song" not in df.columns or "Artist" not in df.columns:
                print(f"  Warning: {csv_file.name} missing Song/Artist columns, skipping")
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
            continue
    
    if not all_dfs:
        print(f"  No valid CSV files found in {personality_dir.name}")
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Create a safe key for counting songs
    combined["_temp_key"] = combined.apply(
        lambda row: f"{safe_lower(row['Song'])}|{safe_lower(row['Artist'])}", axis=1
    )
    
    # Count frequencies
    song_counts = Counter(combined["_temp_key"])
    combined["song_frequency"] = combined["_temp_key"].map(song_counts)
    
    # Clean up: remove unknown entries if you want (optional)
    unknown_count = combined[combined["_temp_key"] == "unknown|unknown"].shape[0]
    if unknown_count > 0:
        print(f"  Found {unknown_count} rows with unknown song/artist")
    
    # Drop the temporary key column
    combined = combined.drop(columns=["_temp_key"])
    
    return combined

# Process all 16 folders
personality_folders = [d for d in RAW_PATH.iterdir() if d.is_dir()]
personality_folders = [f for f in personality_folders if f.name not in ["processed", "raw"]]  # Skip any non-personality folders

print(f"Found {len(personality_folders)} personality folders:")
for f in personality_folders:
    print(f"  - {f.name}")

print("\nProcessing...")

for folder in personality_folders:
    personality = folder.name
    print(f"\nProcessing {personality}...")
    df = combine_playlists_for_personality(folder)
    
    if df is not None and len(df) > 0:
        output_path = PROCESSED_PATH / f"{personality}.csv"
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        unique_songs = df["_temp_key"].nunique() if "_temp_key" in df else df.shape[0]
        total_plays = df["song_frequency"].sum() if "song_frequency" in df else len(df)
        
        print(f"  ✓ Saved {len(df)} total rows, {total_plays} total plays")
        print(f"  ✓ Unique songs: {unique_songs}")
        print(f"  ✓ Average frequency: {total_plays/len(df):.2f}x per song")
    else:
        print(f"  ✗ No data for {personality}")

print("\n" + "="*50)
print("✅ Done! CSV files saved to data/processed/")
print("="*50)

# List all created files
print("\nCreated files:")
for f in PROCESSED_PATH.glob("*.csv"):
    df_temp = pd.read_csv(f)
    print(f"  {f.name}: {len(df_temp)} rows, {df_temp['song_frequency'].sum() if 'song_frequency' in df_temp.columns else len(df_temp)} total plays")