from pathlib import Path
import pandas as pd

RAW_PLAYLIST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_playlists"

print("Checking raw_playlists/ structure:", RAW_PLAYLIST_DIR)

for personality_dir in RAW_PLAYLIST_DIR.iterdir():
    if personality_dir.is_dir():
        print(f"\n=== {personality_dir.name} ===")
        files = list(personality_dir.glob("*.csv"))
        print(f"Found {len(files)} playlist files")

        if files:
            df = pd.read_csv(files[0])
            print("Columns in first playlist file:")
            print(df.columns.tolist())
            break
