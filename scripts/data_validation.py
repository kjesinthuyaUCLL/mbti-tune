import pandas as pd
from pathlib import Path

def validate_datasets(base_path):
    # Define the expected structure based on your project
    datasets = {
        "114k Pretrain": base_path / "data/pretrain/spotify_tracks.csv",
        "Merged Playlist (30k)": base_path / "data/merged/playlist_data.csv",
    }

    # Folders containing multiple MBTI files
    folders = {
        "Deduplicated (16 MBTI)": base_path / "data/deduplicated",
        "Raw Aggregated (16 MBTI)": base_path / "data/raw",
    }

    print("="*80)
    print(f"{'Dataset Validation Report':^80}")
    print("="*80)

    # 1. Check Single CSV Files
    for name, path in datasets.items():
        if path.exists():
            df = pd.read_csv(path)
            print(f"\nFILE: {name} ({path.name})")
            print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
            print(f"   Features: {list(df.columns[:5])}... {list(df.columns[-2:])}")
            print(f"   Dtypes: {df.dtypes.iloc[0]} (Sample)")
        else:
            print(f"\n⚠️ MISSING FILE: {path}")

    # 2. Check Folders with MBTI Files
    for folder_name, folder_path in folders.items():
        if folder_path.exists():
            files = list(folder_path.glob("*.csv"))
            print(f"\nFOLDER: {folder_name} ({len(files)} files found)")
            if files:
                # Inspect a sample file from the folder
                sample_df = pd.read_csv(files[0])
                print(f"   Sample File ({files[0].name}) Stats:")
                print(f"   Rows: {len(sample_df):,} | Columns: {len(sample_df.columns)}")
                print(f"   Columns: {list(sample_df.columns)}")
        else:
            print(f"\n⚠️ MISSING FOLDER: {folder_path}")

    # 3. Specific Check for Raw Playlists (Deep Subdirectories)
    raw_playlist_dir = base_path / "data/raw_playlists"
    if raw_playlist_dir.exists():
        subdirs = [d for d in raw_playlist_dir.iterdir() if d.is_dir()]
        total_csvs = len(list(raw_playlist_dir.rglob("*.csv")))
        print(f"\nFOLDER: Raw Playlists (Nested Structure)")
        print(f"   Sub-folders: {len(subdirs)} (e.g., ENFJ, INTJ)")
        print(f"   Total Playlist Files: {total_csvs}")
    else:
        print(f"\n⚠️ MISSING FOLDER: {raw_playlist_dir}")

    print("\n" + "="*80)

if __name__ == "__main__":
    project_root = Path("./") 
    validate_datasets(project_root)