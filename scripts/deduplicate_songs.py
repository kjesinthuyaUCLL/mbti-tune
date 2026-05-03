import pandas as pd
from pathlib import Path
from collections import defaultdict

PROCESSED_PATH = Path("data/processed")
DEDUP_PATH = Path("data/deduplicated")
DEDUP_PATH.mkdir(parents=True, exist_ok=True)

def get_personality_song_counts():
    """Get total songs per personality (from earlier run)"""
    # From your earlier output
    return {
        'ENFJ': 3216, 'ENFP': 3877, 'ENTJ': 2129, 'ENTP': 2337,
        'ESFJ': 2380, 'ESFP': 2994, 'ESTJ': 1577, 'ESTP': 3019,
        'INFJ': 3993, 'INFP': 5266, 'INTJ': 3177, 'INTP': 3411,
        'ISFJ': 4147, 'ISFP': 7011, 'ISTJ': 2326, 'ISTP': 3483
    }

def create_song_keys(df):
    """Create unique key for each song-artist combination"""
    df['_song_key'] = df.apply(
        lambda row: f"{str(row['Song']).lower().strip()}|{str(row['Artist']).lower().strip()}",
        axis=1
    )
    return df

def deduplicate_across_personalities():
    """For each song, keep it only in the personality with the smallest dataset"""
    
    personality_counts = get_personality_song_counts()
    
    # Load all personality data
    all_data = {}
    for csv_file in PROCESSED_PATH.glob("*.csv"):
        personality = csv_file.stem
        df = pd.read_csv(csv_file)
        df = create_song_keys(df)
        all_data[personality] = df
    
    # Track which songs appear in which personalities
    song_to_personalities = defaultdict(set)
    for personality, df in all_data.items():
        for song_key in df['_song_key'].unique():
            song_to_personalities[song_key].add(personality)
    
    # Find duplicate songs (appear in >1 personality)
    duplicate_songs = {
        song: personalities 
        for song, personalities in song_to_personalities.items() 
        if len(personalities) > 1
    }
    
    print(f"Total unique songs: {len(song_to_personalities)}")
    print(f"Songs appearing in multiple personalities: {len(duplicate_songs)} ({len(duplicate_songs)/len(song_to_personalities)*100:.1f}%)")
    
    # For each duplicate song, decide which personality keeps it
    keep_assignment = {}
    for song, personalities in duplicate_songs.items():
        # Find personality with smallest total songs
        smallest = min(personalities, key=lambda p: personality_counts[p])
        keep_assignment[song] = smallest
    
    print(f"\nSample assignments (first 10):")
    for i, (song, keep_p) in enumerate(list(keep_assignment.items())[:10]):
        all_p = list(song_to_personalities[song])
        print(f"  {song[:40]}... → {keep_p} (was in {all_p})")
    
    # Remove duplicates from other personalities
    for personality, df in all_data.items():
        # Mark rows to keep
        keep_mask = df.apply(
            lambda row: keep_assignment.get(row['_song_key'], personality) == personality,
            axis=1
        )
        
        removed_count = (~keep_mask).sum()
        df_cleaned = df[keep_mask].copy()
        
        # Drop temporary key
        df_cleaned = df_cleaned.drop(columns=['_song_key'])
        
        # Save cleaned file
        output_path = DEDUP_PATH / f"{personality}.csv"
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"\n{personality}:")
        print(f"  Original: {len(df)} rows")
        print(f"  Removed: {removed_count} rows (duplicates assigned to smaller personalities)")
        print(f"  Kept: {len(df_cleaned)} rows")
    
    # Summary of changes
    print("\n" + "="*60)
    print("✅ DEDUPLICATION COMPLETE!")
    print("="*60)
    
    # Verify no duplicates remain across personalities
    all_keys = {}
    for personality in all_data.keys():
        df = pd.read_csv(DEDUP_PATH / f"{personality}.csv")
        df = create_song_keys(df)
        for key in df['_song_key'].unique():
            if key in all_keys:
                print(f"WARNING: {key} still appears in {all_keys[key]} and {personality}")
            all_keys[key] = personality
    
    print(f"\nFinal unique songs across all personalities: {len(all_keys)}")
    
    return DEDUP_PATH

if __name__ == "__main__":
    dedup_path = deduplicate_across_personalities()
    print(f"\nUse these cleaned files for training: {dedup_path}")