# scripts/combine_playlists.py
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Define MBTI types
MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / 'data' / 'raw_playlists'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'combined'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common column names that might appear in your CSVs
SONG_COLUMNS = ['Song', 'song', 'Track', 'track', 'Title', 'title', 'Track Name']
ARTIST_COLUMNS = ['Artist', 'artist', 'Artists', 'artists', 'Artist Name']
GENRE_COLUMNS = ['Genres', 'genres', 'Genre', 'genre']
FEATURE_COLUMNS = ['Energy', 'Dance', 'Acoustic', 'Valence', 'BPM', 'Popularity', 
                   'Instrumental', 'Live', 'Speech', 'Danceability', 'Acousticness',
                   'Instrumentalness', 'Liveness', 'Speechiness', 'Tempo']

def find_column(df, possible_names):
    """Find which column name exists in the dataframe"""
    for name in possible_names:
        if name in df.columns:
            return name
        # Also check case-insensitive
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None

def combine_playlists_for_type(mbti_type):
    """Combine all CSV files for one personality type"""
    folder_path = INPUT_DIR / mbti_type
    
    if not folder_path.exists():
        print(f"[SKIP] {mbti_type} - folder not found")
        return None
    
    csv_files = list(folder_path.glob('*.csv')) + list(folder_path.glob('*.CSV'))
    if not csv_files:
        print(f"[SKIP] {mbti_type} - no CSV files")
        return None
    
    print(f"\n[PROCESSING] {mbti_type} - {len(csv_files)} playlists")
    
    all_rows = []
    song_counter = Counter()
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            if len(df) == 0:
                continue
            
            # Find song and artist columns
            song_col = find_column(df, SONG_COLUMNS)
            artist_col = find_column(df, ARTIST_COLUMNS)
            
            # Track frequency for each song
            if song_col and artist_col:
                for _, row in df.iterrows():
                    song_key = f"{row[song_col]}|{row[artist_col]}"
                    song_counter[song_key] += 1
            elif song_col:
                for song in df[song_col]:
                    song_counter[song] += 1
            
            # Add to collection
            all_rows.append(df)
            print(f"   + {len(df)} songs")
            
        except Exception as e:
            print(f"   ERROR: {csv_file.name} - {str(e)[:50]}")
            continue
    
    if not all_rows:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_rows, ignore_index=True)
    
    # Find all important columns
    song_col = find_column(combined_df, SONG_COLUMNS)
    artist_col = find_column(combined_df, ARTIST_COLUMNS)
    genre_col = find_column(combined_df, GENRE_COLUMNS)
    
    print(f"\n   Found columns: Song={song_col}, Artist={artist_col}, Genre={genre_col}")
    
    # Create a clean dataframe with one row per unique song
    if song_col and artist_col:
        # Create unique key for each song
        combined_df['_song_key'] = combined_df[song_col].astype(str) + '|' + combined_df[artist_col].astype(str)
        
        # Add frequency column
        combined_df['appearance_count'] = combined_df['_song_key'].map(song_counter)
        
        # Drop duplicates, keeping first occurrence
        clean_df = combined_df.drop_duplicates(subset=['_song_key']).copy()
        
        # Remove temporary key
        clean_df = clean_df.drop(columns=['_song_key'])
        
        original_count = len(combined_df)
        print(f"   Duplicates removed: {original_count} -> {len(clean_df)} unique songs")
        
    else:
        clean_df = combined_df.drop_duplicates()
        clean_df['appearance_count'] = 1
        print(f"   No song/artist columns found. Using simple dedup.")
    
    # Add MBTI type column
    clean_df['mbti_type'] = mbti_type
    
    # Ensure numeric columns are proper numbers
    for col in FEATURE_COLUMNS:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    # Sort by appearance count (most frequent first)
    if 'appearance_count' in clean_df.columns:
        clean_df = clean_df.sort_values('appearance_count', ascending=False)
    
    # Save to CSV
    output_file = OUTPUT_DIR / f'{mbti_type}_songs.csv'
    clean_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Calculate statistics
    stats = {
        'playlists': len(csv_files),
        'raw_songs': len(combined_df),
        'unique_songs': len(clean_df),
        'total_appearances': sum(song_counter.values()),
        'max_frequency': max(song_counter.values()) if song_counter else 0,
        'avg_frequency': sum(song_counter.values()) / len(song_counter) if song_counter else 0
    }
    
    print(f"   SAVED: {len(clean_df)} unique songs")
    print(f"   Frequency: max={stats['max_frequency']}, avg={stats['avg_frequency']:.2f}")
    
    return clean_df, stats

def main():
    print("=" * 60)
    print("MBTI Tune - Build Personality Song Datasets")
    print("=" * 60)
    
    all_stats = {}
    all_dfs = []
    
    for mbti in MBTI_TYPES:
        result = combine_playlists_for_type(mbti)
        if result:
            df, stats = result
            all_stats[mbti] = stats
            all_dfs.append(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - 16 Personality Datasets")
    print("=" * 60)
    
    total_raw = 0
    total_unique = 0
    
    for mbti, stats in all_stats.items():
        print(f"\n{mbti}:")
        print(f"   Playlists: {stats['playlists']}")
        print(f"   Raw songs: {stats['raw_songs']}")
        print(f"   Unique songs: {stats['unique_songs']}")
        print(f"   Total appearances: {stats['total_appearances']}")
        print(f"   Most frequent: {stats['max_frequency']} times")
        
        total_raw += stats['raw_songs']
        total_unique += stats['unique_songs']
    
    print("\n" + "=" * 60)
    print("TOTALS")
    print("=" * 60)
    print(f"   Total raw songs across all personalities: {total_raw}")
    print(f"   Total unique songs across all personalities: {total_unique}")
    print(f"\n   Output files: {OUTPUT_DIR}")
    print(f"   Files created: {{MBTI}}_songs.csv (16 files)")
    
    # Optionally create a master file with all personalities combined
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_file = OUTPUT_DIR / 'all_personalities_songs.csv'
        master_df.to_csv(master_file, index=False, encoding='utf-8')
        print(f"\n   Master file: all_personalities_songs.csv")
        print(f"   Total rows in master: {len(master_df)}")

if __name__ == "__main__":
    main()