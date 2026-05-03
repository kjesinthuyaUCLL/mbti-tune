import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pickle
import json

# Use deduplicated data if available, otherwise fallback to processed
PROCESSED_PATH = Path("data/processed")
DEDUP_PATH = Path("data/deduplicated")
OUTPUT_PATH = Path("data/training")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Use deduplicated if it exists and has files
DATA_PATH = DEDUP_PATH if DEDUP_PATH.exists() and list(DEDUP_PATH.glob("*.csv")) else PROCESSED_PATH
print(f"Using data from: {DATA_PATH}")

# MBTI to 4 dimensions mapping
MBTI_DIMENSIONS = {
    'E': ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'],
    'I': ['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
    'N': ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'],
    'S': ['ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'],
    'T': ['ENTJ', 'ENTP', 'ESTJ', 'ESTP', 'INTJ', 'INTP', 'ISTJ', 'ISTP'],
    'F': ['ENFJ', 'ENFP', 'ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'],
    'J': ['ENFJ', 'ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'],
    'P': ['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP']
}

def personality_to_dimensions(personality):
    """Convert MBTI type to 4 binary values"""
    return [
        1.0 if personality in MBTI_DIMENSIONS['E'] else 0.0,
        1.0 if personality in MBTI_DIMENSIONS['N'] else 0.0,
        1.0 if personality in MBTI_DIMENSIONS['T'] else 0.0,
        1.0 if personality in MBTI_DIMENSIONS['J'] else 0.0,
    ]

# Audio feature columns (from your data)
AUDIO_FEATURES = [
    'BPM', 'Energy', 'Dance', 'Acoustic', 'Instrumental',
    'Valence', 'Speech', 'Live', 'Popularity', 'Loud (Db)'
]

# Columns to handle as categorical (convert to numeric)
CATEGORICAL_FEATURES = ['Key', 'Camelot', 'Time Signature']

def encode_camelot(camelot):
    """Convert Camelot key (e.g., '8A', '10B') to numeric value 1-24"""
    if pd.isna(camelot):
        return 12  # Middle value
    try:
        number = int(''.join(filter(str.isdigit, str(camelot))))
        letter = ''.join(filter(str.isalpha, str(camelot))).upper()
        offset = 0 if letter == 'A' else 12
        return number + offset
    except:
        return 12

def encode_key(key):
    """Convert musical key (e.g., 'A Minor', 'D Major') to numeric 0-23"""
    key_map = {
        'C Major': 0, 'C Minor': 1, 'C# Major': 2, 'C# Minor': 3,
        'D Major': 4, 'D Minor': 5, 'D# Major': 6, 'D# Minor': 7,
        'E Major': 8, 'E Minor': 9, 'F Major': 10, 'F Minor': 11,
        'F# Major': 12, 'F# Minor': 13, 'G Major': 14, 'G Minor': 15,
        'G# Major': 16, 'G# Minor': 17, 'A Major': 18, 'A Minor': 19,
        'A# Major': 20, 'A# Minor': 21, 'B Major': 22, 'B Minor': 23
    }
    return key_map.get(str(key), 12)

def parse_duration(duration_str):
    """Convert '03:40' to seconds (220)"""
    try:
        parts = str(duration_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        pass
    return 180  # Default 3 minutes

def load_all_data():
    """Load all personality CSV files and combine"""
    all_data = []
    
    csv_files = list(DATA_PATH.glob("*.csv"))
    print(f"Found {len(csv_files)} personality files")
    
    for csv_file in csv_files:
        personality = csv_file.stem
        df = pd.read_csv(csv_file)
        
        # Handle Duration column
        if 'Duration' in df.columns:
            df['Duration_sec'] = df['Duration'].apply(parse_duration)
        
        # Handle Camelot key
        if 'Camelot' in df.columns:
            df['Camelot_num'] = df['Camelot'].apply(encode_camelot)
        
        # Handle musical Key
        if 'Key' in df.columns:
            df['Key_num'] = df['Key'].apply(encode_key)
        
        # Add personality label and dimensions
        df['personality'] = personality
        dimensions = personality_to_dimensions(personality)
        df['E_score'] = dimensions[0]
        df['N_score'] = dimensions[1]
        df['T_score'] = dimensions[2]
        df['J_score'] = dimensions[3]
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Handle missing values
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())
    
    print(f"Loaded {len(combined)} total rows")
    return combined

def prepare_features(df):
    """Prepare feature matrix and targets"""
    
    # Build feature set
    feature_cols = []
    
    # Add audio features that exist
    for f in AUDIO_FEATURES:
        if f in df.columns:
            feature_cols.append(f)
        elif f == 'Loud (Db)' and 'Loud (Db)' in df.columns:
            feature_cols.append('Loud (Db)')
    
    # Add derived features
    if 'Duration_sec' in df.columns:
        feature_cols.append('Duration_sec')
    if 'Camelot_num' in df.columns:
        feature_cols.append('Camelot_num')
    if 'Key_num' in df.columns:
        feature_cols.append('Key_num')
    
    # Ensure Time Signature is included
    if 'Time Signature' in df.columns:
        feature_cols.append('Time Signature')
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Extract features
    X = df[feature_cols].values.astype(np.float32)
    
    # Extract targets
    y = df[['E_score', 'N_score', 'T_score', 'J_score']].values.astype(np.float32)
    
    # Sample weights (song frequency)
    sample_weights = df['song_frequency'].values.astype(np.float32)
    
    # Metadata
    metadata = df[['Song', 'Artist', 'personality']].copy()
    
    return X, y, sample_weights, metadata, feature_cols

def split_and_save(X, y, sample_weights, metadata, feature_cols, test_size=0.2, val_size=0.1):
    """Split into train/val/test and save"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, w_temp, w_test, meta_temp, meta_test = train_test_split(
        X, y, sample_weights, metadata, test_size=test_size, 
        random_state=42, stratify=metadata['personality']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, w_train, w_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, w_temp, meta_temp, test_size=val_ratio, 
        random_state=42, stratify=meta_temp['personality']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val: {len(X_val):,} rows")
    print(f"  Test: {len(X_test):,} rows")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_data = {
        'X': torch.tensor(X_train_scaled, dtype=torch.float32),
        'y': torch.tensor(y_train, dtype=torch.float32),
        'weights': torch.tensor(w_train, dtype=torch.float32)
    }
    
    val_data = {
        'X': torch.tensor(X_val_scaled, dtype=torch.float32),
        'y': torch.tensor(y_val, dtype=torch.float32),
        'weights': torch.tensor(w_val, dtype=torch.float32)
    }
    
    test_data = {
        'X': torch.tensor(X_test_scaled, dtype=torch.float32),
        'y': torch.tensor(y_test, dtype=torch.float32),
        'weights': torch.tensor(w_test, dtype=torch.float32),
        'metadata': meta_test.reset_index(drop=True)
    }
    
    # Save everything
    torch.save(train_data, OUTPUT_PATH / 'train.pt')
    torch.save(val_data, OUTPUT_PATH / 'val.pt')
    torch.save(test_data, OUTPUT_PATH / 'test.pt')
    
    with open(OUTPUT_PATH / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(OUTPUT_PATH / 'features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"\n✅ Saved to {OUTPUT_PATH}/")
    
    return scaler

def main():
    print("="*60)
    print("DAY 2: Building Training Dataset")
    print("="*60)
    
    print("\n1. Loading data...")
    df = load_all_data()
    
    print("\n2. Preparing features...")
    X, y, sample_weights, metadata, features = prepare_features(df)
    print(f"   Feature matrix: {X.shape}")
    print(f"   Target matrix: {y.shape}")
    
    print("\n3. Splitting and saving...")
    split_and_save(X, y, sample_weights, metadata, features)
    
    print("\n" + "="*60)
    print("✅ DAY 2 COMPLETE!")
    print("="*60)
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(X):,}")
    print(f"   Features: {len(features)}")
    print(f"   Avg frequency weight: {sample_weights.mean():.2f}")
    
    dim_names = ['Extraversion (E)', 'Intuition (N)', 'Thinking (T)', 'Judging (J)']
    print(f"\n📊 MBTI Dimension Balance:")
    for i, name in enumerate(dim_names):
        balance = y[:, i].mean() * 100
        print(f"   {name}: {balance:.1f}%")

if __name__ == "__main__":
    main()