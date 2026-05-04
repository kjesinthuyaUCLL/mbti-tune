import pandas as pd
import os
from pathlib import Path

def get_csv_info(base_path):
    target_dirs = [
        "data/pretrain", 
        "data/merged", 
        "data/deduplicated", 
        "data/processed",
        "data/raw"
    ]
    
    found_any = False
    print("="*80)
    print(f"{'PROJECT CSV DATA REPORT':^80}")
    print("="*80)

    for sub_dir in target_dirs:
        folder = Path(base_path) / sub_dir
        if folder.exists():
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                found_any = True
                print(f"\n📂 DIRECTORY: {sub_dir} ({len(csv_files)} files)")
                print("-" * 50)
                for file_path in csv_files:
                    try:
                        # Load only header to get column info quickly
                        df_head = pd.read_csv(file_path, nrows=0)
                        # Count rows efficiently
                        row_count = sum(1 for _ in open(file_path, encoding='utf-8', errors='ignore')) - 1
                        
                        print(f"📄 {file_path.name}")
                        print(f"   - Total Rows:    {row_count:,}")
                        print(f"   - Total Columns: {len(df_head.columns)}")
                        print(f"   - Columns:       {list(df_head.columns)}")
                        print("")
                    except Exception as e:
                        print(f"❌ Error reading {file_path.name}: {e}")
    
    if not found_any:
        print("\n⚠️ No CSV files found. Ensure your 'data/' folder structure is populated.")
    print("="*80)

# If running in Colab, use your Drive path. Otherwise, use relative path.
drive_path = "/content/drive/My Drive/mbti_tune_data"
project_path = drive_path if os.path.exists(drive_path) else "."

get_csv_info(project_path)