"""
Notebook Dataset and Output Tracking
Run this to see which notebook uses which datasets and outputs
"""

import os
import json
import pandas as pd
from datetime import datetime

def create_notebook_tracker():
    """Create a tracking dataframe for all notebooks"""
    
    notebooks_data = {
        "MBTI_Tracks_Autoencoder.ipynb": {
            "purpose": "Pretrain autoencoder on 113k songs",
            "input_datasets": [
                {
                    "name": "spotify_tracks.csv",
                    "path": "data/raw/pretrain/spotify_tracks.csv",
                    "size": "113,000 songs",
                    "features": "42 audio features (means + stds + key-modes)"
                }
            ],
            "output_files": [
                {
                    "name": "encoder_114k_weights.pth",
                    "path": "models/encoder_114k_weights.pth",
                    "type": "PyTorch state dict",
                    "description": "Pretrained encoder weights (42→16 latent dimensions)"
                },
                {
                    "name": "pretrain_scaler.pkl",
                    "path": "models/pretrain_scaler.pkl",
                    "type": "sklearn StandardScaler",
                    "description": "Scaler fitted on 113k songs"
                },
                {
                    "name": "pretrain_features.json",
                    "path": "models/pretrain_features.json",
                    "type": "JSON list",
                    "description": "42 feature names in exact order"
                }
            ],
            "model_architecture": {
                "type": "Autoencoder",
                "input_dim": 42,
                "encoding_dim": 16,
                "layers": [
                    "Linear(42→128) + BatchNorm + ReLU + Dropout(0.3)",
                    "Linear(128→64) + BatchNorm + ReLU",
                    "Linear(64→16) + BatchNorm + ReLU",
                    "Linear(16→64) + BatchNorm + ReLU",
                    "Linear(64→128) + BatchNorm + ReLU",
                    "Linear(128→42) + Sigmoid"
                ]
            },
            "training_config": {
                "epochs": "Likely 50-100",
                "batch_size": "256-512",
                "learning_rate": "0.001",
                "loss_function": "MSELoss",
                "optimizer": "Adam"
            },
            "expected_performance": {
                "reconstruction_loss": "Should be low (<0.1)",
                "validation_loss": "Should track training loss"
            }
        },
        
        "MBTI_Playlist_Classifier.ipynb": {
            "purpose": "Train playlist-level MBTI classifier using transfer learning",
            "input_datasets": [
                {
                    "name": "playlist_data.csv",
                    "path": "data/merged/playlist_data.csv",
                    "size": "4,816 playlists",
                    "features": "Aggregated audio features per playlist"
                },
                {
                    "name": "encoder_114k_weights.pth",
                    "path": "models/encoder_114k_weights.pth",
                    "type": "Pretrained encoder",
                    "description": "Used as feature extractor"
                }
            ],
            "output_files": [
                {
                    "name": "playlist_classifier_best.pth",
                    "path": "models/playlist_classifier_best.pth",
                    "type": "PyTorch state dict",
                    "description": "Best performing playlist classifier (recommended)"
                },
                {
                    "name": "playlist_classifier.pth",
                    "path": "models/playlist_classifier.pth",
                    "type": "PyTorch state dict",
                    "description": "Final epoch classifier (alternative)"
                }
            ],
            "model_architecture": {
                "type": "Transfer Learning + Classifier",
                "components": [
                    "PretrainedEncoder (frozen or fine-tuned)",
                    "Classifier: Linear(16→64) + BatchNorm + ReLU + Dropout(0.3)",
                    "Classifier: Linear(64→32) + BatchNorm + ReLU",
                    "Classifier: Linear(32→4) + Sigmoid"
                ],
                "total_params": "~30,000"
            },
            "training_config": {
                "epochs": "50-100",
                "batch_size": "32-64",
                "learning_rate": "0.001",
                "loss_function": "BCELoss (Binary Cross Entropy)",
                "optimizer": "Adam",
                "early_stopping": "Likely used"
            },
            "expected_performance": {
                "overall_mae": "37.7%",
                "letter_accuracy": "68.1%",
                "per_dimension": {
                    "E": {"MAE": "33.3%", "Accuracy": "72.7%"},
                    "N": {"MAE": "43.3%", "Accuracy": "60.5%"},
                    "T": {"MAE": "33.7%", "Accuracy": "73.9%"},
                    "J": {"MAE": "40.4%", "Accuracy": "65.1%"}
                }
            }
        },
        
        "MBTI_Song_Classifier.ipynb": {
            "purpose": "Train song-level MBTI classifier (alternative approach)",
            "input_datasets": [
                {
                    "name": "song_level_mbti.csv",
                    "path": "data/processed/song_level_mbti.csv",
                    "size": "Likely 10,000+ songs",
                    "features": "10 core audio features (simplified)"
                }
            ],
            "output_files": [
                {
                    "name": "song_classifier.pt",
                    "path": "models/song_classifier.pt",
                    "type": "PyTorch state dict",
                    "description": "Direct song-level classifier (10 features)"
                }
            ],
            "model_architecture": {
                "type": "Direct Classification",
                "input_dim": 10,
                "layers": [
                    "Linear(10→128) + BatchNorm + ReLU + Dropout(0.3)",
                    "Linear(128→64) + BatchNorm + ReLU + Dropout(0.3)",
                    "Linear(64→32) + BatchNorm + ReLU",
                    "Linear(32→4) + Sigmoid"
                ],
                "total_params": "~15,000"
            },
            "training_config": {
                "epochs": "50-100",
                "batch_size": "128-256",
                "learning_rate": "0.001",
                "loss_function": "BCELoss",
                "optimizer": "Adam"
            },
            "expected_performance": {
                "note": "Lower accuracy than playlist classifier (song-level prediction is harder)",
                "expected_letter_accuracy": "~55-60%"
            }
        },
        
        "MBTI_SHAP_Analysis.ipynb": {
            "purpose": "Explain model predictions using SHAP values",
            "input_datasets": [
                {
                    "name": "playlist_data.csv",
                    "path": "data/merged/playlist_data.csv",
                    "description": "Same dataset as classifier"
                },
                {
                    "name": "playlist_classifier_best.pth",
                    "path": "models/playlist_classifier_best.pth",
                    "description": "Trained model to explain"
                }
            ],
            "output_files": [
                {
                    "name": "shap_analysis_results.json",
                    "path": "models/shap_analysis_results.json",
                    "type": "JSON",
                    "description": "Feature importance scores"
                },
                {
                    "name": "shap_plots.png",
                    "path": "models/shap_plots.png",
                    "type": "Image",
                    "description": "Visualization of feature impacts"
                }
            ],
            "expected_outputs": [
                "Feature importance rankings",
                "SHAP summary plots",
                "Individual prediction explanations"
            ]
        }
    }
    
    return notebooks_data

def generate_tracking_report():
    """Generate a human-readable tracking report"""
    
    tracker = create_notebook_tracker()
    
    print("\n" + "="*80)
    print("MBTI TUNE - NOTEBOOK TRACKING REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for notebook_name, info in tracker.items():
        print("\n" + "="*80)
        print(f"📓 {notebook_name}")
        print("="*80)
        print(f"\n🎯 Purpose: {info['purpose']}")
        
        # Input Datasets
        print("\n📥 INPUT DATASETS:")
        for dataset in info['input_datasets']:
            print(f"   • {dataset['name']}")
            print(f"     Path: {dataset['path']}")
            if 'size' in dataset:
                print(f"     Size: {dataset['size']}")
            if 'features' in dataset:
                print(f"     Features: {dataset['features']}")
        
        # Output Files
        print("\n📤 OUTPUT FILES:")
        for output in info['output_files']:
            print(f"   • {output['name']} ({output['type']})")
            print(f"     Path: {output['path']}")
            print(f"     Description: {output['description']}")
        
        # Model Architecture
        if 'model_architecture' in info:
            print("\n🏗️ MODEL ARCHITECTURE:")
            arch = info['model_architecture']
            print(f"   Type: {arch['type']}")
            if 'input_dim' in arch:
                print(f"   Input Dimension: {arch['input_dim']}")
            if 'encoding_dim' in arch:
                print(f"   Encoding Dimension: {arch['encoding_dim']}")
            if 'layers' in arch:
                print("   Layers:")
                for layer in arch['layers']:
                    print(f"     - {layer}")
            if 'components' in arch:
                print("   Components:")
                for comp in arch['components']:
                    print(f"     - {comp}")
            if 'total_params' in arch:
                print(f"   Total Parameters: {arch['total_params']}")
        
        # Training Configuration
        if 'training_config' in info:
            print("\n⚙️ TRAINING CONFIGURATION:")
            for key, value in info['training_config'].items():
                print(f"   {key}: {value}")
        
        # Expected Performance
        if 'expected_performance' in info:
            print("\n📊 EXPECTED PERFORMANCE:")
            perf = info['expected_performance']
            if isinstance(perf, dict):
                for key, value in perf.items():
                    if isinstance(value, dict):
                        print(f"   {key}:")
                        for k, v in value.items():
                            print(f"     {k}: {v}")
                    else:
                        print(f"   {key}: {value}")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

def verify_files_exist():
    """Check if expected output files actually exist"""
    print("\n" + "="*80)
    print("FILE VERIFICATION")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracker = create_notebook_tracker()
    
    all_exist = True
    
    for notebook_name, info in tracker.items():
        print(f"\n📓 {notebook_name}:")
        for output in info['output_files']:
            filepath = os.path.join(base_dir, output['path'])
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                print(f"   ✅ {output['name']} ({size:.1f} KB)")
            else:
                print(f"   ❌ {output['name']} - NOT FOUND")
                all_exist = False
    
    return all_exist

def main():
    generate_tracking_report()
    verify_files_exist()

if __name__ == "__main__":
    main()
    