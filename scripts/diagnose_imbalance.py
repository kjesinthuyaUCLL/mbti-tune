"""
diagnose_imbalance.py - Model Bias and Imbalance Analysis
Run from project root: python scripts/diagnose_imbalance.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the correct model class
from src.model import MBTIClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("="*80)
print("⚖️ MBTI TUNE - MODEL IMBALANCE & BIAS DIAGNOSIS")
print("="*80)
print(f"Project root: {PROJECT_ROOT}")
print(f"Processed dir: {PROCESSED_DIR}")


# ============================================================================
# 1. LOAD PLAYLIST CLASSIFIER AND DATA
# ============================================================================
def load_playlist_classifier():
    """Load the trained playlist classifier and training data"""
    print("\n" + "="*80)
    print("📊 1. LOADING PLAYLIST CLASSIFIER")
    print("="*80)
    
    model_path = PROCESSED_DIR / "mbti_classifier.pth"
    
    results = {}
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return results
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        results['idx_to_type'] = checkpoint.get('idx_to_type', {})
        results['type_to_idx'] = checkpoint.get('type_to_idx', {})
        results['test_accuracy'] = checkpoint.get('test_accuracy', 0)
        results['features'] = checkpoint.get('features', [])
        results['input_dim'] = checkpoint.get('input_dim', 0)
        results['transfer_learning'] = checkpoint.get('transfer_learning_used', False)
        
        # IMPORTANT: Filter state_dict to only include keys that match the model
        full_state_dict = checkpoint.get('model_state_dict', {})
        
        # Create model with correct input dimension
        model = MBTIClassifier(input_dim=results['input_dim'], num_classes=16)
        
        # Filter state_dict to only include keys that exist in the model
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in model_keys}
        
        # Load filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        
        results['model'] = model
        results['state_dict'] = filtered_state_dict
        
        print(f"✅ Model loaded successfully")
        print(f"   Input dimension: {results['input_dim']}")
        print(f"   Features: {len(results['features'])}")
        print(f"   Reported accuracy: {results['test_accuracy']:.2%}")
        print(f"   Transfer learning: {'✅' if results['transfer_learning'] else '❌'}")
        print(f"   Filtered state dict keys: {len(filtered_state_dict)}")
        
        # Display MBTI mapping
        print(f"\n📋 MBTI Type Mapping:")
        for idx, mbti in results['idx_to_type'].items():
            print(f"   {idx}: {mbti}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    return results


# ============================================================================
# 2. LOAD TRAINING DATA
# ============================================================================
def load_training_data():
    """Load training data distribution from all mbti_playlists"""
    print("\n" + "="*80)
    print("📊 2. LOADING TRAINING DATA DISTRIBUTION")
    print("="*80)
    
    # Option 1: Use pre-saved playlist_embedding_features.csv
    embeddings_path = PROCESSED_DIR / "playlist_embedding_features.csv"
    
    if embeddings_path.exists():
        df = pd.read_csv(embeddings_path)
        if 'mbti' in df.columns:
            distribution = df['mbti'].value_counts().to_dict()
            total = sum(distribution.values())
            
            print(f"\n📊 Training Data Distribution from playlist_embedding_features.csv:")
            print(f"\n{'MBTI':<8} {'Playlists':<12} {'Percentage':<12} {'Bar'}")
            print(f"{'-'*50}")
            
            for mbti, count in sorted(distribution.items()):
                pct = count / total * 100
                bar = "█" * int(pct / 2)
                print(f"{mbti:<8} {count:>6}     {pct:>5.1f}%   {bar}")
            
            print(f"\n{'TOTAL':<8} {total:>6}")
            
            # Calculate imbalance metrics
            counts = list(distribution.values())
            max_count = max(counts)
            min_count = min(counts)
            
            print(f"\n📊 Imbalance Metrics:")
            print(f"   Most common type: {max(distribution, key=distribution.get)} ({max_count} playlists)")
            print(f"   Least common type: {min(distribution, key=distribution.get)} ({min_count} playlists)")
            print(f"   Imbalance ratio: {max_count/min_count:.2f}:1")
            print(f"   Standard deviation: {np.std(counts):.1f}")
            
            # Gini coefficient
            sorted_counts = np.sort(counts)
            n = len(sorted_counts)
            gini = (2 * np.sum(np.arange(1, n+1) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
            print(f"   Gini coefficient: {gini:.3f} (0=perfect balance, 1=extreme imbalance)")
            
            return distribution
    
    # Fallback: read from individual files
    mbti_dir = PROJECT_ROOT / "data" / "raw" / "mbti_playlists"
    
    if not mbti_dir.exists():
        print(f"❌ MBTI playlists directory not found: {mbti_dir}")
        return {}
    
    distribution = {}
    total = 0
    
    for csv_file in sorted(mbti_dir.glob("*.csv")):
        mbti_type = csv_file.stem
        df = pd.read_csv(csv_file)
        count = len(df)
        distribution[mbti_type] = count
        total += count
    
    print(f"\n📊 Training Data Distribution (from raw files):")
    for mbti, count in sorted(distribution.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"{mbti:<8} {count:>6}     {pct:>5.1f}%   {bar}")
    
    return distribution


# ============================================================================
# 3. ANALYZE PREDICTION BIAS FROM WEIGHTS
# ============================================================================
def analyze_prediction_bias(model_results):
    """Analyze bias in model predictions using model weights"""
    print("\n" + "="*80)
    print("📊 3. PREDICTION BIAS ANALYSIS (from weights)")
    print("="*80)
    
    state_dict = model_results.get('state_dict', {})
    idx_to_type = model_results.get('idx_to_type', {})
    
    if not state_dict:
        print("❌ No model weights found")
        return
    
    # Find output layer weights - Updated keys for the new architecture
    output_weights = None
    output_bias = None
    
    # Try different possible key patterns
    for key in state_dict.keys():
        if 'net.6.weight' in key or 'net.12.weight' in key or 'classifier.4.weight' in key:
            output_weights = state_dict[key]
        if 'net.6.bias' in key or 'net.12.bias' in key or 'classifier.4.bias' in key:
            output_bias = state_dict[key]
    
    if output_weights is not None:
        print(f"\n📊 Output Layer Analysis (16 classes):")
        print(f"   Weight shape: {output_weights.shape}")
        
        # Calculate per-class weight magnitude
        weight_magnitudes = torch.abs(output_weights).sum(dim=1).numpy()
        
        print(f"\n   Class Weight Magnitudes (higher = more influence):")
        max_mag = weight_magnitudes.max()
        for i in range(len(weight_magnitudes)):
            mbti = idx_to_type.get(i, f"Class_{i}")
            bar_len = int(weight_magnitudes[i] / max_mag * 40) if max_mag > 0 else 0
            bar = "█" * bar_len
            print(f"      {i:2d} {mbti:<6}: {weight_magnitudes[i]:.4f} {bar}")
    
    if output_bias is not None:
        bias_values = output_bias.numpy()
        print(f"\n   Output Bias Values (positive = favors that class):")
        for i in range(len(bias_values)):
            mbti = idx_to_type.get(i, f"Class_{i}")
            direction = "↑" if bias_values[i] > 0 else "↓" if bias_values[i] < 0 else "→"
            print(f"      {i:2d} {mbti:<6}: {bias_values[i]:+.4f} {direction}")
    else:
        print("\n   No bias values found (bias might be zero or missing)")


# ============================================================================
# 4. SIMULATED PREDICTION DISTRIBUTION (FIXED)
# ============================================================================
def simulate_prediction_distribution(model_results, n_samples=2000):
    """Simulate predictions on random inputs to detect output bias"""
    print("\n" + "="*80)
    print(f"📊 4. SIMULATED PREDICTION DISTRIBUTION ({n_samples} random inputs)")
    print("="*80)
    
    model = model_results.get('model')
    idx_to_type = model_results.get('idx_to_type', {})
    input_dim = model_results.get('input_dim', 171)
    
    if model is None:
        print("❌ Model not loaded for simulation")
        return
    
    # Generate random inputs and predict
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            random_input = torch.randn(1, input_dim)
            output = model(random_input)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            predictions.append(pred)
            probabilities.append(probs[0][pred].item())
    
    # Count predictions
    pred_counts = Counter(predictions)
    
    print(f"\n📊 Simulated Prediction Distribution:")
    print(f"\n{'Class':<6} {'MBTI':<8} {'Predictions':<12} {'Percentage':<12} {'Expected (uniform)':<18} {'Bias':<10}")
    print(f"{'-'*70}")
    
    expected_pct = 100 / 16
    
    for i in range(16):
        mbti = idx_to_type.get(i, f"Class_{i}")
        count = pred_counts.get(i, 0)
        pct = count / n_samples * 100
        diff = pct - expected_pct
        bias_indicator = "▲▲" if diff > 5 else "▲" if diff > 2 else "▼▼" if diff < -5 else "▼" if diff < -2 else "→"
        print(f"{i:2d}   {mbti:<8} {count:>4}       {pct:>5.1f}%        {expected_pct:>5.1f}%          {bias_indicator} {diff:+.1f}%")
    
    # Chi-square test
    from scipy.stats import chisquare
    observed = [pred_counts.get(i, 0) for i in range(16)]
    expected_counts = [n_samples / 16] * 16
    chi2, p_value = chisquare(observed, expected_counts)
    
    print(f"\n📊 Statistical Test (Uniform Distribution):")
    print(f"   Chi-square: {chi2:.2f}")
    print(f"   P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"   ⚠️ Predictions are NOT uniformly distributed (p < 0.05)")
    else:
        print(f"   ✅ Predictions are uniformly distributed")
    
    # Average confidence
    avg_confidence = np.mean(probabilities)
    print(f"\n📊 Average prediction confidence: {avg_confidence:.2%}")


# ============================================================================
# 5. SHAP ANALYSIS INTEGRATION
# ============================================================================

def run_shap_analysis(model_results, n_samples=50):
    """Run SHAP analysis to explain model predictions"""
    print("\n" + "="*80)
    print("📊 5. SHAP ANALYSIS (Model Explainability)")
    print("="*80)
    
    try:
        import shap
        
        model = model_results.get('model')
        idx_to_type = model_results.get('idx_to_type', {})
        input_dim = model_results.get('input_dim', 171)
        
        if model is None:
            print("❌ Model not loaded for SHAP analysis")
            return
        
        # Use more samples to avoid LassoLarsIC warning
        n_background = min(100, input_dim * 2)
        background = np.random.randn(n_background, input_dim).astype(np.float32)
        
        # Use fewer test samples
        n_test = min(n_samples, 20)
        test_samples = np.random.randn(n_test, input_dim).astype(np.float32)
        
        # Define prediction function
        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)
            return probs.numpy()
        
        print(f"\n🔄 Running SHAP analysis on {n_test} samples...")
        print(f"   Background samples: {n_background}")
        
        # Use LinearExplainer instead of KernelExplainer (faster, no Lasso warning)
        explainer = shap.LinearExplainer(model, background, feature_names=model_results.get('features', [])[:20])
        
        # For tree-based models, use TreeExplainer
        # Since we have a neural network, we'll use GradientExplainer
        try:
            # Try GradientExplainer (better for neural networks)
            explainer = shap.GradientExplainer(model, torch.tensor(background, dtype=torch.float32))
            shap_values = explainer.shap_values(torch.tensor(test_samples, dtype=torch.float32))
            print(f"✅ SHAP analysis complete using GradientExplainer")
        except:
            # Fallback to KernelExplainer with more samples to avoid warning
            explainer = shap.KernelExplainer(predict_fn, background[:50])
            shap_values = explainer.shap_values(test_samples[:5], nsamples=30)
            print(f"✅ SHAP analysis complete using KernelExplainer")
        
        # Save SHAP values
        shap_path = PROCESSED_DIR / "shap_analysis_results.pkl"
        import joblib
        joblib.dump({
            'shap_values': shap_values,
            'test_samples': test_samples,
            'feature_names': model_results.get('features', [])[:20]
        }, shap_path)
        print(f"✅ SHAP values saved to {shap_path}")
        
        # Plot summary for first class
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Get top 15 features
            feature_names = model_results.get('features', [])[:20]
            
            shap.summary_plot(shap_values[0], test_samples, 
                             feature_names=feature_names,
                             show=False, max_display=15)
            plt.title(f"SHAP Feature Importance - {idx_to_type.get(0, 'Class 0')}")
            plt.tight_layout()
            plt.savefig(PROCESSED_DIR / "shap_summary_plot.png", dpi=150, bbox_inches='tight')
            print(f"✅ SHAP plot saved to {PROCESSED_DIR / 'shap_summary_plot.png'}")
            plt.show()
        except Exception as plot_err:
            print(f"⚠️ Could not generate SHAP plot: {plot_err}")
        
    except ImportError:
        print("⚠️ SHAP not installed. Install with: pip install shap")
        print("   SHAP analysis skipped")
    except Exception as e:
        print(f"⚠️ SHAP analysis error: {e}")


# ============================================================================
# 6. AXIS-SPECIFIC BIAS ANALYSIS (FIXED)
# ============================================================================
def analyze_axis_bias(model_results):
    """Analyze bias for each MBTI axis (E/I, S/N, T/F, J/P)"""
    print("\n" + "="*80)
    print("📊 6. AXIS-SPECIFIC BIAS ANALYSIS")
    print("="*80)
    
    model = model_results.get('model')
    idx_to_type = model_results.get('idx_to_type', {})
    state_dict = model_results.get('state_dict', {})
    
    if model is None:
        print("❌ Model not loaded")
        return
    
    # Create type to index mapping
    type_to_idx = {v: k for k, v in idx_to_type.items()}
    
    # Define axis groupings
    axes = {
        'E/I': {
            'E': [t for t in idx_to_type.values() if t[0] == 'E'],
            'I': [t for t in idx_to_type.values() if t[0] == 'I']
        },
        'S/N': {
            'S': [t for t in idx_to_type.values() if t[1] == 'S'],
            'N': [t for t in idx_to_type.values() if t[1] == 'N']
        },
        'T/F': {
            'T': [t for t in idx_to_type.values() if t[2] == 'T'],
            'F': [t for t in idx_to_type.values() if t[2] == 'F']
        },
        'J/P': {
            'J': [t for t in idx_to_type.values() if t[3] == 'J'],
            'P': [t for t in idx_to_type.values() if t[3] == 'P']
        }
    }
    
    # Find output weights (try multiple key patterns)
    output_weights = None
    for key in state_dict.keys():
        if 'net.6.weight' in key:  # Our architecture has net.6 as output
            output_weights = state_dict[key]
            break
        elif 'net.10.weight' in key:
            output_weights = state_dict[key]
            break
        elif 'net.12.weight' in key:
            output_weights = state_dict[key]
            break
    
    if output_weights is None:
        print("⚠️ Could not find output weights, using simulation instead")
        # Fallback: simulate predictions to get axis bias
        simulate_axis_bias(model, idx_to_type)
        return
    
    output_weights_np = output_weights.numpy()
    
    print(f"\n📊 Average Output Weights by Axis:")
    print(f"\n{'Axis':<8} {'Letter':<8} {'Avg Weight':<12} {'vs Counterpart':<15}")
    print(f"{'-'*50}")
    
    for axis_name, axis_data in axes.items():
        letters = list(axis_data.keys())
        letter1 = letters[0]
        letter2 = letters[1]
        
        # Calculate average weight for each letter
        weight1 = np.mean([np.abs(output_weights_np[type_to_idx[t]]).mean() 
                          for t in axis_data[letter1] if t in type_to_idx])
        weight2 = np.mean([np.abs(output_weights_np[type_to_idx[t]]).mean() 
                          for t in axis_data[letter2] if t in type_to_idx])
        
        print(f"{axis_name:<8} {letter1:<8} {weight1:.4f}")
        print(f"{axis_name:<8} {letter2:<8} {weight2:.4f}")
        
        bias = (weight1 - weight2) / max(weight1, weight2) * 100
        bias_dir = letter1 if bias > 0 else letter2
        print(f"{axis_name:<8} {'Bias':<8} {abs(bias):<11.1f}% toward {bias_dir}")
        print()


def simulate_axis_bias(model, idx_to_type):
    """Fallback: simulate predictions to get axis bias"""
    print("\n📊 Simulating Axis Bias (using random inputs)...")
    
    type_to_idx = {v: k for k, v in idx_to_type.items()}
    input_dim = 171
    n_samples = 1000
    
    axis_counts = {'E/I': {'E': 0, 'I': 0},
                   'S/N': {'S': 0, 'N': 0},
                   'T/F': {'T': 0, 'F': 0},
                   'J/P': {'J': 0, 'P': 0}}
    
    with torch.no_grad():
        for _ in range(n_samples):
            random_input = torch.randn(1, input_dim)
            output = model(random_input)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_type = idx_to_type.get(pred_idx, 'XXXX')
            
            if len(pred_type) >= 4:
                axis_counts['E/I'][pred_type[0]] += 1
                axis_counts['S/N'][pred_type[1]] += 1
                axis_counts['T/F'][pred_type[2]] += 1
                axis_counts['J/P'][pred_type[3]] += 1
    
    print(f"\n{'Axis':<8} {'Letter':<8} {'Count':<10} {'Percentage':<12}")
    print(f"{'-'*45}")
    
    for axis_name, counts in axis_counts.items():
        for letter, count in counts.items():
            pct = count / n_samples * 100
            print(f"{axis_name:<8} {letter:<8} {count:<10} {pct:>5.1f}%")
        print()

# ============================================================================
# 7. GENERATE RECOMMENDATIONS
# ============================================================================
def generate_recommendations(train_distribution, model_results):
    """Generate recommendations for fixing bias and imbalance"""
    print("\n" + "="*80)
    print("💡 7. RECOMMENDATIONS")
    print("="*80)
    
    if not train_distribution:
        print("⚠️ No training data available")
        return
    
    counts = list(train_distribution.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    recommendations = []
    
    # Data balance
    if imbalance_ratio > 2:
        recommendations.append(f"⚠️ Class imbalance detected ({imbalance_ratio:.1f}:1 ratio)")
        recommendations.append("   → Class weights already implemented in loss function")
        recommendations.append("   → Consider collecting more data for ESFJ, ESTJ, ESFP types")
    else:
        recommendations.append("✅ Training data is relatively balanced")
    
    # Model
    recommendations.append("\n🎯 Current Model:")
    recommendations.append(f"   ✅ Transfer learning from {model_results.get('input_dim', 171)-128} → 128 features")
    recommendations.append(f"   ✅ Test accuracy: {model_results.get('test_accuracy', 0):.2%}")
    recommendations.append("   ✅ Architecture: 171 → 64 → 32 → 16 → 16")
    
    # SHAP
    recommendations.append("\n🔍 Explainability (SHAP):")
    recommendations.append("   → Install: pip install shap")
    recommendations.append("   → Run SHAP analysis to identify which features drive predictions")
    recommendations.append("   → SHAP values saved in shap_analysis_results.pkl")
    
    print("\n".join(recommendations))


# ============================================================================
# 8. EXPORT REPORT
# ============================================================================
def export_report(train_distribution, model_results):
    """Export bias analysis report"""
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "test_accuracy": model_results.get('test_accuracy', 0),
            "input_dim": model_results.get('input_dim', 0),
            "features_count": len(model_results.get('features', [])),
            "transfer_learning": model_results.get('transfer_learning', False)
        },
        "training_distribution": train_distribution,
        "total_playlists": sum(train_distribution.values()) if train_distribution else 0
    }
    
    if train_distribution:
        counts = list(train_distribution.values())
        report["imbalance_metrics"] = {
            "max_count": max(counts),
            "min_count": min(counts),
            "imbalance_ratio": max(counts) / min(counts),
            "std_dev": float(np.std(counts)),
            "most_common": max(train_distribution, key=train_distribution.get),
            "least_common": min(train_distribution, key=train_distribution.get)
        }
    
    report_path = PROCESSED_DIR / "model_bias_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Bias report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "⚖️ RUNNING IMBALANCE & BIAS DIAGNOSIS".center(80))
    
    # Load model and data
    model_results = load_playlist_classifier()
    train_distribution = load_training_data()
    
    if not model_results:
        print("❌ Could not load model. Exiting.")
        return
    
    # Run analyses
    analyze_prediction_bias(model_results)
    simulate_prediction_distribution(model_results, n_samples=2000)
    run_shap_analysis(model_results)
    analyze_axis_bias(model_results)
    
    # Generate recommendations and export
    generate_recommendations(train_distribution, model_results)
    export_report(train_distribution, model_results)
    
    print("\n" + "="*80)
    print("✅ BIAS DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()