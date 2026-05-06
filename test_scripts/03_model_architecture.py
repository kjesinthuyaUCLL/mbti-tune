"""
Inspect the detailed architecture of the song classifier
Place this in: test_scripts/03_model_architecture.py
Run with: python test_scripts/03_model_architecture.py
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_model_architecture():
    """Detailed inspection of the model architecture"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "song_classifier.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    if isinstance(model, dict):
        print("\n📦 Model is a state dictionary")
        print("\n🔍 Layer Analysis:")
        
        # Group layers by network block
        layers = {}
        for key in model.keys():
            parts = key.split('.')
            if len(parts) >= 2:
                block = '.'.join(parts[:2])
                if block not in layers:
                    layers[block] = []
                layers[block].append(key)
        
        for block_name, keys in sorted(layers.items()):
            print(f"\n  {block_name}:")
            for key in sorted(keys):
                tensor = model[key]
                print(f"    - {key}: shape {tuple(tensor.shape)}")
        
        # Try to reconstruct network architecture
        print("\n🏗️ Attempting to reconstruct network architecture:")
        
        input_dim = None
        output_dim = None
        layer_dims = []
        
        for key in sorted(model.keys()):
            if 'weight' in key and 'running' not in key:
                if 'network.0.weight' in key:
                    input_dim = model[key].shape[1]
                    layer_dims.append(model[key].shape[0])
                elif 'network.1.weight' in key:
                    layer_dims.append(model[key].shape[0])
                elif 'network.4.weight' in key:
                    layer_dims.append(model[key].shape[0])
                elif 'network.5.weight' in key:
                    layer_dims.append(model[key].shape[0])
                elif 'network.8.weight' in key:
                    layer_dims.append(model[key].shape[0])
                elif 'network.9.weight' in key:
                    layer_dims.append(model[key].shape[0])
                elif 'network.11.weight' in key:
                    output_dim = model[key].shape[0]
                    layer_dims.append(model[key].shape[0])
        
        print(f"\n  Input dimension: {input_dim}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Layer dimensions: {layer_dims}")
        
        # Check for BatchNorm layers
        bn_layers = [k for k in model.keys() if 'running_mean' in k]
        print(f"\n  BatchNorm layers: {len(bn_layers)}")
        
    elif hasattr(model, 'parameters'):
        print("\n📦 Model is a full nn.Module")
        print(f"\n🔍 Model structure:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Parameter count:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

def check_with_src_model():
    """Check if the model matches src.model.MBTIPredictor"""
    print("\n" + "="*60)
    print("CHECKING AGAINST SRC.MODEL")
    print("="*60)
    
    try:
        from src.model import MBTIPredictor
        print("✅ Successfully imported MBTIPredictor from src.model")
        
        # Create a default model to see expected architecture
        default_model = MBTIPredictor()
        print(f"\nExpected architecture:")
        print(f"  Input dimension: {default_model.input_dim}")
        print(f"  Output dimension: {default_model.output_dim}")
        print(f"\nDefault model structure:")
        print(default_model)
        
        # Try to load the state dict into MBTIPredictor
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "song_classifier.pt")
        
        state_dict = torch.load(model_path, map_location='cpu')
        
        if isinstance(state_dict, dict):
            # Check if keys match
            model_keys = set(state_dict.keys())
            expected_keys = set(default_model.state_dict().keys())
            
            missing_keys = expected_keys - model_keys
            unexpected_keys = model_keys - expected_keys
            
            if missing_keys:
                print(f"\n⚠️ Missing keys in loaded model: {missing_keys}")
            if unexpected_keys:
                print(f"\n⚠️ Unexpected keys in loaded model: {unexpected_keys}")
            
            if not missing_keys and not unexpected_keys:
                print("\n✅ Model keys match perfectly!")
            else:
                print("\n⚠️ Model keys don't match exactly - may need to adapt loading")
                
    except ImportError as e:
        print(f"❌ Could not import src.model: {e}")
    except Exception as e:
        print(f"❌ Error checking against src.model: {e}")

def main():
    print("\n" + "🔬 MODEL ARCHITECTURE ANALYSIS".center(60))
    inspect_model_architecture()
    check_with_src_model()

if __name__ == "__main__":
    main()