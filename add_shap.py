import json

file_path = 'c:/Users/jobra/Desktop/UNI SECONDO ANNO/Advanced AI/mbti-tune/notebooks/MBTI_Playlist_Training.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

shap_code = """# --- ADDED: Day 4 SHAP Explainability ---
# Added SHAP explainer to understand feature importance for the 4 MBTI dimensions.

import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("="*60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*60)

# 1. Initialize the explainer with a background dataset (subset of training data to save time)
model.eval()
background = X_train[:100].to(device)
test_samples = X_test[:50].to(device)

# DeepExplainer is optimized for PyTorch models
explainer = shap.DeepExplainer(model, background)

# 2. Calculate SHAP values for the test samples
print("Calculating SHAP values... (this might take a minute)")
shap_values = explainer.shap_values(test_samples)

# 3. Visualizations
# SHAP values for PyTorch multi-output regression comes as a list of arrays (one for each output dimension)

for i, name in enumerate(dim_names):
    print(f"\\n{'='*40}")
    print(f"Feature Importance for {name}")
    print(f"{'='*40}")
    
    # We use summary_plot to show the global importance of features
    shap.summary_plot(
        shap_values[i], 
        test_samples.cpu().numpy(), 
        feature_names=feature_cols, 
        plot_type="bar", 
        show=False
    )
    plt.title(f"SHAP Feature Importance: {name}")
    plt.tight_layout()
    plt.show()

print("\\n✅ SHAP Analysis Complete")
"""

# Check if SHAP cell already exists to avoid duplicates
already_exists = any('import shap' in ''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code')

if not already_exists:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\\n' for line in shap_code.split('\\n')]
    }
    # Remove the very last newline from the last string to keep format clean
    new_cell['source'][-1] = new_cell['source'][-1].strip('\\n')
    
    nb['cells'].append(new_cell)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\\n')

    print("SHAP cell successfully appended to the notebook.")
else:
    print("SHAP cell already exists. No changes made.")
