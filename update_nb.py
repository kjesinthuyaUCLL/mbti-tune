import json

file_path = 'c:/Users/jobra/Desktop/UNI SECONDO ANNO/Advanced AI/mbti-tune/notebooks/MBTI_Playlist_Training.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score' in line:
                new_source.append('# from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score\n')
                new_source.append('from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score # added: classification metrics requirement\n')
            else:
                new_source.append(line)
        
        if 'print(f"Accuracy (within 10%): {accuracy_within_10:.2%}")' in ''.join(new_source):
            idx = -1
            for i, line in enumerate(new_source):
                if 'print(f"Accuracy (within 10%): {accuracy_within_10:.2%}")' in line:
                    idx = i
                    break
            
            if idx != -1:
                added_code = """
# --- OLD CODE COMMENTED OUT ---
# # (No specific classification metrics were calculated here before)

# --- ADDED: Strict grading requirement for imbalanced datasets ---
print(f"\\n{'='*60}")
print("CLASSIFICATION METRICS (Threshold = 0.5) - added")
print(f"{'='*60}")
# Convert continuous percentages to binary classes (0 or 1) using 0.5 threshold
binary_preds = (predictions > 0.5).astype(int)
binary_targets = (targets > 0.5).astype(int)

for i, (name, letter) in enumerate(zip(dim_names, dim_letters)):
    acc = accuracy_score(binary_targets[:, i], binary_preds[:, i])
    prec = precision_score(binary_targets[:, i], binary_preds[:, i], zero_division=0)
    rec = recall_score(binary_targets[:, i], binary_preds[:, i], zero_division=0)
    f1 = f1_score(binary_targets[:, i], binary_preds[:, i], zero_division=0)
    
    print(f"\\n{name} ({letter}):")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

# Overall macro-averaged metrics across all 4 dimensions
macro_acc = accuracy_score(binary_targets.flatten(), binary_preds.flatten())
macro_prec = precision_score(binary_targets.flatten(), binary_preds.flatten(), zero_division=0)
macro_rec = recall_score(binary_targets.flatten(), binary_preds.flatten(), zero_division=0)
macro_f1 = f1_score(binary_targets.flatten(), binary_preds.flatten(), zero_division=0)

print(f"\\n{'='*60}")
print("MACRO-AVERAGED CLASSIFICATION METRICS - added")
print(f"{'='*60}")
print(f"Macro Accuracy:  {macro_acc:.3f}")
print(f"Macro Precision: {macro_prec:.3f}")
print(f"Macro Recall:    {macro_rec:.3f}")
print(f"Macro F1-Score:  {macro_f1:.3f}\\n")
"""
                added_lines = [line + '\n' for line in added_code.strip('\n').split('\n')]
                new_source = new_source[:idx+1] + ['\n'] + added_lines + new_source[idx+1:]
                
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Notebook modified successfully.")
