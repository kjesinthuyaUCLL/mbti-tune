import json

try:
    with open('notebooks/MBTI_Playlist_Training.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # skip magic commands
            source = '\n'.join([line for line in source.split('\n') if not line.strip().startswith('%') and not line.strip().startswith('!')])
            try:
                compile(source, f'cell_{i}', 'exec')
            except SyntaxError as e:
                print(f'SyntaxError in cell {i}: {e}')
                exit(1)
    print('All code cells have valid Python syntax!')
except Exception as e:
    print(f"Error: {e}")
