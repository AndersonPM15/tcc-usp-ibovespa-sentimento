import nbformat as nbf
from pathlib import Path
path = Path('notebooks/16_models_tfidf_baselines.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(i, cell.source.split('\n')[0])
