import nbformat as nbf
from pathlib import Path
path = Path('notebooks/15_features_tfidf_daily.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
for cell in nb.cells:
    cell.pop('id', None)
nbf.write(nb, path.open('w', encoding='utf-8'))
