import nbformat as nbf
from pathlib import Path
path = Path('notebooks/15_features_tfidf_daily.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
cell0 = nb.cells[0]
if 'import json' not in cell0.source:
    cell0.source = cell0.source.replace('from pathlib import Path', 'from pathlib import Path\nimport json')
    nbf.write(nb, path.open('w', encoding='utf-8'))
    print('Added json import to notebook 15')
