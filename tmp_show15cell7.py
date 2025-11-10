import nbformat as nbf
from pathlib import Path
path = Path('notebooks/15_features_tfidf_daily.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.source.startswith('# 7.'):
        print(cell.source.encode('unicode_escape').decode())
        break
