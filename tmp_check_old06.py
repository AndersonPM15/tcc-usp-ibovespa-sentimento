import nbformat as nbf
from pathlib import Path
path = Path('notebooks/_runs/20251110_161730_06_preprocessing_real.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code' and '!pip install -q sentence-transformers' in cell.source:
        print(cell.source)
        break
