import nbformat as nbf
from pathlib import Path
nb=nbf.read(Path("notebooks/17_sentiment_validation.ipynb").open("r",encoding="utf-8"),as_version=4)
for i,cell in enumerate(nb.cells[:6]):
    if cell.cell_type=='code':
        print(f"Cell {i}:")
        print(cell.source)
        print('-----')
