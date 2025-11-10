import nbformat as nbf
from pathlib import Path
nb=nbf.read(Path("notebooks/17_sentiment_validation.ipynb").open("r",encoding="utf-8"),as_version=4)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code':
        line=cell.source.split('\n')[0]
        print(i, line.encode('unicode_escape').decode())
