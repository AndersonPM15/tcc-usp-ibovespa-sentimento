import nbformat as nbf
from pathlib import Path
nb=nbf.read(Path("notebooks/17_sentiment_validation.ipynb").open("r",encoding="utf-8"),as_version=4)
cell = nb.cells[6]
print(cell.source)
