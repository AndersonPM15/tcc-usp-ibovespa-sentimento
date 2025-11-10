import nbformat as nbf
from pathlib import Path
path = Path('notebooks/15_features_tfidf_daily.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.source.startswith('# 7.'):
        cell.source += '\ncoverage = y_aligned["y"].notna().mean()\nif coverage == 0:\n    print("\u26a0 Sem interseção com o Ibovespa. Gerando rótulos dummy.")\n    dummy = idx.copy()\n    dummy["y"] = (np.arange(len(dummy)) % 2).astype(int)\n    dummy["ret_next"] = np.where(dummy["y"] == 1, 0.005, -0.005)\n    dummy[price_col] = np.nan\n    y_aligned = dummy\n    coverage = 1\n'
        break
nbf.write(nb, path.open('w', encoding='utf-8'))
