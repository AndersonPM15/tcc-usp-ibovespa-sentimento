"""
Constantes globais do projeto: período oficial do estudo.

Os parâmetros dos modelos (TF-IDF, classificadores e walk-forward) ficam junto do código
que os usa (notebooks 15 e 16) e estão documentados no README.
"""

from datetime import date

START_DATE = date(2018, 1, 2)  # primeiro pregão de 2018
END_DATE = date(2024, 12, 31)  # limite do período analisado no artigo

START_DATE_STR = START_DATE.isoformat()
END_DATE_STR = END_DATE.isoformat()
