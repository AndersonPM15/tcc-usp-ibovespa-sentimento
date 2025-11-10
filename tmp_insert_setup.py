import nbformat as nbf
from pathlib import Path

setup_src = """# 1. Setup de caminhos locais
import os
from datetime import datetime
from pathlib import Path

from src.io import paths
from src.config import loader as cfg

DATA_PATHS = paths.get_data_paths()
PROJECT_PATHS = path_utils.get_project_paths() if 'path_utils' in globals() else paths.get_project_paths()
BASE_PATH = str(DATA_PATHS["base"])
RAW_PATH = str(DATA_PATHS["data_raw"])
PROC_PATH = str(DATA_PATHS["data_processed"])
INTERIM_PATH = str(DATA_PATHS["data_interim"])
CONFIG = cfg.load_config()

NB_NAME = "17_sentiment_validation"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print("BASE_PATH:", BASE_PATH)
print("PROC_PATH:", PROC_PATH)
"""
path = Path('notebooks/17_sentiment_validation.ipynb')
nb = nbf.read(path.open('r', encoding='utf-8'), as_version=4)
nb.cells.insert(0, nbf.v4.new_code_cell(setup_src))
nbf.write(nb, path.open('w', encoding='utf-8'))
