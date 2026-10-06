"""
Higiene do repositório: os notebooks são versionados sem saídas nem metadados de execução.

As saídas guardavam caminhos pessoais, domínios e manchetes de terceiros, e os metadados
do Colab guardavam nome e ID de usuário. Rode os notebooks localmente e limpe as saídas
antes de commitar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
EXECUTION_METADATA = {"colab", "executionInfo", "outputId"}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_have_no_outputs_or_execution_metadata(notebook: Path) -> None:
    content = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(content["cells"]):
        assert not cell.get("outputs"), f"célula {index} com saída"
        assert cell.get("execution_count") is None, f"célula {index} com contador de execução"
        assert not EXECUTION_METADATA & set(cell.get("metadata", {})), f"célula {index}"
    assert "colab" not in content.get("metadata", {})
