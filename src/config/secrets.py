"""
Leitura de segredos (chaves de API) a partir de variáveis de ambiente.

Os valores ficam no arquivo `.env` na raiz do projeto, que não é versionado
(ver `.gitignore`). Use `.env.example` como modelo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def get_secret(name: str) -> str:
    """Retorna o valor da variável `name`, carregando antes o `.env` da raiz, se existir.

    Variáveis já definidas no ambiente têm prioridade sobre o `.env`.
    Falha com mensagem clara se a variável estiver ausente ou vazia.
    """
    load_dotenv(ENV_FILE, override=False)
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Variável de ambiente {name} não definida. "
            f"Copie .env.example para .env e preencha {name}."
        )
    return value
