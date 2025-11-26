"""
Pipeline Completo - TCC USP: Análise de Sentimento x Ibovespa
==============================================================

Este script orquestra a execução sequencial do pipeline multisource do TCC,
processando dados de notícias financeiras desde a coleta até a geração de
features para modelagem.

Fluxo de execução:
------------------
    1. ETL & Deduplicação (13_etl_dedup.ipynb)
       → Consolida fontes, remove duplicatas, padroniza schema
    
    2. Pré-processamento PT-BR (14_preprocess_ptbr.ipynb)
       → Limpeza textual, normalização, remoção de stopwords
    
    3. Features TF-IDF Diárias (15_features_tfidf_daily.ipynb)
       → Vetorização TF-IDF, agregação diária, geração de matriz

Arquivos de saída:
------------------
    - data_interim/news_clean_multisource.parquet (intermediário)
    - data_processed/news_clean.parquet (notícias limpas)
    - data_processed/bow_daily.parquet (bag-of-words diário)
    - data_processed/tfidf_daily_matrix.npz (matriz TF-IDF)
    - data_processed/tfidf_daily_index.csv (índice de datas)
    - data_processed/tfidf_daily_vocab.json (vocabulário)

Uso:
----
    python run_pipeline_complete.py

Requisitos:
-----------
    - Jupyter instalado (jupyter nbconvert)
    - Dados brutos em data_raw/news_multisource.parquet
    - Ambiente virtual ativado com dependências instaladas

Autor: TCC USP - Sentimento x Ibovespa
Data: 2025
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuração dos notebooks do pipeline
# ---------------------------------------------------------------------------
# Cada tupla contém: (nome_arquivo, descrição, etapa_número)
NOTEBOOKS = [
    ("13_etl_dedup.ipynb", "ETL & Deduplicação", 1),
    ("14_preprocess_ptbr.ipynb", "Pré-processamento PT-BR", 2),
    ("15_features_tfidf_daily.ipynb", "Features TF-IDF Diárias", 3),
]

# Timeout máximo por notebook (em segundos)
NOTEBOOK_TIMEOUT = 3600  # 1 hora


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def print_header(title: str, char: str = "=", width: int = 80) -> None:
    """Imprime um cabeçalho formatado para melhor visualização."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_step_start(step_num: int, description: str, filename: str) -> None:
    """Imprime mensagem de início de etapa."""
    print_header(f"ETAPA {step_num}/3: {description}")
    print(f"📂 Notebook: {filename}")
    print(f"⏱️  Início: {datetime.now().strftime('%H:%M:%S')}")


def print_step_end(step_num: int, description: str, success: bool, duration: float) -> None:
    """Imprime mensagem de fim de etapa com status."""
    status = "✅ SUCESSO" if success else "❌ ERRO"
    print(f"\n{status} | Etapa {step_num}: {description}")
    print(f"⏱️  Duração: {duration:.1f} segundos")


def run_notebook(notebook_path: Path, description: str, step_num: int) -> bool:
    """
    Executa um notebook Jupyter usando nbconvert.
    
    Parameters
    ----------
    notebook_path : Path
        Caminho completo para o arquivo .ipynb
    description : str
        Descrição da etapa para exibição
    step_num : int
        Número da etapa no pipeline
    
    Returns
    -------
    bool
        True se executou com sucesso, False caso contrário
    """
    print_step_start(step_num, description, notebook_path.name)
    
    start_time = datetime.now()
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={NOTEBOOK_TIMEOUT}",
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = (datetime.now() - start_time).total_seconds()
        print_step_end(step_num, description, success=True, duration=duration)
        return True
        
    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        print_step_end(step_num, description, success=False, duration=duration)
        print(f"\n🔴 Detalhes do erro:")
        print(f"{'-' * 40}")
        if e.stderr:
            print(e.stderr[:500])  # Limita output de erro
        return False


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main():
    """
    Orquestra a execução sequencial dos notebooks do pipeline.
    
    O pipeline é interrompido imediatamente se qualquer etapa falhar,
    garantindo que erros sejam tratados antes de prosseguir.
    """
    base_path = Path(__file__).parent / "notebooks"
    pipeline_start = datetime.now()
    
    # -----------------------------------------------------------------------
    # Cabeçalho do pipeline
    # -----------------------------------------------------------------------
    print_header("🚀 PIPELINE TCC - REGENERAÇÃO COMPLETA")
    print(f"📅 Data/Hora: {pipeline_start.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"📁 Diretório: {base_path}")
    print(f"📊 Base: GDELT histórica (2.771 dias)")
    print(f"📋 Etapas: {len(NOTEBOOKS)} notebooks")
    
    # -----------------------------------------------------------------------
    # Execução sequencial das etapas
    # -----------------------------------------------------------------------
    results = []
    
    for notebook_file, description, step_num in NOTEBOOKS:
        notebook_path = base_path / notebook_file
        
        # Verifica se o arquivo existe
        if not notebook_path.exists():
            print(f"\n⚠️  AVISO: Arquivo não encontrado!")
            print(f"   Esperado: {notebook_path}")
            results.append(False)
            continue
        
        # Executa o notebook
        success = run_notebook(notebook_path, description, step_num)
        results.append(success)
        
        # Interrompe pipeline em caso de erro
        if not success:
            print_header("❌ PIPELINE INTERROMPIDO", char="!")
            print("Corrija o erro acima antes de continuar.")
            sys.exit(1)
    
    # -----------------------------------------------------------------------
    # Resumo final do pipeline
    # -----------------------------------------------------------------------
    pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
    
    print_header("📊 RESUMO DO PIPELINE")
    
    if all(results):
        print("✅ STATUS: TODAS AS ETAPAS CONCLUÍDAS COM SUCESSO")
        print(f"⏱️  Duração total: {pipeline_duration:.1f} segundos ({pipeline_duration/60:.1f} min)")
        
        print("\n📁 Arquivos gerados:")
        print("   ┌─ data_interim/")
        print("   │  └── news_clean_multisource.parquet")
        print("   └─ data_processed/")
        print("      ├── news_clean.parquet")
        print("      ├── bow_daily.parquet")
        print("      ├── tfidf_daily_matrix.npz")
        print("      ├── tfidf_daily_index.csv")
        print("      └── tfidf_daily_vocab.json")
        
        print("\n🎯 Próximos passos:")
        print("   → Execute 16_models_tfidf_baselines.ipynb para treinar modelos")
        print("   → Execute 17_sentiment_validation.ipynb para validação")
    else:
        print("❌ STATUS: PIPELINE INCOMPLETO - VERIFIQUE OS ERROS ACIMA")
    
    print("=" * 80)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
