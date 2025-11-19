"""
Executa notebooks 13-15 em sequência para regenerar pipeline TCC
"""
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    ("13_etl_dedup.ipynb", "ETL & Deduplicação"),
    ("14_preprocess_ptbr.ipynb", "Pré-processamento PT-BR"),
    ("15_features_tfidf_daily.ipynb", "Features TF-IDF Diárias"),
]

def run_notebook(notebook_path: Path, description: str):
    """Executa um notebook usando jupyter nbconvert"""
    print("\n" + "="*80)
    print(f"EXECUTANDO: {description}")
    print(f"Arquivo: {notebook_path.name}")
    print("="*80)
    
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=3600",  # 1 hora timeout
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} - CONCLUÍDO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ERRO")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    base_path = Path(__file__).parent / "notebooks"
    
    print("="*80)
    print("PIPELINE TCC - REGENERAÇÃO COMPLETA")
    print("Notebooks 13-15 com base GDELT histórica (2.771 dias)")
    print("="*80)
    
    results = []
    for notebook_file, description in NOTEBOOKS:
        notebook_path = base_path / notebook_file
        if not notebook_path.exists():
            print(f"⚠️ Arquivo não encontrado: {notebook_path}")
            results.append(False)
            continue
        
        success = run_notebook(notebook_path, description)
        results.append(success)
        
        if not success:
            print("\n❌ Pipeline interrompida devido a erro")
            sys.exit(1)
    
    print("\n" + "="*80)
    if all(results):
        print("✅ PIPELINE COMPLETA - TODOS OS NOTEBOOKS EXECUTADOS COM SUCESSO")
        print("="*80)
        print("\nArquivos gerados:")
        print("  - data_interim/news_clean_multisource.parquet")
        print("  - data_processed/news_clean.parquet")
        print("  - data_processed/bow_daily.parquet")
        print("  - data_processed/tfidf_daily_matrix.npz")
        print("  - data_processed/tfidf_daily_index.csv")
        print("  - data_processed/tfidf_daily_vocab.json")
    else:
        print("❌ PIPELINE INCOMPLETA - VERIFIQUE OS ERROS ACIMA")
    print("="*80)

if __name__ == "__main__":
    main()
