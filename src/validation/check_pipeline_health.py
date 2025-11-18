"""
Verificação de saúde do pipeline completo
Verifica se todos os notebooks foram executados e se os outputs existem
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.io import paths


# Mapeamento de notebooks e seus outputs esperados
PIPELINE_OUTPUTS = {
    "00_data_download": [],  # Baixa dados externos
    "01_preprocessing": [],  # Processa dados baixados
    "02_baseline_logit": ["results_02_baseline_logit.json"],
    "03_tfidf_models": ["results_03_tfidf_models.json"],
    "04_embeddings_models": ["results_04_embeddings_models.json"],
    "05_data_collection_real": [],  # Coleta dados reais
    "06_preprocessing_real": ["noticias_real_clean.csv"],
    "07_tfidf_real": ["results_07_tfidf_real.json"],
    "08_embeddings_real": ["results_08_embeddings_real.json"],
    "09_lstm_real": ["results_09_lstm_real.json"],
    "10_dashboard_results": [],  # Dashboard visual
    "11_event_study_latency": [],  # Análise
    "12_data_collection_multisource": ["news_multisource_raw_*.parquet"],
    "13_etl_dedup": ["news_multisource.parquet", "etl_report_*.json"],
    "14_preprocess_ptbr": ["news_clean.parquet", "bow_daily.parquet"],
    "15_features_tfidf_daily": [
        "tfidf_daily_matrix.npz",
        "tfidf_daily_vocab.json",
        "labels_y_daily.csv",
        "dataset_daily_complete.parquet"
    ],
    "16_models_tfidf_baselines": ["results_16_models_tfidf.json"],
    "17_sentiment_validation": [],  # Análise
    "18_backtest_simulation": [],  # Simulação
    "19_future_extension": [],  # Extensão
    "20_final_dashboard_analysis": []  # Dashboard final
}


def check_notebook_outputs(notebook_name: str, outputs: List[str]) -> Dict[str, any]:
    """
    Verifica se os outputs de um notebook existem
    
    Args:
        notebook_name: Nome do notebook
        outputs: Lista de outputs esperados
    
    Returns:
        Dict com status dos outputs
    """
    DATA_PATHS = paths.get_data_paths()
    PROC_PATH = DATA_PATHS["data_processed"]
    BASE_PATH = DATA_PATHS["base"]
    
    result = {
        'notebook': notebook_name,
        'outputs': {},
        'all_exist': True
    }
    
    if not outputs:
        # Notebook sem outputs esperados
        result['all_exist'] = None  # N/A
        return result
    
    for output in outputs:
        # Suportar wildcards (ex: etl_report_*.json)
        if '*' in output:
            # Buscar arquivos que correspondem ao padrão
            pattern = output.replace('*', '')
            found_files = list(Path(PROC_PATH).glob(output))
            
            if found_files:
                result['outputs'][output] = {
                    'exists': True,
                    'files': [str(f.name) for f in found_files[:3]]  # Primeiros 3
                }
            else:
                result['outputs'][output] = {'exists': False}
                result['all_exist'] = False
        else:
            # Arquivo específico
            filepath = os.path.join(PROC_PATH, output)
            exists = os.path.exists(filepath)
            
            result['outputs'][output] = {'exists': exists}
            
            if exists:
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                result['outputs'][output]['size_mb'] = round(size_mb, 2)
            else:
                result['all_exist'] = False
    
    return result


def check_pipeline_health() -> Dict[str, any]:
    """
    Verifica saúde de todo o pipeline
    
    Returns:
        Dict com status de todos os notebooks
    """
    print("="*80)
    print("🏥 VERIFICAÇÃO DE SAÚDE DO PIPELINE")
    print("="*80)
    print()
    
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'notebooks': {}
    }
    
    total_notebooks = len(PIPELINE_OUTPUTS)
    notebooks_ok = 0
    notebooks_partial = 0
    notebooks_failed = 0
    
    for notebook_name, outputs in PIPELINE_OUTPUTS.items():
        print(f"📓 {notebook_name}:")
        
        result = check_notebook_outputs(notebook_name, outputs)
        health_report['notebooks'][notebook_name] = result
        
        if result['all_exist'] is None:
            print(f"   ℹ️ Sem outputs esperados (análise/dashboard)")
            notebooks_ok += 1
        elif result['all_exist']:
            print(f"   ✅ Todos os outputs existem")
            notebooks_ok += 1
        else:
            # Verificar quantos existem
            total_outputs = len(outputs)
            existing_outputs = sum(1 for o in result['outputs'].values() if o.get('exists', False))
            
            if existing_outputs == 0:
                print(f"   ❌ Nenhum output encontrado ({total_outputs} esperados)")
                notebooks_failed += 1
            else:
                print(f"   ⚠️ Outputs parciais ({existing_outputs}/{total_outputs})")
                notebooks_partial += 1
            
            # Listar faltando
            for output_name, output_info in result['outputs'].items():
                if not output_info.get('exists', False):
                    print(f"      ❌ Faltando: {output_name}")
        
        print()
    
    # Resumo
    print("="*80)
    print("📊 RESUMO:")
    print(f"   Total de notebooks: {total_notebooks}")
    print(f"   ✅ OK: {notebooks_ok}")
    print(f"   ⚠️ Parcial: {notebooks_partial}")
    print(f"   ❌ Falhou: {notebooks_failed}")
    
    health_report['summary'] = {
        'total': total_notebooks,
        'ok': notebooks_ok,
        'partial': notebooks_partial,
        'failed': notebooks_failed
    }
    
    # Status geral
    if notebooks_failed == 0 and notebooks_partial == 0:
        print("\n✅ PIPELINE SAUDÁVEL")
        health_report['status'] = 'HEALTHY'
    elif notebooks_failed == 0:
        print("\n⚠️ PIPELINE COM AVISOS (alguns outputs faltando)")
        health_report['status'] = 'WARNING'
    else:
        print("\n❌ PIPELINE COM PROBLEMAS (notebooks sem outputs)")
        health_report['status'] = 'UNHEALTHY'
    
    print("="*80)
    
    # Salvar relatório
    DATA_PATHS = paths.get_data_paths()
    output_file = os.path.join(
        DATA_PATHS["base"], 
        "reports", 
        f"pipeline_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(health_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Relatório salvo em: {output_file}")
    
    return health_report


if __name__ == "__main__":
    import sys
    
    result = check_pipeline_health()
    
    # Exit code baseado no status
    if result['status'] == 'UNHEALTHY':
        sys.exit(1)
    elif result['status'] == 'WARNING':
        sys.exit(2)
    else:
        sys.exit(0)
