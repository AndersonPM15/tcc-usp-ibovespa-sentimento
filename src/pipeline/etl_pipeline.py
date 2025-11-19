"""
Script completo de ETL e deduplicação de notícias multisource.
Pode ser executado standalone ou importado por notebooks.

Usage:
    python -m src.pipeline.etl_pipeline
    
Ou import src.pipeline.etl_pipeline no notebook
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Adicionar src ao path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import paths
from src.config import loader as cfg
from src.utils.etl_dedup import (
    dedup_by_url,
    dedup_by_title_date,
    dedup_by_embedding_similarity,
    validate_and_clean_fields,
    normalize_timezone,
    create_etl_report
)


def run_etl_pipeline(input_file: str = None, use_embedding_dedup: bool = False) -> pd.DataFrame:
    """
    Executa pipeline completo de ETL e deduplicação.
    
    Args:
        input_file: Caminho para arquivo de entrada. Se None, busca último no RAW_PATH
        use_embedding_dedup: Se True, aplica dedup por similaridade (lento para datasets grandes)
    
    Returns:
        DataFrame limpo e dedupado
    """
    print("="*70)
    print("🔧 PIPELINE ETL - DEDUPLICAÇÃO DE NOTÍCIAS MULTISOURCE")
    print("="*70)
    
    # 1. Setup de caminhos
    DATA_PATHS = paths.get_data_paths()
    RAW_PATH = str(DATA_PATHS["data_raw"])
    PROC_PATH = str(DATA_PATHS["data_processed"])
    
    REPORTS_PATH = PROJECT_ROOT / "reports"
    REPORTS_PATH.mkdir(exist_ok=True)
    
    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📂 RAW_PATH:  {RAW_PATH}")
    print(f"📂 PROC_PATH: {PROC_PATH}")
    
    # 2. Carregar dados brutos
    print("\n" + "="*70)
    print("📥 CARREGANDO DADOS BRUTOS")
    print("="*70)
    
    if input_file is None:
        # Buscar último arquivo multisource_raw
        raw_files = list(Path(RAW_PATH).glob("news_multisource_raw_*.parquet"))
        if not raw_files:
            raise FileNotFoundError(
                f"❌ Nenhum arquivo news_multisource_raw_*.parquet encontrado em {RAW_PATH}\n"
                f"Execute primeiro o Notebook 12 (coleta multisource)"
            )
        input_file = str(sorted(raw_files)[-1])
    
    print(f"Carregando: {input_file}")
    df_raw = pd.read_parquet(input_file)
    
    print(f"✅ Dados carregados: {len(df_raw):,} registros")
    print(f"   Colunas: {list(df_raw.columns)}")
    print(f"   Fontes: {df_raw['source_type'].value_counts().to_dict() if 'source_type' in df_raw.columns else 'N/A'}")
    
    df_initial = df_raw.copy()
    
    # 3. Deduplicação
    print("\n" + "="*70)
    print("🔄 DEDUPLICAÇÃO")
    print("="*70)
    
    dedup_stats = {}
    
    # 3.1 Por URL
    print("\n1️⃣ Deduplicação por URL canonizada...")
    df, removed_url = dedup_by_url(df_raw)
    dedup_stats['by_url'] = removed_url
    
    # 3.2 Por título + data
    print("\n2️⃣ Deduplicação por título + data...")
    df, removed_title = dedup_by_title_date(df)
    dedup_stats['by_title_date'] = removed_title
    
    # 3.3 Por similaridade de embeddings (opcional - muito lento)
    if use_embedding_dedup:
        print("\n3️⃣ Deduplicação por similaridade de embeddings...")
        print("⚠️ AVISO: Esta etapa pode demorar vários minutos para datasets grandes")
        df, removed_embedding = dedup_by_embedding_similarity(df, threshold=0.92)
        dedup_stats['by_embedding'] = removed_embedding
    else:
        print("\n3️⃣ Deduplicação por embedding: DESATIVADA (use use_embedding_dedup=True para ativar)")
        dedup_stats['by_embedding'] = 0
    
    dedup_stats['total'] = len(df_initial) - len(df)
    
    print(f"\n✅ Deduplicação concluída: {dedup_stats['total']:,} duplicatas removidas")
    print(f"   Registros restantes: {len(df):,}")
    
    # 4. Validação e limpeza de campos
    print("\n" + "="*70)
    print("📋 VALIDAÇÃO E LIMPEZA DE CAMPOS")
    print("="*70)
    
    df, validation_stats = validate_and_clean_fields(df)
    
    print(f"\n✅ Validação concluída: {validation_stats['total_removed']:,} registros inválidos removidos")
    print(f"   Registros válidos: {len(df):,}")
    
    # 5. Normalização de timezone
    print("\n" + "="*70)
    print("🌐 NORMALIZAÇÃO DE TIMEZONE")
    print("="*70)
    
    df = normalize_timezone(df, target_tz='America/Sao_Paulo')
    
    # 6. Criar coluna 'date' (apenas dia)
    if 'published_at' in df.columns:
        df['date'] = df['published_at'].dt.date
        print("✅ Coluna 'date' criada (apenas dia)")
    
    # 7. Limpeza de colunas temporárias
    temp_cols = ['url_canonical', 'title_normalized', 'date_only', 'text_length', 'year']
    df = df.drop(columns=[col for col in temp_cols if col in df.columns], errors='ignore')
    
    # 8. Salvar dataset limpo
    print("\n" + "="*70)
    print("💾 SALVANDO DATASET LIMPO")
    print("="*70)
    
    output_parquet = os.path.join(PROC_PATH, "news_multisource.parquet")
    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"✅ Salvo: {output_parquet}")
    print(f"   Shape: {df.shape}")
    print(f"   Size: {os.path.getsize(output_parquet) / (1024*1024):.2f} MB")
    
    # 9. Criar relatório
    print("\n" + "="*70)
    print("📊 CRIANDO RELATÓRIO ETL")
    print("="*70)
    
    report_path = REPORTS_PATH / f"etl_report_{STAMP}.json"
    report = create_etl_report(df_initial, df, dedup_stats, validation_stats, str(report_path))
    
    # 10. Resumo final
    print("\n" + "="*70)
    print("✅ PIPELINE ETL CONCLUÍDO")
    print("="*70)
    print(f"\n📊 Resumo:")
    print(f"   Inicial:  {len(df_initial):,} registros")
    print(f"   Final:    {len(df):,} registros")
    print(f"   Removido: {len(df_initial) - len(df):,} registros ({((len(df_initial) - len(df)) / len(df_initial)) * 100:.1f}%)")
    print(f"\n📅 Cobertura temporal:")
    if 'published_at' in df.columns:
        print(f"   Início: {df['published_at'].min()}")
        print(f"   Fim:    {df['published_at'].max()}")
        print(f"   Dias:   {(df['published_at'].max() - df['published_at'].min()).days}")
    print(f"\n📈 Por fonte:")
    if 'source_type' in df.columns:
        for source, count in df['source_type'].value_counts().items():
            print(f"   {source:15s}: {count:6,} ({(count/len(df))*100:.1f}%)")
    
    print(f"\n📄 Relatório: {report_path}")
    print(f"📦 Output:    {output_parquet}")
    print(f"\n⏭️ Próximo passo: Executar Notebook 14 (Preprocessamento PT-BR)")
    
    return df


if __name__ == "__main__":
    # Execução standalone
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline ETL de notícias multisource")
    parser.add_argument("--input", type=str, default=None, help="Arquivo de entrada (parquet)")
    parser.add_argument("--embedding-dedup", action="store_true", help="Ativar dedup por embedding (lento)")
    
    args = parser.parse_args()
    
    df_clean = run_etl_pipeline(input_file=args.input, use_embedding_dedup=args.embedding_dedup)
    print(f"\n✅ Pipeline concluído. Dataset limpo: {len(df_clean):,} registros")
