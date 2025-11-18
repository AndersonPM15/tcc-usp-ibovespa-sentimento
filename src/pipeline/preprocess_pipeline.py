"""
Pipeline completo de preprocessamento PT-BR.
Pode ser executado standalone ou importado por notebooks.

Usage:
    python -m src.pipeline.preprocess_pipeline
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Adicionar src ao path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import paths
from src.utils.preprocess_ptbr import (
    preprocess_pipeline,
    detect_language,
    generate_embeddings,
    analyze_sentiment,
    calculate_credibility_score,
    calculate_novelty_score
)


def run_preprocess_pipeline(input_file: str = None, 
                            generate_emb: bool = True,
                            analyze_sent: bool = True) -> pd.DataFrame:
    """
    Executa pipeline completo de preprocessamento PT-BR.
    
    Args:
        input_file: Arquivo de entrada. Se None, busca news_multisource.parquet
        generate_emb: Se True, gera embeddings (lento)
        analyze_sent: Se True, analisa sentiment
    
    Returns:
        DataFrame preprocessado
    """
    print("="*70)
    print("🧹 PIPELINE PREPROCESSAMENTO PT-BR")
    print("="*70)
    
    # 1. Setup
    DATA_PATHS = paths.get_data_paths()
    PROC_PATH = str(DATA_PATHS["data_processed"])
    
    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📂 PROC_PATH: {PROC_PATH}")
    
    # 2. Carregar dados
    print("\n" + "="*70)
    print("📥 CARREGANDO DADOS")
    print("="*70)
    
    if input_file is None:
        input_file = os.path.join(PROC_PATH, "news_multisource.parquet")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"❌ Arquivo não encontrado: {input_file}\n"
            f"Execute primeiro o Notebook 13 (ETL + Deduplicação)"
        )
    
    print(f"Carregando: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"✅ Dados carregados: {len(df):,} registros")
    
    # 3. Preprocessamento de texto
    print("\n" + "="*70)
    print("🧹 PREPROCESSAMENTO DE TEXTO")
    print("="*70)
    
    df = preprocess_pipeline(df, remove_stopwords=True)
    
    # 4. Detecção de idioma
    print("\n" + "="*70)
    print("🌐 DETECÇÃO DE IDIOMA")
    print("="*70)
    
    df = detect_language(df, sample_size=1000)
    
    # 5. Geração de embeddings (opcional - lento)
    if generate_emb:
        print("\n" + "="*70)
        print("🧠 GERAÇÃO DE EMBEDDINGS")
        print("="*70)
        print("⚠️ Esta etapa pode demorar vários minutos...")
        
        df = generate_embeddings(df, batch_size=32)
    else:
        print("\n⏭️ Geração de embeddings: DESATIVADA")
        df['embedding_768'] = None
    
    # 6. Análise de sentimento
    if analyze_sent:
        print("\n" + "="*70)
        print("😊 ANÁLISE DE SENTIMENTO")
        print("="*70)
        
        df = analyze_sentiment(df)
    else:
        print("\n⏭️ Análise de sentimento: DESATIVADA")
        df['sentiment'] = 0.0
    
    # 7. Credibility score
    print("\n" + "="*70)
    print("🔍 CREDIBILITY SCORE")
    print("="*70)
    
    df = calculate_credibility_score(df)
    
    # 8. Novelty score
    print("\n" + "="*70)
    print("🆕 NOVELTY SCORE")
    print("="*70)
    
    df = calculate_novelty_score(df)
    
    # 9. Criar agregação diária (BoW)
    print("\n" + "="*70)
    print("📅 AGREGAÇÃO DIÁRIA (Bag-of-Words)")
    print("="*70)
    
    if 'date' in df.columns or 'published_at' in df.columns:
        # Criar coluna date se não existir
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['published_at']).dt.date
        
        # Agregar por dia
        df_daily = df.groupby('date').agg({
            'clean_text': lambda x: ' '.join(x),
            'sentiment': 'mean',
            'credibility_score': 'mean',
            'novelty_score': 'mean',
            'token_count': 'sum',
            'id': 'count'  # Número de notícias por dia
        }).reset_index()
        
        df_daily = df_daily.rename(columns={'id': 'news_count'})
        
        print(f"✅ Agregação diária criada: {len(df_daily):,} dias únicos")
        print(f"   Média de notícias/dia: {df_daily['news_count'].mean():.1f}")
        
        # Salvar BoW diário
        bow_output = os.path.join(PROC_PATH, "bow_daily.parquet")
        df_daily.to_parquet(bow_output, index=False)
        print(f"💾 Salvo: {bow_output}")
    
    # 10. Salvar dataset preprocessado
    print("\n" + "="*70)
    print("💾 SALVANDO DATASET PREPROCESSADO")
    print("="*70)
    
    output_file = os.path.join(PROC_PATH, "news_clean.parquet")
    df.to_parquet(output_file, index=False)
    
    print(f"✅ Salvo: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # 11. Resumo final
    print("\n" + "="*70)
    print("✅ PREPROCESSAMENTO CONCLUÍDO")
    print("="*70)
    print(f"\n📊 Estatísticas finais:")
    print(f"   Total de registros: {len(df):,}")
    print(f"   Tokens médios:      {df['token_count'].mean():.1f}")
    print(f"   Sentiment médio:    {df['sentiment'].mean():.3f}")
    print(f"   Credibility média:  {df['credibility_score'].mean():.3f}")
    print(f"   Novelty média:      {df['novelty_score'].mean():.3f}")
    
    print(f"\n📦 Outputs:")
    print(f"   news_clean.parquet")
    print(f"   bow_daily.parquet")
    print(f"\n⏭️ Próximo passo: Executar Notebook 15 (Features TF-IDF Diário)")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline preprocessamento PT-BR")
    parser.add_argument("--input", type=str, default=None, help="Arquivo de entrada")
    parser.add_argument("--no-embeddings", action="store_true", help="Não gerar embeddings")
    parser.add_argument("--no-sentiment", action="store_true", help="Não analisar sentiment")
    
    args = parser.parse_args()
    
    df_clean = run_preprocess_pipeline(
        input_file=args.input,
        generate_emb=not args.no_embeddings,
        analyze_sent=not args.no_sentiment
    )
    
    print(f"\n✅ Pipeline concluído: {len(df_clean):,} registros processados")
