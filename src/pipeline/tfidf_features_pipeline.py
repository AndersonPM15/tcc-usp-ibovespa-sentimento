"""
Pipeline de Features TF-IDF Diário
Gera matriz TF-IDF, labels target e dataset completo para modelagem
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import yfinance as yf

from src.io import paths
from src.config import loader as cfg


def download_ibovespa_data(start_date="2018-01-01", end_date=None):
    """
    Baixa dados históricos do Ibovespa via yfinance
    
    Args:
        start_date: Data inicial (formato YYYY-MM-DD)
        end_date: Data final (formato YYYY-MM-DD), None para hoje
    
    Returns:
        DataFrame com colunas: date, open, high, low, close, volume
    """
    print(f"\n📊 Baixando dados do Ibovespa (^BVSP)...")
    print(f"   Período: {start_date} → {end_date or 'hoje'}")
    
    try:
        ibov = yf.download("^BVSP", start=start_date, end=end_date, progress=False)
        
        if ibov.empty:
            raise ValueError("Nenhum dado retornado pelo yfinance")
        
        # Resetar índice e renomear colunas
        ibov = ibov.reset_index()
        ibov.columns = [col.lower() for col in ibov.columns]
        
        # Garantir coluna 'date'
        if 'date' not in ibov.columns and 'index' in ibov.columns:
            ibov = ibov.rename(columns={'index': 'date'})
        
        ibov['date'] = pd.to_datetime(ibov['date'])
        
        print(f"✅ Download completo: {len(ibov)} dias")
        print(f"   Período obtido: {ibov['date'].min().date()} → {ibov['date'].max().date()}")
        
        return ibov
    
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        raise


def prepare_daily_documents(df_news, group_by_source=False):
    """
    Agrega textos limpos por dia (ou dia x fonte)
    
    Args:
        df_news: DataFrame com colunas 'date', 'clean_text', 'source'
        group_by_source: Se True, agrupa por (date, source), senão só por date
    
    Returns:
        DataFrame com colunas: date, doc [, source]
    """
    print(f"\n📝 Preparando documentos diários...")
    
    # Garantir coluna date
    if 'published_at' in df_news.columns and 'date' not in df_news.columns:
        df_news['date'] = pd.to_datetime(df_news['published_at']).dt.date
    elif 'date' in df_news.columns:
        df_news['date'] = pd.to_datetime(df_news['date']).dt.date
    else:
        raise ValueError("DataFrame precisa ter coluna 'date' ou 'published_at'")
    
    # Filtrar textos válidos
    df_valid = df_news[df_news['clean_text'].notna()].copy()
    df_valid['clean_text'] = df_valid['clean_text'].astype(str)
    
    print(f"   Registros válidos: {len(df_valid)}")
    
    # Agregação
    if group_by_source:
        group_cols = ['date', 'source']
    else:
        group_cols = ['date']
    
    docs = (
        df_valid.groupby(group_cols, as_index=False)
        .agg({'clean_text': lambda x: ' '.join(x)})
        .rename(columns={'clean_text': 'doc'})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    
    print(f"✅ Documentos agregados: {len(docs)} registros")
    print(f"   Período: {docs['date'].min()} → {docs['date'].max()}")
    
    return docs


def create_tfidf_features(docs, min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=5000):
    """
    Cria matriz TF-IDF a partir dos documentos
    
    Args:
        docs: DataFrame com coluna 'doc'
        min_df: Min document frequency
        max_df: Max document frequency (proporção)
        ngram_range: Range de n-gramas (ex: (1,2) = unigram + bigram)
        max_features: Máximo de features (None = sem limite)
    
    Returns:
        X: Matriz TF-IDF (sparse)
        vectorizer: Objeto TfidfVectorizer fitted
        vocab: Lista ordenada de termos
    """
    print(f"\n🔢 Criando matriz TF-IDF...")
    print(f"   Parâmetros: min_df={min_df}, max_df={max_df}, ngram={ngram_range}, max_features={max_features}")
    
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        token_pattern=r"(?u)\b\w[\w\-áàâãéêíóôõúç]+\b"
    )
    
    X = vectorizer.fit_transform(docs['doc'].fillna(""))
    vocab = np.array(vectorizer.get_feature_names_out())
    
    print(f"✅ Matriz criada: {X.shape} (docs x features)")
    print(f"   Vocabulário: {len(vocab)} termos únicos")
    print(f"   Densidade: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%")
    
    return X, vectorizer, vocab


def create_rolling_features(docs, ibov_data, windows=[3, 5, 7]):
    """
    Cria features de janelas móveis para sentiment e volatilidade
    
    Args:
        docs: DataFrame com colunas 'date', 'sentiment' (opcional)
        ibov_data: DataFrame com colunas 'date', 'close'
        windows: Lista de tamanhos de janela (em dias)
    
    Returns:
        DataFrame com features de janelas móveis
    """
    print(f"\n📈 Criando features de janelas móveis...")
    print(f"   Janelas: {windows} dias")
    
    # Preparar dados
    df = docs.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    ibov = ibov_data.copy()
    ibov['date'] = pd.to_datetime(ibov['date'])
    ibov = ibov.sort_values('date')
    
    # Calcular retornos e volatilidade
    ibov['returns'] = ibov['close'].pct_change()
    
    for window in windows:
        ibov[f'volatility_{window}d'] = ibov['returns'].rolling(window).std()
        ibov[f'returns_mean_{window}d'] = ibov['returns'].rolling(window).mean()
    
    # Merge com documentos
    df_merged = df.merge(
        ibov[['date', 'close', 'returns'] + [f'volatility_{w}d' for w in windows] + [f'returns_mean_{w}d' for w in windows]],
        on='date',
        how='left'
    )
    
    # Se tiver sentiment, criar rolling sentiment
    if 'sentiment' in df.columns:
        for window in windows:
            df_merged[f'sentiment_{window}d'] = df_merged['sentiment'].rolling(window, min_periods=1).mean()
    
    print(f"✅ Features criadas: {df_merged.shape[1]} colunas")
    
    return df_merged


def create_target_labels(docs, ibov_data, horizons=[1, 3, 5]):
    """
    Cria labels target (retornos futuros) para diferentes horizontes
    
    Args:
        docs: DataFrame com coluna 'date'
        ibov_data: DataFrame com colunas 'date', 'close'
        horizons: Lista de horizontes (em dias) para labels D+1, D+3, D+5
    
    Returns:
        DataFrame com colunas date + target_1d, target_3d, target_5d (binário: 1=subiu, 0=caiu)
    """
    print(f"\n🎯 Criando labels target...")
    print(f"   Horizontes: {horizons} dias")
    
    # Preparar preços
    ibov = ibov_data.copy()
    ibov['date'] = pd.to_datetime(ibov['date'])
    ibov = ibov.sort_values('date').reset_index(drop=True)
    
    # Criar labels para cada horizonte
    for h in horizons:
        ibov[f'close_+{h}d'] = ibov['close'].shift(-h)
        ibov[f'return_+{h}d'] = (ibov[f'close_+{h}d'] / ibov['close']) - 1
        ibov[f'target_{h}d'] = (ibov[f'return_+{h}d'] > 0).astype(int)
    
    # Merge com documentos
    label_cols = ['date'] + [f'target_{h}d' for h in horizons] + [f'return_+{h}d' for h in horizons]
    df_labels = docs[['date']].merge(
        ibov[label_cols],
        on='date',
        how='left'
    )
    
    # Estatísticas
    for h in horizons:
        count = df_labels[f'target_{h}d'].notna().sum()
        if count > 0:
            positive_rate = df_labels[f'target_{h}d'].mean() * 100
            print(f"   D+{h}: {count} labels, {positive_rate:.1f}% positivos")
    
    return df_labels


def run_tfidf_pipeline(
    input_news_file=None,
    input_ibov_file=None,
    output_dir=None,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    max_features=5000,
    target_horizons=[1, 3, 5],
    rolling_windows=[3, 5, 7]
):
    """
    Pipeline completo de features TF-IDF
    
    Args:
        input_news_file: Path para news_clean.parquet (None = usar default)
        input_ibov_file: Path para ibovespa.csv (None = baixar via yfinance)
        output_dir: Diretório de saída (None = usar data_processed/)
        min_df, max_df, ngram_range, max_features: Parâmetros TF-IDF
        target_horizons: Horizontes de previsão em dias
        rolling_windows: Janelas móveis em dias
    
    Returns:
        Dict com paths dos arquivos gerados
    """
    print("="*80)
    print("🚀 PIPELINE DE FEATURES TF-IDF DIÁRIO")
    print("="*80)
    
    # Configurar paths
    DATA_PATHS = paths.get_data_paths()
    
    if input_news_file is None:
        input_news_file = os.path.join(DATA_PATHS["data_processed"], "news_clean.parquet")
    
    if output_dir is None:
        output_dir = DATA_PATHS["data_processed"]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Timestamp para outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== 1. CARREGAR DADOS DE NOTÍCIAS ==========
    print(f"\n{'='*80}")
    print("📰 ETAPA 1: Carregar dados de notícias")
    print(f"{'='*80}")
    
    if not os.path.exists(input_news_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_news_file}\nExecute o notebook 14 primeiro.")
    
    df_news = pd.read_parquet(input_news_file)
    print(f"✅ Notícias carregadas: {len(df_news)} registros")
    print(f"   Colunas: {df_news.columns.tolist()}")
    
    # ========== 2. CARREGAR/BAIXAR DADOS DO IBOVESPA ==========
    print(f"\n{'='*80}")
    print("📊 ETAPA 2: Carregar dados do Ibovespa")
    print(f"{'='*80}")
    
    if input_ibov_file and os.path.exists(input_ibov_file):
        print(f"📂 Carregando arquivo existente: {input_ibov_file}")
        ibov_data = pd.read_csv(input_ibov_file)
        ibov_data['date'] = pd.to_datetime(ibov_data['date'] if 'date' in ibov_data.columns else ibov_data.iloc[:, 0])
    else:
        # Determinar período baseado nas notícias
        if 'published_at' in df_news.columns:
            min_date = pd.to_datetime(df_news['published_at']).min()
            max_date = pd.to_datetime(df_news['published_at']).max()
        elif 'date' in df_news.columns:
            min_date = pd.to_datetime(df_news['date']).min()
            max_date = pd.to_datetime(df_news['date']).max()
        else:
            min_date = "2018-01-01"
            max_date = None
        
        start_date = (min_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d") if isinstance(min_date, pd.Timestamp) else min_date
        end_date = (max_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d") if isinstance(max_date, pd.Timestamp) and max_date else None
        
        ibov_data = download_ibovespa_data(start_date=start_date, end_date=end_date)
        
        # Salvar para uso futuro
        ibov_save_path = os.path.join(output_dir, f"ibovespa_{timestamp}.csv")
        ibov_data.to_csv(ibov_save_path, index=False)
        print(f"💾 Dados salvos em: {ibov_save_path}")
    
    # ========== 3. PREPARAR DOCUMENTOS DIÁRIOS ==========
    print(f"\n{'='*80}")
    print("📝 ETAPA 3: Preparar documentos diários")
    print(f"{'='*80}")
    
    docs = prepare_daily_documents(df_news, group_by_source=False)
    
    # ========== 4. CRIAR MATRIZ TF-IDF ==========
    print(f"\n{'='*80}")
    print("🔢 ETAPA 4: Criar matriz TF-IDF")
    print(f"{'='*80}")
    
    X_tfidf, vectorizer, vocab = create_tfidf_features(
        docs,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features
    )
    
    # Salvar artefatos TF-IDF
    tfidf_matrix_file = os.path.join(output_dir, "tfidf_daily_matrix.npz")
    tfidf_vocab_file = os.path.join(output_dir, "tfidf_daily_vocab.json")
    tfidf_index_file = os.path.join(output_dir, "tfidf_daily_index.csv")
    
    save_npz(tfidf_matrix_file, X_tfidf)
    
    with open(tfidf_vocab_file, "w", encoding="utf-8") as f:
        json.dump({"terms": vocab.tolist(), "n_features": len(vocab)}, f, ensure_ascii=False, indent=2)
    
    docs[['date']].to_csv(tfidf_index_file, index=False)
    
    print(f"💾 Artefatos salvos:")
    print(f"   - {tfidf_matrix_file}")
    print(f"   - {tfidf_vocab_file}")
    print(f"   - {tfidf_index_file}")
    
    # ========== 5. CRIAR LABELS TARGET ==========
    print(f"\n{'='*80}")
    print("🎯 ETAPA 5: Criar labels target")
    print(f"{'='*80}")
    
    labels = create_target_labels(docs, ibov_data, horizons=target_horizons)
    
    labels_file = os.path.join(output_dir, "labels_y_daily.csv")
    labels.to_csv(labels_file, index=False)
    print(f"💾 Labels salvos em: {labels_file}")
    
    # ========== 6. CRIAR FEATURES DE JANELAS MÓVEIS ==========
    print(f"\n{'='*80}")
    print("📈 ETAPA 6: Criar features de janelas móveis")
    print(f"{'='*80}")
    
    # Se tiver sentiment no df_news, agregar por dia
    if 'sentiment' in df_news.columns:
        sentiment_daily = df_news.groupby(
            pd.to_datetime(df_news['published_at'] if 'published_at' in df_news.columns else df_news['date']).dt.date
        )['sentiment'].mean().reset_index()
        sentiment_daily.columns = ['date', 'sentiment']
        docs = docs.merge(sentiment_daily, on='date', how='left')
    
    rolling_features = create_rolling_features(docs, ibov_data, windows=rolling_windows)
    
    # ========== 7. CRIAR DATASET COMPLETO ==========
    print(f"\n{'='*80}")
    print("📦 ETAPA 7: Criar dataset completo")
    print(f"{'='*80}")
    
    # Merge de todas as features
    dataset = docs[['date']].copy()
    
    # Adicionar labels
    for col in labels.columns:
        if col != 'date':
            dataset[col] = labels[col]
    
    # Adicionar features de rolling
    rolling_cols = [col for col in rolling_features.columns if col not in ['date', 'doc']]
    for col in rolling_cols:
        dataset[col] = rolling_features[col]
    
    # Adicionar row_id para alinhar com matriz TF-IDF
    dataset['row_id'] = np.arange(len(dataset))
    
    dataset_file = os.path.join(output_dir, "dataset_daily_complete.parquet")
    dataset.to_parquet(dataset_file, index=False)
    print(f"💾 Dataset completo salvo em: {dataset_file}")
    print(f"   Shape: {dataset.shape}")
    print(f"   Colunas: {dataset.columns.tolist()}")
    
    # ========== 8. RELATÓRIO FINAL ==========
    print(f"\n{'='*80}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*80}")
    
    # Calcular estatísticas
    coverage = {}
    for h in target_horizons:
        total = labels[f'target_{h}d'].notna().sum()
        if total > 0:
            positive = labels[f'target_{h}d'].sum()
            coverage[f"target_{h}d"] = {
                "total": int(total),
                "positive": int(positive),
                "positive_rate": float(positive / total)
            }
    
    report = {
        "timestamp": timestamp,
        "input_files": {
            "news": input_news_file,
            "ibovespa": input_ibov_file or "yfinance_download"
        },
        "data_summary": {
            "n_news": int(len(df_news)),
            "n_days": int(len(docs)),
            "date_range": {
                "start": str(docs['date'].min()),
                "end": str(docs['date'].max())
            }
        },
        "tfidf_params": {
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "max_features": max_features,
            "vocab_size": int(len(vocab)),
            "matrix_shape": X_tfidf.shape,
            "density": float(X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))
        },
        "target_labels": coverage,
        "rolling_windows": rolling_windows,
        "output_files": {
            "tfidf_matrix": tfidf_matrix_file,
            "tfidf_vocab": tfidf_vocab_file,
            "tfidf_index": tfidf_index_file,
            "labels": labels_file,
            "dataset": dataset_file
        }
    }
    
    report_file = os.path.join(output_dir, f"tfidf_report_{timestamp}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Pipeline concluído com sucesso!")
    print(f"📄 Relatório salvo em: {report_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline de Features TF-IDF Diário")
    parser.add_argument("--input-news", type=str, help="Path para news_clean.parquet")
    parser.add_argument("--input-ibov", type=str, help="Path para ibovespa.csv")
    parser.add_argument("--output-dir", type=str, help="Diretório de saída")
    parser.add_argument("--min-df", type=int, default=2, help="Min document frequency")
    parser.add_argument("--max-df", type=float, default=0.95, help="Max document frequency")
    parser.add_argument("--max-features", type=int, default=5000, help="Max features (None=sem limite)")
    
    args = parser.parse_args()
    
    max_features = args.max_features if args.max_features > 0 else None
    
    run_tfidf_pipeline(
        input_news_file=args.input_news,
        input_ibov_file=args.input_ibov,
        output_dir=args.output_dir,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=max_features
    )
