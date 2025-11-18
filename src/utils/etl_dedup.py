"""
Funções de ETL e deduplicação para pipeline de notícias.
Utilizado por notebooks/13_etl_dedup.ipynb
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import hashlib
from urllib.parse import urlparse
import json
from datetime import datetime


def normalize_url(url: str) -> str:
    """
    Normaliza URL para detectar duplicatas.
    Remove parâmetros de query, fragmentos e www.
    """
    if pd.isna(url) or not url:
        return ""
    
    try:
        parsed = urlparse(url.lower())
        # Remover www
        netloc = parsed.netloc.replace('www.', '')
        # Reconstruir sem query e fragment
        normalized = f"{parsed.scheme}://{netloc}{parsed.path}"
        return normalized.rstrip('/')
    except:
        return url.lower()


def dedup_by_url(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicatas por URL canonizada.
    Retorna: (df limpo, número de duplicatas removidas)
    """
    initial_count = len(df)
    
    # Criar coluna de URL normalizada
    df['url_canonical'] = df['url'].apply(normalize_url)
    
    # Remover duplicatas mantendo primeiro
    df_clean = df.drop_duplicates(subset=['url_canonical'], keep='first')
    
    removed = initial_count - len(df_clean)
    print(f"  Duplicatas por URL: {removed:,} removidas")
    
    return df_clean, removed


def dedup_by_title_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicatas por título + data (mesmo dia).
    """
    initial_count = len(df)
    
    # Normalizar títulos
    df['title_normalized'] = df['title'].str.lower().str.strip()
    
    # Extrair data (apenas dia)
    df['date_only'] = pd.to_datetime(df['published_at'], errors='coerce').dt.date
    
    # Remover duplicatas por título + data
    df_clean = df.drop_duplicates(subset=['title_normalized', 'date_only'], keep='first')
    
    removed = initial_count - len(df_clean)
    print(f"  Duplicatas por título+data: {removed:,} removidas")
    
    return df_clean, removed


def dedup_by_embedding_similarity(df: pd.DataFrame, threshold: float = 0.92) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicatas por similaridade de embeddings.
    Usa cosine similarity com threshold configurável.
    
    NOTA: Para datasets grandes (>10k), esta função pode ser lenta.
    Considere usar apenas em datasets já parcialmente dedupados.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    initial_count = len(df)
    
    # Verificar se vale a pena (muito custoso para datasets grandes)
    if len(df) > 10000:
        print(f"  ⚠️ Dataset grande ({len(df):,}). Dedup por embedding pode demorar.")
        print(f"  Processando sample de 10.000 registros...")
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    else:
        df_sample = df.copy()
    
    # Gerar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df_sample['title'].fillna('') + ' ' + df_sample['description'].fillna('')
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    
    # Calcular similaridade
    similarities = cosine_similarity(embeddings)
    
    # Identificar duplicatas
    to_remove = set()
    for i in range(len(similarities)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(similarities)):
            if j in to_remove:
                continue
            if similarities[i, j] >= threshold:
                to_remove.add(j)
    
    # Remover duplicatas
    indices_to_keep = [i for i in range(len(df_sample)) if i not in to_remove]
    df_clean = df_sample.iloc[indices_to_keep].copy()
    
    # Se processamos sample, retornar sample limpo + resto
    if len(df) > len(df_sample):
        remaining = df[~df.index.isin(df_sample.index)]
        df_clean = pd.concat([df_clean, remaining], ignore_index=True)
    
    removed = initial_count - len(df_clean)
    print(f"  Duplicatas por similaridade: {removed:,} removidas (threshold={threshold})")
    
    return df_clean, removed


def validate_and_clean_fields(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Valida e limpa campos essenciais.
    Remove registros inválidos.
    """
    stats = {}
    initial_count = len(df)
    
    # 1. Remover linhas sem título
    df = df[df['title'].notna() & (df['title'].str.strip() != '')]
    stats['no_title'] = initial_count - len(df)
    
    # 2. Remover linhas com texto muito curto (< 20 caracteres úteis)
    df['text_length'] = df['raw_text'].fillna('').str.len()
    df = df[df['text_length'] >= 20]
    stats['text_too_short'] = initial_count - len(df) - stats['no_title']
    
    # 3. Normalizar datas
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    invalid_dates = df['published_at'].isna().sum()
    stats['invalid_dates'] = invalid_dates
    
    # 4. Remover URLs inválidas
    df = df[df['url'].notna() & (df['url'].str.strip() != '')]
    stats['no_url'] = initial_count - len(df) - sum(stats.values())
    
    stats['total_removed'] = initial_count - len(df)
    
    print("\n📋 Validação de campos:")
    print(f"  Sem título:        {stats['no_title']:,}")
    print(f"  Texto < 20 chars:  {stats['text_too_short']:,}")
    print(f"  Datas inválidas:   {stats['invalid_dates']:,}")
    print(f"  Sem URL:           {stats['no_url']:,}")
    print(f"  Total removido:    {stats['total_removed']:,}")
    
    return df, stats


def normalize_timezone(df: pd.DataFrame, target_tz: str = 'America/Sao_Paulo') -> pd.DataFrame:
    """
    Normaliza todas as datas para timezone único.
    """
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
        df['published_at'] = df['published_at'].dt.tz_convert(target_tz)
        print(f"✅ Datas normalizadas para timezone: {target_tz}")
    
    return df


def create_etl_report(df_initial: pd.DataFrame, df_final: pd.DataFrame, 
                     dedup_stats: dict, validation_stats: dict, 
                     output_path: str) -> None:
    """
    Cria relatório JSON com estatísticas do ETL.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "initial_count": len(df_initial),
        "final_count": len(df_final),
        "total_removed": len(df_initial) - len(df_final),
        "removal_rate": ((len(df_initial) - len(df_final)) / len(df_initial)) * 100,
        "deduplication": dedup_stats,
        "validation": validation_stats,
        "sources": {
            "by_type": df_final['source_type'].value_counts().to_dict() if 'source_type' in df_final.columns else {},
            "by_source": df_final['source'].value_counts().head(10).to_dict() if 'source' in df_final.columns else {}
        },
        "temporal_coverage": {
            "start": str(df_final['published_at'].min()) if 'published_at' in df_final.columns else None,
            "end": str(df_final['published_at'].max()) if 'published_at' in df_final.columns else None,
            "days": int((df_final['published_at'].max() - df_final['published_at'].min()).days) if 'published_at' in df_final.columns else None
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Relatório ETL salvo: {output_path}")
    return report
