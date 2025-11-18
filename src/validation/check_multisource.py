"""
Validação de dados multisource
Verifica se os dados do pipeline multisource (NB 12-14) estão completos e corretos
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.io import paths
from src.config import loader as cfg


def check_multisource_files() -> Dict[str, bool]:
    """
    Verifica se todos os arquivos esperados do pipeline multisource existem
    
    Returns:
        Dict com status de cada arquivo
    """
    DATA_PATHS = paths.get_data_paths()
    PROC_PATH = DATA_PATHS["data_processed"]
    
    expected_files = {
        "news_multisource.parquet": os.path.join(PROC_PATH, "news_multisource.parquet"),
        "news_clean.parquet": os.path.join(PROC_PATH, "news_clean.parquet"),
        "bow_daily.parquet": os.path.join(PROC_PATH, "bow_daily.parquet"),
        "tfidf_daily_matrix.npz": os.path.join(PROC_PATH, "tfidf_daily_matrix.npz"),
        "tfidf_daily_vocab.json": os.path.join(PROC_PATH, "tfidf_daily_vocab.json"),
        "labels_y_daily.csv": os.path.join(PROC_PATH, "labels_y_daily.csv"),
        "dataset_daily_complete.parquet": os.path.join(PROC_PATH, "dataset_daily_complete.parquet"),
    }
    
    results = {}
    print("📁 Verificando arquivos do pipeline multisource...\n")
    
    for name, filepath in expected_files.items():
        exists = os.path.exists(filepath)
        results[name] = exists
        
        status = "✅" if exists else "❌"
        print(f"{status} {name}")
        
        if exists:
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"   Tamanho: {size_mb:.2f} MB")
    
    return results


def check_data_sources(df_news: pd.DataFrame) -> Dict[str, any]:
    """
    Verifica se existem dados de todas as fontes esperadas
    
    Args:
        df_news: DataFrame com notícias
    
    Returns:
        Dict com estatísticas por fonte
    """
    print("\n📊 Verificando fontes de dados...\n")
    
    if 'source' not in df_news.columns:
        print("⚠️ Coluna 'source' não encontrada no dataset")
        return {}
    
    # Fontes esperadas
    expected_sources = ['gdelt', 'gnews', 'rss', 'newsapi']
    
    source_stats = {}
    source_counts = df_news['source'].value_counts()
    
    for source in expected_sources:
        count = source_counts.get(source, 0)
        percentage = (count / len(df_news) * 100) if len(df_news) > 0 else 0
        
        source_stats[source] = {
            'count': int(count),
            'percentage': float(percentage)
        }
        
        status = "✅" if count > 0 else "❌"
        print(f"{status} {source}: {count} notícias ({percentage:.1f}%)")
    
    # Fontes não esperadas
    unexpected = set(source_counts.index) - set(expected_sources)
    if unexpected:
        print(f"\n⚠️ Fontes inesperadas encontradas: {unexpected}")
    
    return source_stats


def check_date_coverage(df_news: pd.DataFrame, min_years: int = 3) -> Dict[str, any]:
    """
    Verifica cobertura temporal dos dados
    
    Args:
        df_news: DataFrame com notícias
        min_years: Mínimo de anos de cobertura esperado
    
    Returns:
        Dict com estatísticas de cobertura
    """
    print(f"\n📅 Verificando cobertura temporal (mínimo: {min_years} anos)...\n")
    
    # Determinar coluna de data
    date_col = None
    for col in ['published_at', 'date', 'data']:
        if col in df_news.columns:
            date_col = col
            break
    
    if date_col is None:
        print("❌ Nenhuma coluna de data encontrada")
        return {'error': 'No date column'}
    
    df_news[date_col] = pd.to_datetime(df_news[date_col], errors='coerce')
    df_valid = df_news[df_news[date_col].notna()].copy()
    
    if len(df_valid) == 0:
        print("❌ Nenhuma data válida encontrada")
        return {'error': 'No valid dates'}
    
    min_date = df_valid[date_col].min()
    max_date = df_valid[date_col].max()
    date_range = (max_date - min_date).days
    years_coverage = date_range / 365.25
    
    print(f"Período: {min_date.date()} → {max_date.date()}")
    print(f"Cobertura: {years_coverage:.1f} anos ({date_range} dias)")
    
    # Verificar se atende ao mínimo
    if years_coverage >= min_years:
        print(f"✅ Cobertura adequada (>= {min_years} anos)")
    else:
        print(f"⚠️ Cobertura insuficiente (< {min_years} anos)")
    
    # Verificar gaps
    df_sorted = df_valid.sort_values(date_col)
    df_sorted['date_only'] = df_sorted[date_col].dt.date
    daily_counts = df_sorted.groupby('date_only').size()
    
    # Dias sem notícias
    all_dates = pd.date_range(min_date, max_date, freq='D')
    missing_dates = set(all_dates.date) - set(daily_counts.index)
    missing_pct = len(missing_dates) / len(all_dates) * 100
    
    print(f"\nDias sem notícias: {len(missing_dates)} ({missing_pct:.1f}%)")
    
    if missing_pct > 50:
        print("⚠️ Muitos dias sem cobertura (>50%)")
    
    stats = {
        'min_date': str(min_date.date()),
        'max_date': str(max_date.date()),
        'date_range_days': int(date_range),
        'years_coverage': float(years_coverage),
        'missing_days': int(len(missing_dates)),
        'missing_percentage': float(missing_pct),
        'meets_minimum': years_coverage >= min_years
    }
    
    return stats


def check_volume_threshold(df_news: pd.DataFrame, min_volume: int = 5000) -> Dict[str, any]:
    """
    Verifica se o volume de dados atende ao mínimo esperado
    
    Args:
        df_news: DataFrame com notícias
        min_volume: Mínimo de notícias esperado
    
    Returns:
        Dict com estatísticas de volume
    """
    print(f"\n📈 Verificando volume de dados (mínimo: {min_volume:,})...\n")
    
    total = len(df_news)
    print(f"Total de notícias: {total:,}")
    
    if total >= min_volume:
        print(f"✅ Volume adequado (>= {min_volume:,})")
    else:
        print(f"❌ Volume insuficiente (< {min_volume:,})")
    
    # Estatísticas adicionais
    if 'clean_text' in df_news.columns:
        valid_text = df_news['clean_text'].notna().sum()
        print(f"Com texto limpo: {valid_text:,} ({valid_text/total*100:.1f}%)")
    
    if 'sentiment' in df_news.columns:
        valid_sentiment = df_news['sentiment'].notna().sum()
        print(f"Com sentiment: {valid_sentiment:,} ({valid_sentiment/total*100:.1f}%)")
    
    stats = {
        'total': int(total),
        'meets_minimum': total >= min_volume,
        'threshold': min_volume
    }
    
    return stats


def check_data_quality(df_news: pd.DataFrame) -> Dict[str, any]:
    """
    Verifica qualidade dos dados (duplicatas, campos vazios, etc)
    
    Args:
        df_news: DataFrame com notícias
    
    Returns:
        Dict com métricas de qualidade
    """
    print("\n🔍 Verificando qualidade dos dados...\n")
    
    quality = {}
    
    # Duplicatas
    if 'url' in df_news.columns:
        duplicates = df_news['url'].duplicated().sum()
        dup_pct = duplicates / len(df_news) * 100
        print(f"Duplicatas (URL): {duplicates} ({dup_pct:.1f}%)")
        quality['duplicates_url'] = int(duplicates)
        
        if dup_pct > 5:
            print("⚠️ Muitas duplicatas (>5%)")
    
    # Campos obrigatórios
    required_fields = ['title', 'clean_text', 'date']
    
    for field in required_fields:
        if field in df_news.columns or field in ['published_at', 'data']:
            col = field if field in df_news.columns else ('published_at' if 'published_at' in df_news.columns else 'data')
            missing = df_news[col].isna().sum()
            missing_pct = missing / len(df_news) * 100
            
            status = "✅" if missing_pct < 5 else "⚠️"
            print(f"{status} {field}: {missing} vazios ({missing_pct:.1f}%)")
            quality[f'missing_{field}'] = int(missing)
    
    # Textos muito curtos
    if 'clean_text' in df_news.columns:
        df_news['text_len'] = df_news['clean_text'].fillna('').str.len()
        too_short = (df_news['text_len'] < 50).sum()
        too_short_pct = too_short / len(df_news) * 100
        
        print(f"\nTextos muito curtos (<50 chars): {too_short} ({too_short_pct:.1f}%)")
        quality['too_short'] = int(too_short)
        
        if too_short_pct > 20:
            print("⚠️ Muitos textos curtos (>20%)")
    
    return quality


def run_full_validation(min_years: int = 3, min_volume: int = 5000) -> Dict[str, any]:
    """
    Executa todas as validações do pipeline multisource
    
    Args:
        min_years: Mínimo de anos de cobertura
        min_volume: Mínimo de notícias
    
    Returns:
        Dict com resultado completo da validação
    """
    print("="*80)
    print("🔍 VALIDAÇÃO DO PIPELINE MULTISOURCE")
    print("="*80)
    
    DATA_PATHS = paths.get_data_paths()
    PROC_PATH = DATA_PATHS["data_processed"]
    
    validation_result = {
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # 1. Verificar arquivos
    files_check = check_multisource_files()
    validation_result['checks']['files'] = files_check
    
    all_files_ok = all(files_check.values())
    
    if not all_files_ok:
        print("\n❌ VALIDAÇÃO FALHOU: Arquivos faltando")
        print("   Execute os notebooks 12-14 para gerar todos os arquivos")
        validation_result['status'] = 'FAILED'
        validation_result['reason'] = 'Missing files'
        return validation_result
    
    # 2. Carregar dados
    try:
        news_file = os.path.join(PROC_PATH, "news_clean.parquet")
        df_news = pd.read_parquet(news_file)
        print(f"\n✅ Dados carregados: {len(df_news):,} registros")
    except Exception as e:
        print(f"\n❌ Erro ao carregar dados: {e}")
        validation_result['status'] = 'FAILED'
        validation_result['reason'] = f'Load error: {e}'
        return validation_result
    
    # 3. Verificar fontes
    sources_check = check_data_sources(df_news)
    validation_result['checks']['sources'] = sources_check
    
    # 4. Verificar cobertura temporal
    coverage_check = check_date_coverage(df_news, min_years=min_years)
    validation_result['checks']['coverage'] = coverage_check
    
    # 5. Verificar volume
    volume_check = check_volume_threshold(df_news, min_volume=min_volume)
    validation_result['checks']['volume'] = volume_check
    
    # 6. Verificar qualidade
    quality_check = check_data_quality(df_news)
    validation_result['checks']['quality'] = quality_check
    
    # Resultado final
    print("\n" + "="*80)
    
    all_passed = (
        all_files_ok and
        coverage_check.get('meets_minimum', False) and
        volume_check.get('meets_minimum', False)
    )
    
    if all_passed:
        print("✅ VALIDAÇÃO PASSOU")
        validation_result['status'] = 'PASSED'
    else:
        print("⚠️ VALIDAÇÃO COM AVISOS")
        validation_result['status'] = 'WARNING'
        
        reasons = []
        if not coverage_check.get('meets_minimum', False):
            reasons.append('Insufficient time coverage')
        if not volume_check.get('meets_minimum', False):
            reasons.append('Insufficient volume')
        
        validation_result['reasons'] = reasons
    
    print("="*80)
    
    # Salvar resultado
    output_file = os.path.join(DATA_PATHS["base"], "reports", f"validation_multisource_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Relatório salvo em: {output_file}")
    
    return validation_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validação do pipeline multisource")
    parser.add_argument("--min-years", type=int, default=3, help="Mínimo de anos de cobertura")
    parser.add_argument("--min-volume", type=int, default=5000, help="Mínimo de notícias")
    
    args = parser.parse_args()
    
    result = run_full_validation(min_years=args.min_years, min_volume=args.min_volume)
    
    # Exit code baseado no resultado
    if result['status'] == 'FAILED':
        sys.exit(1)
    elif result['status'] == 'WARNING':
        sys.exit(2)
    else:
        sys.exit(0)
