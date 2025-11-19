"""
Funções auxiliares para coleta multisource de notícias.
Utilizado por notebooks/12_data_collection_multisource.ipynb
"""

import requests
import time
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta


def fetch_gdelt_articles(query: str, start_date: str, end_date: str, max_records: int = 250) -> List[Dict]:
    """
    Coleta artigos do GDELT Project usando API v2 DOC.
    
    Args:
        query: Termo de busca
        start_date: Data inicial (YYYYMMDDHHMMSS)
        end_date: Data final (YYYYMMDDHHMMSS)
        max_records: Máximo de registros (padrão 250)
    
    Returns:
        Lista de dicionários com artigos
    """
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    params = {
        "query": f'{query} sourcelang:portuguese',
        "mode": "artlist",
        "maxrecords": max_records,
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get('articles', [])
        print(f"  ✓ GDELT: {len(articles)} artigos para '{query}' ({start_date[:8]}-{end_date[:8]})")
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Erro GDELT para '{query}': {e}")
        return []
    except Exception as e:
        print(f"  ✗ Erro inesperado: {e}")
        return []


def collect_gdelt_batch(start_date: pd.Timestamp, end_date: pd.Timestamp, 
                       terms: List[str], stamp: str) -> pd.DataFrame:
    """
    Coleta GDELT em batches para evitar timeout.
    Divide período em chunks de 3 meses.
    """
    all_articles = []
    current = start_date
    
    while current < end_date:
        # Chunk de 3 meses
        chunk_end = min(current + pd.DateOffset(months=3), end_date)
        
        start_str = current.strftime("%Y%m%d%H%M%S")
        end_str = chunk_end.strftime("%Y%m%d%H%M%S")
        
        print(f"\n📥 Coletando GDELT: {current.date()} → {chunk_end.date()}")
        
        for term in terms:
            articles = fetch_gdelt_articles(term, start_str, end_str, max_records=250)
            
            for art in articles:
                all_articles.append({
                    'id': f"gdelt_{hash(art.get('url', ''))}", 
                    'source': art.get('domain', 'gdelt'),
                    'title': art.get('title', ''),
                    'description': '',
                    'content': art.get('title', ''),
                    'published_at': art.get('seendate', ''),
                    'author': '',
                    'url': art.get('url', ''),
                    'raw_text': art.get('title', ''),
                    'scraped_text': '',
                    'query_term': term,
                    'source_type': 'gdelt',
                    'language': art.get('language', 'pt'),
                    'collected_at': stamp
                })
            
            time.sleep(1)  # Rate limiting
        
        current = chunk_end
        time.sleep(2)  # Pausa entre chunks
    
    df = pd.DataFrame(all_articles)
    print(f"\n✅ GDELT Total: {len(df)} artigos coletados")
    return df


def collect_gnews(terms: List[str], max_results: int, stamp: str) -> pd.DataFrame:
    """
    Coleta notícias usando GNews (últimos 6 meses - limitação da biblioteca).
    """
    from gnews import GNews
    
    google_news = GNews(
        language='pt',
        country='BR',
        period='6m',  # Últimos 6 meses
        max_results=max_results
    )
    
    all_articles = []
    
    print("\n📥 Coletando GNews (últimos 6 meses)...")
    
    for term in terms:
        try:
            articles = google_news.get_news(term)
            print(f"  ✓ GNews: {len(articles)} artigos para '{term}'")
            
            for art in articles:
                all_articles.append({
                    'id': f"gnews_{hash(art.get('url', ''))}",
                    'source': art.get('publisher', {}).get('title', 'gnews'),
                    'title': art.get('title', ''),
                    'description': art.get('description', ''),
                    'content': art.get('description', ''),
                    'published_at': art.get('published date', ''),
                    'author': '',
                    'url': art.get('url', ''),
                    'raw_text': art.get('title', '') + ' ' + art.get('description', ''),
                    'scraped_text': '',
                    'query_term': term,
                    'source_type': 'gnews',
                    'language': 'pt',
                    'collected_at': stamp
                })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ Erro GNews para '{term}': {e}")
            continue
    
    df = pd.DataFrame(all_articles)
    print(f"\n✅ GNews Total: {len(df)} artigos")
    return df


def collect_rss_feeds(stamp: str) -> pd.DataFrame:
    """
    Coleta feeds RSS de jornais financeiros brasileiros.
    """
    import feedparser
    from bs4 import BeautifulSoup
    
    RSS_SOURCES = {
        'valor': 'https://valor.globo.com/rss/home/',
        'infomoney': 'https://www.infomoney.com.br/feed/',
        'exame': 'https://exame.com/feed/',
        'seudinheiro': 'https://www.seudinheiro.com/feed/',
        'investing_br': 'https://br.investing.com/rss/news.rss',
        'reuters_br': 'https://www.reuters.com/rssFeed/brasil'
    }
    
    all_articles = []
    
    print("\n📥 Coletando RSS Feeds...")
    
    for source_name, rss_url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(rss_url)
            entries = feed.get('entries', [])
            
            print(f"  ✓ RSS {source_name}: {len(entries)} artigos")
            
            for entry in entries:
                # Extrair texto do summary se disponível
                summary = entry.get('summary', '')
                if summary:
                    soup = BeautifulSoup(summary, 'html.parser')
                    summary_text = soup.get_text()
                else:
                    summary_text = ''
                
                all_articles.append({
                    'id': f"rss_{source_name}_{hash(entry.get('id', ''))}",
                    'source': source_name,
                    'title': entry.get('title', ''),
                    'description': summary_text,
                    'content': summary_text,
                    'published_at': entry.get('published', entry.get('updated', '')),
                    'author': entry.get('author', ''),
                    'url': entry.get('link', ''),
                    'raw_text': entry.get('title', '') + ' ' + summary_text,
                    'scraped_text': summary_text,
                    'query_term': 'rss_feed',
                    'source_type': 'rss',
                    'language': 'pt',
                    'collected_at': stamp
                })
            
        except Exception as e:
            print(f"  ✗ Erro RSS {source_name}: {e}")
            continue
    
    df = pd.DataFrame(all_articles)
    print(f"\n✅ RSS Total: {len(df)} artigos")
    return df


def collect_newsapi(terms: List[str], days_back: int, stamp: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Coleta NewsAPI (limitado a ~30 dias free tier).
    """
    import os
    
    if not api_key:
        api_key = os.environ.get("NEWSAPI_KEY", None)
    
    if not api_key:
        print("⚠️ NewsAPI KEY não configurada - pulando coleta NewsAPI")
        return pd.DataFrame()
    
    all_articles = []
    base_url = "https://newsapi.org/v2/everything"
    
    # NewsAPI free tier: últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(days_back, 30))
    
    print(f"\n📥 Coletando NewsAPI ({start_date.date()} → {end_date.date()})...")
    print("⚠️ Free tier limitado a 30 dias")
    
    for term in terms:
        try:
            params = {
                'q': term,
                'language': 'pt',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': api_key
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"  ✓ NewsAPI: {len(articles)} artigos para '{term}'")
                
                for art in articles:
                    all_articles.append({
                        'id': f"newsapi_{hash(art.get('url', ''))}",
                        'source': art.get('source', {}).get('name', 'newsapi'),
                        'title': art.get('title', ''),
                        'description': art.get('description', ''),
                        'content': art.get('content', ''),
                        'published_at': art.get('publishedAt', ''),
                        'author': art.get('author', ''),
                        'url': art.get('url', ''),
                        'raw_text': art.get('title', '') + ' ' + art.get('description', ''),
                        'scraped_text': '',
                        'query_term': term,
                        'source_type': 'newsapi',
                        'language': 'pt',
                        'collected_at': stamp
                    })
            else:
                print(f"  ✗ NewsAPI erro {response.status_code} para '{term}'")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Erro NewsAPI para '{term}': {e}")
            continue
    
    df = pd.DataFrame(all_articles)
    print(f"\n✅ NewsAPI Total: {len(df)} artigos")
    return df


def collect_cvm_fatos_relevantes(start_date: pd.Timestamp, end_date: pd.Timestamp, stamp: str) -> pd.DataFrame:
    """
    Coleta Fatos Relevantes da CVM (Comissão de Valores Mobiliários).
    
    Fonte: Sistema de Informações Periódicas da CVM
    Endpoint: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/
    
    Args:
        start_date: Data inicial da coleta
        end_date: Data final da coleta
        stamp: Timestamp da coleta
    
    Returns:
        DataFrame com fatos relevantes no schema padrão
    """
    all_articles = []
    
    print(f"\n📥 Coletando CVM - Fatos Relevantes ({start_date.date()} → {end_date.date()})...")
    
    # A CVM disponibiliza arquivos por ano
    # Formato: fre_cia_aberta_YYYY.csv
    base_url = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS"
    
    for year in range(start_date.year, end_date.year + 1):
        try:
            # URL do arquivo anual
            file_url = f"{base_url}/fre_cia_aberta_{year}.csv"
            
            print(f"  📄 Tentando baixar: fre_cia_aberta_{year}.csv")
            
            # Baixar CSV com encoding latin1 (padrão CVM)
            df_year = pd.read_csv(
                file_url,
                sep=';',
                encoding='latin1',
                decimal=',',
                thousands='.',
                low_memory=False
            )
            
            # Converter data de referência
            if 'DT_REFER' in df_year.columns:
                df_year['DT_REFER'] = pd.to_datetime(df_year['DT_REFER'], format='%Y-%m-%d', errors='coerce')
            
            # Filtrar pelo período
            df_year = df_year[
                (df_year['DT_REFER'] >= start_date) & 
                (df_year['DT_REFER'] <= end_date)
            ]
            
            print(f"  ✓ CVM: {len(df_year)} fatos relevantes em {year}")
            
            # Processar cada fato relevante
            for _, row in df_year.iterrows():
                # Campos típicos do CSV da CVM
                cnpj = row.get('CNPJ_CIA', '')
                denom_cia = row.get('DENOM_CIA', 'Empresa não identificada')
                data_refer = row.get('DT_REFER', '')
                descricao = row.get('DESCRICAO_FATO', '')
                
                # Construir URL para visualização (se disponível)
                # Formato típico: https://www.rad.cvm.gov.br/...
                url_cvm = f"https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx?CNPJ={cnpj}"
                
                # Criar registro no schema padrão
                all_articles.append({
                    'id': f"cvm_fr_{hash(f'{cnpj}_{data_refer}_{descricao}')}",
                    'source': 'CVM_FR',
                    'title': f"Fato Relevante: {denom_cia}",
                    'description': descricao[:500] if descricao else '',  # Limitar tamanho
                    'content': descricao,
                    'published_at': data_refer,
                    'author': 'CVM',
                    'url': url_cvm,
                    'raw_text': f"{denom_cia} {descricao}",
                    'scraped_text': descricao,
                    'query_term': 'fato_relevante',
                    'source_type': 'cvm',
                    'language': 'pt',
                    'collected_at': stamp
                })
            
            time.sleep(1)  # Rate limiting cortesia
            
        except Exception as e:
            print(f"  ✗ Erro ao coletar CVM {year}: {e}")
            # Se falhar em um ano, continua para os próximos
            continue
    
    df = pd.DataFrame(all_articles)
    print(f"\n✅ CVM Total: {len(df)} fatos relevantes")
    
    return df
