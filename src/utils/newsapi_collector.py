"""
Coletor de notícias NewsAPI

NewsAPI é uma fonte premium com excelente qualidade, mas limitações:
- Plano FREE: apenas últimos 30 dias
- Plano Developer ($449/mês): até 2 anos de histórico

Usamos NewsAPI como fonte COMPLEMENTAR para dados recentes (últimos 30 dias).
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional


class NewsAPICollector:
    """
    Coletor de notícias via NewsAPI
    
    Limitações do plano gratuito:
    - Histórico: apenas últimos 30 dias
    - Rate limit: 100 requisições/dia
    - Max results: 100 por requisição
    """
    
    BASE_URL = "https://newsapi.org/v2/everything"
    
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        """
        Args:
            api_key: Chave da API NewsAPI
            rate_limit_delay: Pausa entre requisições (segundos)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        
    def collect_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        query: str = "Ibovespa OR Bovespa OR bolsa OR ações OR mercado financeiro OR economia OR dólar",
        language: str = "pt",
        page_size: int = 100,
        max_pages: int = 10,
    ) -> pd.DataFrame:
        """
        Coleta notícias em português por intervalo de datas.
        
        Args:
            start_date: Data inicial
            end_date: Data final
            query: Query de busca (ampliada para capturar mais artigos)
            language: Idioma das notícias (pt = português)
            page_size: Artigos por página (max 100)
            max_pages: Máximo de páginas a coletar
            
        Returns:
            DataFrame com colunas: [date, source, title, description, url, content]
        """
        # NewsAPI aceita datas no formato YYYY-MM-DD
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        print(f"📰 NewsAPI: Coletando {from_date} → {to_date}")
        print(f"   Query: {query[:80]}...")
        
        all_articles = []
        
        for page in range(1, max_pages + 1):
            try:
                params = {
                    "q": query,
                    "from": from_date,
                    "to": to_date,
                    "language": language,
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "page": page,
                    "apiKey": self.api_key,
                }
                
                response = requests.get(self.BASE_URL, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    total_results = data.get("totalResults", 0)
                    
                    if not articles:
                        print(f"  ℹ️ Página {page}: sem mais resultados")
                        break
                    
                    print(f"  ✅ Página {page}: {len(articles)} artigos (total disponível: {total_results})")
                    all_articles.extend(articles)
                    
                    # Se retornou menos que page_size, não há mais páginas
                    if len(articles) < page_size:
                        print(f"  ℹ️ Última página alcançada")
                        break
                        
                elif response.status_code == 426:
                    print(f"  ⚠️ Upgrade necessário - plano FREE limita histórico a 30 dias")
                    break
                    
                elif response.status_code == 429:
                    print(f"  ⏳ Rate limit atingido - pausando 60s...")
                    time.sleep(60)
                    continue
                    
                else:
                    error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    error_msg = error_data.get("message", response.text[:200])
                    print(f"  ❌ HTTP {response.status_code}: {error_msg}")
                    break
                    
            except requests.exceptions.Timeout:
                print(f"  ⏱️ Timeout na página {page}")
            except Exception as e:
                print(f"  ❌ Erro na página {page}: {type(e).__name__}: {str(e)[:100]}")
                
            time.sleep(self.rate_limit_delay)
        
        # Converter para DataFrame
        if not all_articles:
            print("⚠️ Nenhum artigo coletado")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_articles)
        df_normalized = self._normalize_newsapi_dataframe(df)
        
        print(f"✅ Total: {len(df_normalized)} artigos")
        if not df_normalized.empty:
            print(f"   Período: {df_normalized['date'].min()} → {df_normalized['date'].max()}")
            print(f"   Dias distintos: {df_normalized['date'].nunique()}")
            print(f"   Fontes: {df_normalized['source'].nunique()}")
        
        return df_normalized
    
    def _normalize_newsapi_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza DataFrame NewsAPI para schema unificado.
        
        NewsAPI retorna:
        - source: {"id": "...", "name": "..."}
        - author: autor do artigo
        - title: título
        - description: descrição/resumo
        - url: link do artigo
        - urlToImage: URL da imagem
        - publishedAt: data de publicação (ISO 8601)
        - content: primeiros ~200 chars do conteúdo
        
        Schema unificado:
        - date: datetime normalizado
        - source: nome da fonte
        - title: título
        - url: link
        - text_full: description + content (NewsAPI não dá texto completo)
        """
        if df.empty:
            return pd.DataFrame(columns=["date", "source", "title", "url", "text_full"])
        
        # Parse publishedAt
        df["date"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df["date"] = df["date"].dt.normalize()
        
        # Source: extrair nome do dict
        if "source" in df.columns:
            df["source"] = df["source"].apply(
                lambda s: s.get("name") if isinstance(s, dict) else str(s)
            )
        else:
            df["source"] = "newsapi_unknown"
        
        # Title
        df["title"] = df.get("title", "").fillna("")
        
        # URL
        df["url"] = df.get("url", "").fillna("")
        
        # Text: combinar description + content
        description = df.get("description", "").fillna("")
        content = df.get("content", "").fillna("")
        df["text_full"] = (description.astype(str) + " " + content.astype(str)).str.strip()
        
        # Selecionar colunas finais
        result = df[["date", "source", "title", "url", "text_full"]].copy()
        
        # Remover vazios e duplicatas
        result = result.dropna(subset=["title"])
        result = result[result["title"].str.len() > 10]
        result = result.drop_duplicates(subset=["url", "title"])
        
        return result.reset_index(drop=True)
    
    def collect_recent(self, days: int = 7) -> pd.DataFrame:
        """
        Atalho para coletar últimos N dias.
        
        Args:
            days: Número de dias recentes (max 30 no plano FREE)
            
        Returns:
            DataFrame normalizado
        """
        if days > 30:
            print("⚠️ Plano FREE limita a 30 dias - ajustando para 30")
            days = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.collect_by_date_range(start_date, end_date)


def collect_newsapi_recent(
    api_key: str,
    days: int = 30,
) -> pd.DataFrame:
    """
    Função auxiliar para coleta recente via NewsAPI.
    
    Args:
        api_key: Chave da API
        days: Dias recentes (max 30 no plano FREE)
        
    Returns:
        DataFrame normalizado
    """
    collector = NewsAPICollector(api_key)
    
    print(f"\n{'='*80}")
    print(f"COLETA NewsAPI: Últimos {days} dias")
    print(f"{'='*80}\n")
    
    df = collector.collect_recent(days=days)
    
    if not df.empty:
        print(f"\n{'='*80}")
        print(f"✅ COLETA CONCLUÍDA")
        print(f"   Total: {len(df)} artigos")
        print(f"   Período: {df['date'].min()} → {df['date'].max()}")
        print(f"   Dias distintos: {df['date'].nunique()}")
        print(f"   Fontes: {df['source'].nunique()}")
        print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    # Teste rápido
    print("🧪 Teste do coletor NewsAPI\n")
    print("⚠️ Informe sua chave NewsAPI:")
    api_key = input("API Key: ").strip()
    
    if api_key and api_key != "SUA_CHAVE_AQUI":
        collector = NewsAPICollector(api_key)
        df_test = collector.collect_recent(days=7)
        
        if not df_test.empty:
            print("\n📊 Amostra dos dados coletados:")
            print(df_test.head(10))
            print(f"\nShape: {df_test.shape}")
        else:
            print("⚠️ Nenhum dado retornado")
    else:
        print("❌ Chave inválida - teste cancelado")
