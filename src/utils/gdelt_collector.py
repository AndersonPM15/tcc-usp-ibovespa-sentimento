"""
Coletor de notícias GDELT (Global Database of Events, Language, and Tone)

GDELT é uma fonte pública e gratuita com cobertura histórica completa desde 2015.
Permite coleta de notícias globais filtradas por país, idioma e palavras-chave.

Documentação: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path


class GDELTCollector:
    """
    Coletor de notícias via GDELT 2.0 GKG API
    
    GDELT oferece:
    - Histórico desde 2015 (sem limitação de tempo)
    - Cobertura global em múltiplos idiomas
    - API pública gratuita (rate limit: ~50 requisições/minuto)
    - Dados atualizados a cada 15 minutos
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(self, rate_limit_delay: float = 1.5):
        """
        Args:
            rate_limit_delay: Pausa entre requisições (segundos) para evitar rate limit
        """
        self.rate_limit_delay = rate_limit_delay
        
    def collect_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        query: str = "(Ibovespa OR Bovespa OR bolsa OR ações OR mercado financeiro) sourcelang:por",
        max_records: int = 250,
    ) -> pd.DataFrame:
        """
        Coleta notícias em português do Brasil por intervalo de datas.
        
        GDELT API permite busca com:
        - Filtro temporal: últimos X dias, ou datas específicas
        - Filtro de idioma: sourcelang:por
        - Palavras-chave: boolean query (AND, OR, NOT)
        - Max records: até 250 por requisição (limitação da API)
        
        Estratégia: Loop dia-a-dia ou semana-a-semana para cobrir período completo
        
        Args:
            start_date: Data inicial
            end_date: Data final
            query: Query de busca com keywords e filtros
            max_records: Máximo de registros por requisição (250 é o limite da API)
            
        Returns:
            DataFrame com colunas: [date, source, title, url, seendate, domain, language]
        """
        print(f"🌍 GDELT: Coletando {start_date.date()} → {end_date.date()}")
        
        all_articles = []
        current_date = start_date
        
        # Loop dia-a-dia (ou ajuste para semanas se volume for alto)
        while current_date <= end_date:
            try:
                # GDELT usa formato "YYYYMMDDHHMMSS" para datas
                # Para busca de 1 dia, usamos startdatetime e enddatetime
                day_start = current_date.strftime("%Y%m%d") + "000000"
                day_end = current_date.strftime("%Y%m%d") + "235959"
                
                # Parâmetros da API
                params = {
                    "query": query,
                    "mode": "artlist",  # Lista de artigos (não agregação)
                    "maxrecords": max_records,
                    "format": "json",
                    "startdatetime": day_start,
                    "enddatetime": day_end,
                    "sort": "hybridrel",  # Relevância híbrida
                }
                
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    if articles:
                        print(f"  ✅ {current_date.date()}: {len(articles)} artigos")
                        all_articles.extend(articles)
                    else:
                        print(f"  ⚠️ {current_date.date()}: sem dados")
                        
                elif response.status_code == 429:
                    print(f"  ⏳ Rate limit atingido - pausando 60s...")
                    time.sleep(60)
                    continue  # Tenta novamente este dia
                    
                else:
                    print(f"  ❌ {current_date.date()}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"  ⏱️ {current_date.date()}: Timeout - pulando")
            except Exception as e:
                print(f"  ❌ {current_date.date()}: Erro {type(e).__name__}: {str(e)[:100]}")
            
            # Próximo dia
            current_date += timedelta(days=1)
            time.sleep(self.rate_limit_delay)
            
        # Converter para DataFrame
        if not all_articles:
            print("⚠️ Nenhum artigo coletado no período")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_articles)
        
        # Normalizar colunas GDELT para schema unificado
        df_normalized = self._normalize_gdelt_dataframe(df)
        
        print(f"✅ Total coletado: {len(df_normalized)} artigos")
        print(f"   Período efetivo: {df_normalized['date'].min()} → {df_normalized['date'].max()}")
        print(f"   Fontes únicas: {df_normalized['source'].nunique()}")
        
        return df_normalized
    
    def _normalize_gdelt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza DataFrame GDELT para schema unificado do projeto.
        
        GDELT retorna:
        - url: URL do artigo
        - url_mobile: URL mobile (se disponível)
        - title: Título do artigo
        - seendate: Data de captura pelo GDELT (formato YYYYMMDDHHMMSS)
        - socialimage: URL da imagem social
        - domain: Domínio da fonte
        - language: Código do idioma
        - sourcecountry: País da fonte
        
        Schema unificado (RÍGIDO - TCC):
        - date: datetime normalizado UTC, sem timezone (date only)
        - source: nome do domínio
        - title: título do artigo
        - url: link do artigo
        - text_full: título (GDELT não fornece texto completo via API)
        """
        if df.empty:
            return pd.DataFrame(columns=["date", "source", "title", "url", "text_full"])
        
        # CRÍTICO: Parse robusto de seendate
        if "seendate" in df.columns:
            # Converte para string e limpa
            df["seendate_str"] = df["seendate"].astype(str).str.strip()
            
            # Tenta formato GDELT padrão: YYYYMMDDHHMMSS (14 dígitos)
            df["date"] = pd.to_datetime(df["seendate_str"], format="%Y%m%d%H%M%S", errors="coerce")
            
            # Fallback: se não funcionou, tenta parse genérico
            mask_failed = df["date"].isna()
            if mask_failed.any():
                print(f"⚠️ {mask_failed.sum()} datas falharam no parse GDELT - tentando fallback...")
                # Parse fallback com normalização UTC imediata
                dates_fallback = pd.to_datetime(df.loc[mask_failed, "seendate"], errors="coerce")
                # Normalizar e remover timezone ANTES de atribuir
                dates_fallback = dates_fallback.dt.normalize().dt.tz_localize(None)
                df.loc[mask_failed, "date"] = dates_fallback
            
            # Verificação final
            still_failed = df["date"].isna().sum()
            if still_failed > 0:
                print(f"❌ {still_failed} datas inválidas após todos os parsers - serão removidas")
        else:
            raise ValueError("GDELT retornou dados sem coluna 'seendate' - coleta inválida")
        
        # Normalizar para meia-noite UTC (remove hora, mantém apenas date)
        # Já fizemos isso no fallback, mas garantir para datas que passaram no parse principal
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["date"] = df["date"].dt.normalize()
        
        # Source: usar domínio como identificador
        df["source"] = df.get("domain", "gdelt_unknown").fillna("gdelt_unknown")
        
        # Title
        df["title"] = df.get("title", "").fillna("")
        
        # URL
        df["url"] = df.get("url", "").fillna("")
        
        # Text: GDELT API não fornece texto completo, apenas título
        df["text_full"] = df["title"]  # Título é o melhor disponível
        
        # Selecionar colunas finais
        result = df[["date", "source", "title", "url", "text_full"]].copy()
        
        # Filtros de qualidade
        result = result.dropna(subset=["date", "title"])  # Data e título obrigatórios
        result = result[result["title"].str.len() >= 15]  # Títulos muito curtos são inválidos
        result = result[result["url"].str.len() > 0]  # URL obrigatória
        result = result.drop_duplicates(subset=["url"])  # URL única (primário)
        
        return result.reset_index(drop=True)
    
    def collect_recent(self, days: int = 7, query: str = "(Ibovespa OR Bovespa) sourcelang:por") -> pd.DataFrame:
        """
        Atalho para coletar últimos N dias.
        
        Args:
            days: Número de dias recentes
            query: Query de busca
            
        Returns:
            DataFrame normalizado
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.collect_by_date_range(start_date, end_date, query)


def collect_gdelt_historical(
    start_date: str,
    end_date: str,
    query: str = "(Ibovespa OR Bovespa OR B3 OR 'Bolsa de valores' OR ações OR mercado) sourcelang:por",
    output_path: Optional[Path] = None,
    checkpoint_interval: int = 30,
    min_days_threshold: int = 200,
) -> pd.DataFrame:
    """
    Função auxiliar para coleta histórica GDELT com checkpoints e validação.
    
    CHECAGENS OBRIGATÓRIAS (TCC):
    - Mínimo de dias distintos configurável (padrão: 200)
    - Erro fatal se base resultante for insuficiente
    
    Args:
        start_date: Data inicial (formato "YYYY-MM-DD")
        end_date: Data final (formato "YYYY-MM-DD")
        query: Query GDELT com termos financeiros PT-BR
        output_path: Caminho para salvar resultado final
        checkpoint_interval: Salvar checkpoint a cada N dias
        min_days_threshold: Mínimo de dias distintos aceitável (padrão: 200)
        
    Returns:
        DataFrame consolidado
        
    Raises:
        RuntimeError: Se base resultante tiver menos de min_days_threshold dias
    """
    collector = GDELTCollector()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"\n{'='*80}")
    print(f"COLETA HISTÓRICA GDELT: {start_date} → {end_date}")
    print(f"Query: {query[:60]}...")
    print(f"Checagem mínima: {min_days_threshold} dias distintos")
    print(f"{'='*80}\n")
    
    # Coleta com checkpoint
    all_data = []
    current = start_dt
    checkpoint_counter = 0
    
    # Dividir em chunks de 30 dias para gerenciar memória
    chunk_days = 30
    
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        
        df_chunk = collector.collect_by_date_range(
            start_date=current,
            end_date=chunk_end,
            query=query,
            max_records=250,
        )
        
        if not df_chunk.empty:
            all_data.append(df_chunk)
        
        checkpoint_counter += chunk_days
        
        # Checkpoint a cada intervalo
        if checkpoint_counter >= checkpoint_interval and output_path and len(all_data) > 0:
            df_temp = pd.concat(all_data, ignore_index=True)
            df_temp = df_temp.drop_duplicates(subset=["url"])
            checkpoint_file = output_path.parent / f"{output_path.stem}_checkpoint.parquet"
            df_temp.to_parquet(checkpoint_file, index=False)
            print(f"💾 Checkpoint: {len(df_temp):,} artigos ({df_temp['date'].nunique():,} dias) → {checkpoint_file.name}")
        
        current = chunk_end + timedelta(days=1)
    
    # Consolidar resultado final
    if not all_data:
        raise RuntimeError(
            f"[GDELT] ERRO CRÍTICO: Nenhum dado coletado no período {start_date} → {end_date}. "
            f"Verifique conexão, query ou limitações da API GDELT."
        )
    
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["url"])
    df_final = df_final.sort_values("date").reset_index(drop=True)
    
    # CHECAGEM OBRIGATÓRIA
    n_days = df_final["date"].nunique()
    n_news = len(df_final)
    
    print(f"\n{'='*80}")
    print(f"✅ COLETA CONCLUÍDA")
    print(f"   Total: {n_news:,} artigos únicos")
    print(f"   Período: {df_final['date'].min().date()} → {df_final['date'].max().date()}")
    print(f"   Dias distintos: {n_days:,}")
    print(f"   Fontes: {df_final['source'].nunique():,}")
    print(f"   Média: {n_news / n_days:.1f} artigos/dia")
    print(f"{'='*80}\n")
    
    # Validação de limiar mínimo
    if n_days < min_days_threshold:
        raise RuntimeError(
            f"[GDELT] Base de notícias INSUFICIENTE para TCC:\n"
            f"   Dias coletados: {n_days}\n"
            f"   Mínimo exigido: {min_days_threshold}\n"
            f"   Período solicitado: {start_date} → {end_date}\n"
            f"   AÇÃO: Revisar query, período ou limitações do GDELT para idioma PT-BR."
        )
    
    # Salvar resultado final
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_path, index=False)
        print(f"💾 Arquivo salvo: {output_path}")
        print(f"   Tamanho: {output_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    return df_final


if __name__ == "__main__":
    # Teste rápido: últimos 7 dias
    print("🧪 Teste do coletor GDELT - últimos 7 dias\n")
    
    collector = GDELTCollector()
    df_test = collector.collect_recent(days=7)
    
    if not df_test.empty:
        print("\n📊 Amostra dos dados coletados:")
        print(df_test.head(10))
        print(f"\nShape: {df_test.shape}")
        print(f"Colunas: {df_test.columns.tolist()}")
    else:
        print("⚠️ Nenhum dado retornado no teste")
