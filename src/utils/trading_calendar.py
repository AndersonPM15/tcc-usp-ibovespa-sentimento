"""
Calendário de feriados e dias úteis da B3 (Brasil, Bolsa, Balcão).
Fornece funções para identificar dias de negociação válidos.
"""

import pandas as pd
from datetime import date, datetime
from typing import Union, List
import numpy as np


# ============================================================================
# FERIADOS FIXOS BRASILEIROS QUE AFETAM A B3
# ============================================================================
FERIADOS_FIXOS = {
    (1, 1): "Ano Novo",
    (4, 21): "Tiradentes",
    (5, 1): "Dia do Trabalho",
    (9, 7): "Independência do Brasil",
    (10, 12): "Nossa Senhora Aparecida",
    (11, 2): "Finados",
    (11, 15): "Proclamação da República",
    (11, 20): "Consciência Negra",  # Feriado nacional desde 2024
    (12, 25): "Natal",
}


def _calcular_pascoa(ano: int) -> date:
    """
    Calcula a data da Páscoa para um ano usando o algoritmo de Meeus/Jones/Butcher.
    
    Args:
        ano: Ano para calcular a Páscoa
        
    Returns:
        Data da Páscoa
    """
    a = ano % 19
    b = ano // 100
    c = ano % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


def _obter_feriados_moveis(ano: int) -> List[tuple]:
    """
    Calcula feriados móveis da B3 para um ano específico.
    
    Args:
        ano: Ano de referência
        
    Returns:
        Lista de tuplas (data, nome_feriado)
    """
    pascoa = _calcular_pascoa(ano)
    
    feriados_moveis = []
    
    # Carnaval: 47 dias antes da Páscoa (terça-feira)
    carnaval = pascoa - pd.Timedelta(days=47)
    feriados_moveis.append((carnaval.date(), "Carnaval"))
    
    # Sexta-feira Santa: 2 dias antes da Páscoa
    sexta_santa = pascoa - pd.Timedelta(days=2)
    feriados_moveis.append((sexta_santa.date(), "Sexta-feira Santa"))
    
    # Corpus Christi: 60 dias após a Páscoa (quinta-feira)
    corpus_christi = pascoa + pd.Timedelta(days=60)
    feriados_moveis.append((corpus_christi.date(), "Corpus Christi"))
    
    return feriados_moveis


def obter_feriados_b3(start: Union[date, str], end: Union[date, str]) -> pd.DatetimeIndex:
    """
    Retorna todos os feriados da B3 no período especificado.
    
    Args:
        start: Data inicial (date ou string 'YYYY-MM-DD')
        end: Data final (date ou string 'YYYY-MM-DD')
        
    Returns:
        DatetimeIndex com os feriados do período
    """
    # Converter strings para date se necessário
    if isinstance(start, str):
        start = pd.to_datetime(start).date()
    if isinstance(end, str):
        end = pd.to_datetime(end).date()
    
    # Lista de todos os feriados
    feriados = []
    
    # Iterar pelos anos no período
    for ano in range(start.year, end.year + 1):
        # Feriados fixos
        for (mes, dia), nome in FERIADOS_FIXOS.items():
            feriado = date(ano, mes, dia)
            if start <= feriado <= end:
                feriados.append(feriado)
        
        # Feriados móveis
        feriados_moveis = _obter_feriados_moveis(ano)
        for feriado, nome in feriados_moveis:
            if start <= feriado <= end:
                feriados.append(feriado)
    
    # Converter para DatetimeIndex e remover duplicatas
    return pd.DatetimeIndex(sorted(set(feriados)))


def get_b3_trading_days(start: Union[date, str], end: Union[date, str]) -> pd.DatetimeIndex:
    """
    Retorna apenas os dias úteis de negociação da B3 (excluindo fins de semana e feriados).
    
    Esta é a função principal a ser usada em todo o pipeline para garantir
    que apenas dias de pregão sejam considerados nas análises.
    
    Args:
        start: Data inicial (date ou string 'YYYY-MM-DD')
        end: Data final (date ou string 'YYYY-MM-DD')
        
    Returns:
        DatetimeIndex com os dias úteis de negociação da B3
        
    Example:
        >>> from src.config.constants import START_DATE, END_DATE
        >>> trading_days = get_b3_trading_days(START_DATE, END_DATE)
        >>> print(f"Total de pregões: {len(trading_days)}")
    """
    # Converter strings para date se necessário
    if isinstance(start, str):
        start = pd.to_datetime(start).date()
    if isinstance(end, str):
        end = pd.to_datetime(end).date()
    
    # Obter feriados do período
    feriados = obter_feriados_b3(start, end)
    
    # Criar range de dias úteis (excluindo sábados e domingos)
    business_days = pd.bdate_range(start=start, end=end, freq='B')
    
    # Remover feriados da B3
    trading_days = business_days.difference(feriados)
    
    return trading_days


def is_trading_day(data: Union[date, str, pd.Timestamp]) -> bool:
    """
    Verifica se uma data específica é dia de negociação na B3.
    
    Args:
        data: Data a ser verificada
        
    Returns:
        True se for dia de negociação, False caso contrário
    """
    if isinstance(data, str):
        data = pd.to_datetime(data).date()
    elif isinstance(data, pd.Timestamp):
        data = data.date()
    
    # Verificar se é fim de semana
    if data.weekday() >= 5:  # 5=sábado, 6=domingo
        return False
    
    # Verificar se é feriado
    feriados = obter_feriados_b3(data, data)
    return len(feriados) == 0


def filter_trading_days_only(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Filtra um DataFrame para manter apenas linhas com dias de negociação da B3.
    
    Args:
        df: DataFrame com coluna de data
        date_col: Nome da coluna de data (padrão: 'date')
        
    Returns:
        DataFrame filtrado apenas com dias de pregão
    """
    df = df.copy()
    
    # Garantir que a coluna é datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Obter range de trading days
    start = df[date_col].min().date()
    end = df[date_col].max().date()
    trading_days = get_b3_trading_days(start, end)
    
    # Filtrar
    df_filtered = df[df[date_col].dt.normalize().isin(trading_days)]
    
    n_removed = len(df) - len(df_filtered)
    if n_removed > 0:
        print(f"🗓️ Removidos {n_removed} registros de dias não-úteis (fins de semana + feriados B3)")
    
    return df_filtered


# ============================================================================
# EXEMPLOS DE USO
# ============================================================================
if __name__ == "__main__":
    from src.config.constants import START_DATE, END_DATE
    
    print("=" * 70)
    print("CALENDÁRIO B3 - DIAS ÚTEIS DE NEGOCIAÇÃO")
    print("=" * 70)
    
    # Obter trading days do período do TCC
    trading_days = get_b3_trading_days(START_DATE, END_DATE)
    print(f"\n📅 Período: {START_DATE} a {END_DATE}")
    print(f"📊 Total de pregões: {len(trading_days)}")
    print(f"📊 Pregões por ano: {len(trading_days) / ((END_DATE - START_DATE).days / 365.25):.1f}")
    
    # Listar feriados
    feriados = obter_feriados_b3(START_DATE, END_DATE)
    print(f"\n🎉 Total de feriados: {len(feriados)}")
    print("\nPrimeiros 10 feriados:")
    for feriado in feriados[:10]:
        print(f"  - {feriado.strftime('%Y-%m-%d (%A)')}")
