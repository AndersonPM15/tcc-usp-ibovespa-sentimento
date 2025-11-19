"""
Audit Final de Cobertura - TCC USP Ibovespa Sentiment
======================================================
Gera relatório de cobertura da base de dados após coleta GDELT histórica.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

BASE_PATH = Path(__file__).parent.parent  # Subir um nível para TCC_USP
DATA_RAW = BASE_PATH / "data_raw"
DATA_INTERIM = BASE_PATH / "data_interim"
DATA_PROC = BASE_PATH / "data_processed"

def check_file(path: Path):
    """Retorna estatísticas de um arquivo parquet/csv"""
    if not path.exists():
        return None
    
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            return None
        
        # Detectar coluna de data
        date_col = None
        for col in ["date", "day", "data"]:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            n_days = df[date_col].nunique()
            date_min = df[date_col].min()
            date_max = df[date_col].max()
        else:
            n_days = None
            date_min = None
            date_max = None
        
        return {
            "arquivo": path.name,
            "caminho": str(path.relative_to(BASE_PATH)),
            "registros": len(df),
            "dias_distintos": n_days,
            "data_min": date_min,
            "data_max": date_max,
            "tamanho_mb": path.stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        print(f"❌ Erro ao processar {path.name}: {e}")
        return None

def main():
    print("="*80)
    print("RELATÓRIO FINAL DE COBERTURA - TCC USP")
    print("Gerado em:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    # Arquivos-chave da pipeline
    files_to_check = [
        DATA_RAW / "news_multisource.parquet",
        DATA_INTERIM / "news_clean_multisource.parquet",
        DATA_PROC / "news_clean.parquet",
        DATA_PROC / "tfidf_daily_index.csv",
        DATA_PROC / "ibovespa_clean.csv"
    ]
    
    results = []
    for file_path in files_to_check:
        result = check_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ ERRO: Nenhum arquivo encontrado para análise")
        print("Execute o notebook 12 em modo FULL primeiro!")
        return
    
    # Criar tabela de resultados
    df_report = pd.DataFrame(results)
    
    # Formatar para exibição
    print("\n" + "="*80)
    print("TABELA DE COBERTURA")
    print("="*80)
    
    for _, row in df_report.iterrows():
        print(f"\n📄 {row['arquivo']}")
        print(f"   Caminho: {row['caminho']}")
        print(f"   Registros: {row['registros']:,}")
        if row['dias_distintos']:
            print(f"   Dias distintos: {row['dias_distintos']:,}")
            print(f"   Período: {row['data_min'].date()} → {row['data_max'].date()}")
        print(f"   Tamanho: {row['tamanho_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("VALIDAÇÃO DE COBERTURA")
    print("="*80)
    
    # Verificar cobertura mínima de 200 dias
    MIN_DAYS = 200
    news_multisource = df_report[df_report["arquivo"] == "news_multisource.parquet"]
    tfidf_index = df_report[df_report["arquivo"] == "tfidf_daily_index.csv"]
    ibov_clean = df_report[df_report["arquivo"] == "ibovespa_clean.csv"]
    
    status_ok = True
    
    if not news_multisource.empty:
        days = news_multisource.iloc[0]["dias_distintos"]
        if days >= MIN_DAYS:
            print(f"✅ news_multisource.parquet: {days:,} dias >= {MIN_DAYS}")
        else:
            print(f"❌ news_multisource.parquet: {days:,} dias < {MIN_DAYS}")
            status_ok = False
    else:
        print("❌ news_multisource.parquet: Arquivo não encontrado")
        status_ok = False
    
    if not tfidf_index.empty:
        days = tfidf_index.iloc[0]["dias_distintos"]
        if days >= MIN_DAYS:
            print(f"✅ tfidf_daily_index.csv: {days:,} dias >= {MIN_DAYS}")
        else:
            print(f"❌ tfidf_daily_index.csv: {days:,} dias < {MIN_DAYS}")
            status_ok = False
    else:
        print("⚠️ tfidf_daily_index.csv: Arquivo não encontrado (execute notebooks 13-15)")
    
    if not ibov_clean.empty:
        days = ibov_clean.iloc[0]["dias_distintos"]
        print(f"✅ ibovespa_clean.csv: {days:,} dias (referência mercado)")
    
    print("\n" + "="*80)
    print("STATUS FINAL")
    print("="*80)
    
    if status_ok:
        print("✅ APROVADO: Base histórica com cobertura suficiente para TCC")
        print(f"   Utilizando GDELT como fonte oficial, a base cobre:")
        if not news_multisource.empty:
            row = news_multisource.iloc[0]
            print(f"   - Período: {row['data_min'].date()} → {row['data_max'].date()}")
            print(f"   - Dias: {row['dias_distintos']:,}")
            print(f"   - Notícias: {row['registros']:,}")
    else:
        print("❌ REPROVADO: Base insuficiente para TCC")
        print("   Execute o notebook 12 em modo FULL para coletar base histórica completa")
    
    print("="*80)
    
    # Salvar relatório em CSV
    output_file = BASE_PATH / "reports" / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file.parent.mkdir(exist_ok=True)
    df_report.to_csv(output_file, index=False)
    print(f"\n📊 Relatório salvo em: {output_file}")

if __name__ == "__main__":
    main()
