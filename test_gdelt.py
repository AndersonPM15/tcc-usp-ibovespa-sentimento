from datetime import datetime, timedelta
from src.utils.gdelt_collector import GDELTCollector

c = GDELTCollector(rate_limit_delay=0.5)
end = datetime.now()
start = end - timedelta(days=2)
df = c.collect_by_date_range(start, end)

print(f"\n✅ Teste: {len(df)} artigos")
if not df.empty:
    print(f"Datas: {df['date'].min()} → {df['date'].max()}")
    print(f"Dias únicos: {df['date'].nunique()}")
    print(f"\nAmostra:")
    print(df.head(3))
else:
    print("⚠️ Nenhum dado retornado")
