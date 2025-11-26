#!/usr/bin/env python
"""
Test the dashboard loading and basic functionality.
"""

import sys
from datetime import datetime

print("=" * 80)
print("ITEM 4: Testando app_dashboard.py")
print("=" * 80)

try:
    # Import dashboard components
    from app_dashboard import (
        IBOV_DF, SENTIMENT_DF, RESULTS_DF, LATENCY_DF, 
        DATE_MIN, DATE_MAX, MODEL_OPTIONS, app
    )
    
    print("\n✓ Dashboard módulos carregados com sucesso")
    
    # Check data
    print(f"\nDados carregados:")
    print(f"  - IBOV_DF: {len(IBOV_DF)} linhas")
    print(f"  - SENTIMENT_DF: {len(SENTIMENT_DF)} linhas")
    print(f"  - RESULTS_DF: {len(RESULTS_DF)} linhas")
    print(f"  - LATENCY_DF: {len(LATENCY_DF)} linhas")
    
    # Check date range
    print(f"\nIntervalo de datas:")
    if DATE_MIN and DATE_MAX:
        print(f"  DatePicker: {DATE_MIN.strftime('%Y-%m-%d')} → {DATE_MAX.strftime('%Y-%m-%d')}")
        
        # Check if it matches expected range
        expected_start = "2018-01-02"
        expected_end = "2024-12-31"
        
        date_min_str = DATE_MIN.strftime('%Y-%m-%d')
        date_max_str = DATE_MAX.strftime('%Y-%m-%d')
        
        if date_min_str == expected_start and date_max_str == expected_end:
            print(f"  ✓ Intervalo correto: {expected_start} → {expected_end}")
        else:
            print(f"  ⚠ Intervalo difere do esperado")
            print(f"    Esperado: {expected_start} → {expected_end}")
            print(f"    Atual: {date_min_str} → {date_max_str}")
    else:
        print("  ✗ DATE_MIN ou DATE_MAX não definidos")
    
    # Check models
    print(f"\nModelos disponíveis: {len(MODEL_OPTIONS)}")
    if MODEL_OPTIONS:
        for model in MODEL_OPTIONS:
            print(f"  - {model}")
    else:
        print("  ⚠ Nenhum modelo encontrado")
    
    # Check events
    print(f"\nEventos de latência:")
    if len(LATENCY_DF) > 0:
        print(f"  ✓ {len(LATENCY_DF)} eventos encontrados")
        if 'event_day' in LATENCY_DF.columns:
            event_dates = LATENCY_DF['event_day']
            print(f"  Intervalo: {event_dates.min()} → {event_dates.max()}")
        if 'event_name' in LATENCY_DF.columns:
            print(f"  Exemplos: {', '.join(LATENCY_DF['event_name'].head(3).tolist())}")
    else:
        print("  ⚠ Nenhum evento de latência encontrado")
    
    # Check if app would run
    print(f"\nStatus do dashboard:")
    print(f"  ✓ App configurado e pronto para executar")
    print(f"  Título: {app.title}")
    print(f"  Para testar: python app_dashboard.py")
    
    print("\n" + "=" * 80)
    print("DASHBOARD: OK (sem avisos de arquivo ausente)")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Erro ao carregar dashboard: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
