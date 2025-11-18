#!/usr/bin/env python
"""
Test the dashboard loading and basic functionality.
"""

import sys
from datetime import datetime

# Import config to get expected date range
from src.config import loader as cfg

print("=" * 80)
print("ITEM 4: Testando app_dashboard.py")
print("=" * 80)

# Get expected date range from config
periodo = cfg.get_periodo_estudo()
expected_start = periodo["start"]
expected_end = periodo["end"]
print(f"\nPeríodo esperado (config): {expected_start} → {expected_end}")

validation_errors = []

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
    
    # CRITICAL: Validate minimum data requirements
    if len(IBOV_DF) < 100:
        msg = f"IBOV_DF muito pequeno ({len(IBOV_DF)} linhas) - esperado dados 2018-2025"
        validation_errors.append(msg)
        print(f"  ✗ {msg}")
    
    if len(SENTIMENT_DF) < 100:
        msg = f"SENTIMENT_DF muito pequeno ({len(SENTIMENT_DF)} linhas) - esperado dados 2018-2025"
        validation_errors.append(msg)
        print(f"  ✗ {msg}")
    
    # Check date range
    print(f"\nIntervalo de datas:")
    if DATE_MIN and DATE_MAX:
        print(f"  DatePicker: {DATE_MIN.strftime('%Y-%m-%d')} → {DATE_MAX.strftime('%Y-%m-%d')}")
        
        date_min_str = DATE_MIN.strftime('%Y-%m-%d')
        date_max_str = DATE_MAX.strftime('%Y-%m-%d')
        
        # CRITICAL: Assert date range matches config
        if date_min_str != expected_start:
            msg = f"DATE_MIN ({date_min_str}) não corresponde ao esperado ({expected_start})"
            validation_errors.append(msg)
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ DATE_MIN correto: {expected_start}")
            
        if date_max_str != expected_end:
            msg = f"DATE_MAX ({date_max_str}) não corresponde ao esperado ({expected_end})"
            validation_errors.append(msg)
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ DATE_MAX correto: {expected_end}")
    else:
        msg = "DATE_MIN ou DATE_MAX não definidos"
        validation_errors.append(msg)
        print(f"  ✗ {msg}")
    
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
    
    # Final validation summary
    print("\n" + "=" * 80)
    if validation_errors:
        print("DASHBOARD: FALHOU - Problemas encontrados:")
        for err in validation_errors:
            print(f"  ✗ {err}")
        print("=" * 80)
        sys.exit(1)
    else:
        print("DASHBOARD: ✅ PASSOU - Todos os testes bem-sucedidos")
        print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Erro ao carregar dashboard: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
