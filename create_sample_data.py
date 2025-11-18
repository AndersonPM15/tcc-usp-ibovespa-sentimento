#!/usr/bin/env python
"""
Create sample data files for testing the dashboard and verification.
This script generates minimal sample data covering the expected date range.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

from src.io import paths as path_utils
from src.config import loader as cfg

def create_sample_ibovespa():
    """Create sample Ibovespa data."""
    periodo = cfg.get_periodo_estudo()
    start_date = pd.to_datetime(periodo["start"])
    end_date = pd.to_datetime(periodo["end"])
    
    # Create daily date range (business days)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    # Generate random walk prices starting at 100,000
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'day': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
        'close': prices,
        'adj_close': prices,
        'volume': np.random.randint(1e9, 5e9, len(dates))
    })
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "ibovespa_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    print(f"Date range: {df['day'].min()} to {df['day'].max()}")
    return df

def create_sample_oof_predictions(ibov_df):
    """Create sample out-of-fold predictions."""
    # Sample some dates from ibovespa
    sample_dates = ibov_df.sample(min(500, len(ibov_df)), random_state=42)['day']
    
    np.random.seed(42)
    df = pd.DataFrame({
        'day': sample_dates,
        'proba': np.random.beta(2, 2, len(sample_dates)),  # Probabilities between 0 and 1
        'model': np.random.choice(['logreg_l2', 'rf_100', 'xgb_default'], len(sample_dates)),
        'fold': np.random.randint(0, 5, len(sample_dates))
    })
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "16_oof_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

def create_sample_results_json():
    """Create sample results JSON for models."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "logreg_l2": {
                "auc": {"value": 0.65, "std": 0.03},
                "mda": {"value": 0.58, "std": 0.02},
            },
            "rf_100": {
                "auc": {"value": 0.62, "std": 0.04},
                "mda": {"value": 0.56, "std": 0.03},
            },
            "xgb_default": {
                "auc": {"value": 0.68, "std": 0.02},
                "mda": {"value": 0.60, "std": 0.02},
            }
        }
    }
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "results_16_models_tfidf.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Created {output_path}")
    return results

def create_sample_backtest_results():
    """Create sample backtest results."""
    models = ['logreg_l2', 'rf_100', 'xgb_default']
    strategies = ['long_short', 'long_only', 'threshold_05']
    
    rows = []
    np.random.seed(42)
    for model in models:
        for strategy in strategies:
            rows.append({
                'model': model,
                'strategy': strategy,
                'cagr': np.random.uniform(0.05, 0.25),
                'sharpe': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(-0.3, -0.1),
                'win_rate': np.random.uniform(0.45, 0.65)
            })
    
    df = pd.DataFrame(rows)
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "18_backtest_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

def create_sample_latency_events(ibov_df):
    """Create sample latency events."""
    # Pick some random dates
    sample_dates = ibov_df.sample(20, random_state=42)['day']
    
    event_names = [
        "Copom sobe Selic",
        "Petrobras anuncia dividendos",
        "Eleições presidenciais",
        "Mudança fiscal",
        "Crise energética",
        "PIB acima do esperado",
        "Desemprego cai",
        "Inflação sobe",
        "BC intervém no câmbio",
        "Reforma tributária aprovada"
    ]
    
    np.random.seed(42)
    df = pd.DataFrame({
        'event_day': sample_dates,
        'event_name': np.random.choice(event_names, len(sample_dates)),
        'T_half_days': np.random.uniform(0.5, 5.0, len(sample_dates)),
        'impact_bps': np.random.uniform(-200, 200, len(sample_dates))
    })
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "event_study_latency.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

def create_sample_labels():
    """Create sample labels for daily predictions."""
    periodo = cfg.get_periodo_estudo()
    start_date = pd.to_datetime(periodo["start"])
    end_date = pd.to_datetime(periodo["end"])
    
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    
    np.random.seed(42)
    df = pd.DataFrame({
        'day': dates,
        'label': np.random.choice([0, 1], len(dates), p=[0.45, 0.55]),
        'return_pct': np.random.normal(0.001, 0.02, len(dates))
    })
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "labels_y_daily.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

def create_sample_tfidf_index(ibov_df):
    """Create sample TF-IDF index."""
    # Use business days from ibov
    df = pd.DataFrame({
        'day': ibov_df['day'],
        'n_docs': np.random.randint(5, 50, len(ibov_df))
    })
    
    data_paths = path_utils.get_data_paths()
    output_path = data_paths["data_processed"] / "tfidf_daily_index.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows")
    return df

def main():
    """Create all sample data files."""
    print("Creating sample data files...")
    print("=" * 80)
    
    # Create data directories
    data_paths = path_utils.get_data_paths(create=True)
    
    # Create sample data
    ibov_df = create_sample_ibovespa()
    oof_df = create_sample_oof_predictions(ibov_df)
    results = create_sample_results_json()
    backtest_df = create_sample_backtest_results()
    latency_df = create_sample_latency_events(ibov_df)
    labels_df = create_sample_labels()
    tfidf_idx_df = create_sample_tfidf_index(ibov_df)
    
    print("=" * 80)
    print("Sample data creation complete!")
    print(f"All files created in: {data_paths['data_processed']}")

if __name__ == "__main__":
    main()
