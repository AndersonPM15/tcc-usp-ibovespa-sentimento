"""
Generate sample Ibovespa data for testing when yfinance is unavailable.

This script creates realistic sample data matching the structure expected
by the notebooks, covering the period defined in config_tcc.yaml.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io import paths
from src.config import loader as cfg


def generate_realistic_prices(start_date, end_date, initial_price=100000, volatility=0.02):
    """
    Generate realistic stock price data using geometric Brownian motion.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    initial_price : float
        Starting price
    volatility : float
        Daily volatility (standard deviation of returns)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: day, open, high, low, close, adj_close, volume
    """
    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Generate returns using geometric Brownian motion
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.0003, volatility, n_days)  # Slight upward drift
    
    # Calculate prices
    prices = initial_price * np.exp(np.cumsum(daily_returns))
    
    # Generate OHLC data
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        # High and low are within a realistic range of close
        daily_volatility = np.random.uniform(0.005, 0.02)
        high = close * (1 + daily_volatility)
        low = close * (1 - daily_volatility)
        # Open is between low and high
        open_price = np.random.uniform(low, high)
        
        # Volume in millions
        volume = np.random.uniform(5e9, 15e9)
        
        data.append({
            'day': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'adj_close': close,  # For simplicity, adj_close = close
            'volume': int(volume)
        })
    
    return pd.DataFrame(data)


def generate_ibovespa_data(start_date, end_date, raw_path):
    """Generate sample Ibovespa (^BVSP) data."""
    print(f"Generating Ibovespa data for {start_date} to {end_date}...")
    
    df = generate_realistic_prices(start_date, end_date, initial_price=75000, volatility=0.018)
    df['source_ticker'] = '^BVSP'
    
    output_file = Path(raw_path) / 'ibovespa.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    print(f"Date range: {df['day'].min()} to {df['day'].max()}")
    
    return df


def generate_bova11_data(start_date, end_date, raw_path):
    """Generate sample BOVA11.SA (ETF) data."""
    print(f"\nGenerating BOVA11 data for {start_date} to {end_date}...")
    
    df = generate_realistic_prices(start_date, end_date, initial_price=90, volatility=0.017)
    df['source_ticker'] = 'BOVA11.SA'
    
    output_file = Path(raw_path) / 'bova11.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    print(f"Date range: {df['day'].min()} to {df['day'].max()}")
    
    return df


def main():
    """Main function to generate sample data."""
    # Get configuration
    periodo = cfg.get_periodo_estudo()
    start_date = periodo['start']
    end_date = periodo['end']
    
    # Get paths
    data_paths = paths.get_data_paths()
    raw_path = data_paths['data_raw']
    
    print("=" * 80)
    print("GENERATING SAMPLE IBOVESPA DATA")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Output directory: {raw_path}")
    print()
    
    # Create output directory
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    ibov_df = generate_ibovespa_data(start_date, end_date, raw_path)
    bova_df = generate_bova11_data(start_date, end_date, raw_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = pd.DataFrame([
        {
            'arquivo': 'ibovespa.csv',
            'ticker': '^BVSP',
            'records': len(ibov_df),
            'inicio': ibov_df['day'].min(),
            'fim': ibov_df['day'].max()
        },
        {
            'arquivo': 'bova11.csv',
            'ticker': 'BOVA11.SA',
            'records': len(bova_df),
            'inicio': bova_df['day'].min(),
            'fim': bova_df['day'].max()
        }
    ])
    print(summary.to_string(index=False))
    print("\nData generation completed successfully!")


if __name__ == '__main__':
    main()
