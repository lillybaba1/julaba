# indicator.py
# Placeholder indicator module for Benscript trading bot
# Replace this with your actual trading signal logic

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on OHLCV data.
    
    Args:
        df: DataFrame with columns: Epoch, Open, High, Low, Close, Volume
        
    Returns:
        DataFrame with a 'Side' column:
            1 = Long signal
           -1 = Short signal
            0 = No signal
    """
    df = df.copy()
    
    # Normalize column names to handle both lowercase and capitalized
    col_map = {col: col.capitalize() for col in df.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']}
    df = df.rename(columns=col_map)
    
    # Simple example: SMA crossover strategy
    # Replace this with your actual indicator logic
    
    short_window = 10
    long_window = 30
    
    df["SMA_short"] = df["Close"].rolling(window=short_window, min_periods=1).mean()
    df["SMA_long"] = df["Close"].rolling(window=long_window, min_periods=1).mean()
    
    # Generate signals
    df["Side"] = 0
    
    # Long when short SMA crosses above long SMA
    df.loc[(df["SMA_short"] > df["SMA_long"]) & 
           (df["SMA_short"].shift(1) <= df["SMA_long"].shift(1)), "Side"] = 1
    
    # Short when short SMA crosses below long SMA
    df.loc[(df["SMA_short"] < df["SMA_long"]) & 
           (df["SMA_short"].shift(1) >= df["SMA_long"].shift(1)), "Side"] = -1
    
    return df
