"""
Preprocessing Module
====================
Handles data cleaning and target variable creation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def clean_flights_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the flight data by removing invalid records.
    
    Steps:
    1. Remove cancelled flights
    2. Remove diverted flights
    3. Remove rows with missing arrival delay
    4. Handle missing values in other columns
    
    Args:
        df: Raw flight data
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Starting cleaning with {len(df):,} rows")
    initial_count = len(df)
    
    # 1. Remove cancelled flights
    if 'CANCELLED' in df.columns:
        df = df[df['CANCELLED'] == 0].copy()
        print(f"  After removing cancelled: {len(df):,} rows")
    
    # 2. Remove diverted flights
    if 'DIVERTED' in df.columns:
        df = df[df['DIVERTED'] == 0].copy()
        print(f"  After removing diverted: {len(df):,} rows")
    
    # 3. Remove rows with missing target (ARRIVAL_DELAY)
    df = df.dropna(subset=['ARRIVAL_DELAY'])
    print(f"  After removing null delays: {len(df):,} rows")
    
    # 4. Handle missing values in key columns
    key_columns = [
        'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'SCHEDULED_TIME',
        'DISTANCE', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'
    ]
    
    for col in key_columns:
        if col in df.columns:
            if df[col].dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
                # Fill numeric with median
                df[col] = df[col].fillna(df[col].median())
            else:
                # Fill categorical with mode
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'UNKNOWN')
    
    final_count = len(df)
    print(f"Cleaning complete: {initial_count:,} -> {final_count:,} rows ({final_count/initial_count*100:.1f}% retained)")
    
    return df.reset_index(drop=True)


def create_target(df: pd.DataFrame, delay_threshold: int = 15) -> pd.DataFrame:
    """
    Create binary target variable for classification.
    
    A flight is considered "delayed" if ARRIVAL_DELAY > threshold minutes.
    
    Args:
        df: DataFrame with ARRIVAL_DELAY column
        delay_threshold: Minutes threshold for delay (default: 15)
    
    Returns:
        DataFrame with new 'delayed' column
    """
    df = df.copy()
    df['delayed'] = (df['ARRIVAL_DELAY'] > delay_threshold).astype(int)
    
    delay_rate = df['delayed'].mean() * 100
    print(f"Target created: {delay_rate:.1f}% flights delayed (>{delay_threshold} min)")
    print(f"  Class distribution: 0={len(df[df['delayed']==0]):,}, 1={len(df[df['delayed']==1]):,}")
    
    return df


def preprocess_pipeline(df: pd.DataFrame, delay_threshold: int = 15) -> pd.DataFrame:
    """
    Run full preprocessing pipeline.
    
    Args:
        df: Raw flight data
        delay_threshold: Minutes threshold for delay classification
    
    Returns:
        Cleaned DataFrame with target variable
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Clean data
    df = clean_flights_data(df)
    
    # Step 2: Create target
    df = create_target(df, delay_threshold)
    
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    
    return df


if __name__ == "__main__":
    # Quick test with sample data
    from data_loader import load_raw_flights
    
    df = load_raw_flights(nrows=10000)
    df = preprocess_pipeline(df)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
