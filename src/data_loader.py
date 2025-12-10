"""
Data Loader Module
==================
Handles loading, sampling, and saving of flight delay data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_flights(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw flights.csv with optimized dtypes.
    
    Args:
        nrows: Number of rows to load. None for all rows.
    
    Returns:
        DataFrame with flight data
    """
    # Define optimized dtypes to reduce memory usage
    dtype_dict = {
        'YEAR': 'int16',
        'MONTH': 'int8',
        'DAY': 'int8',
        'DAY_OF_WEEK': 'int8',
        'AIRLINE': 'category',
        'FLIGHT_NUMBER': 'int16',
        'TAIL_NUMBER': 'category',
        'ORIGIN_AIRPORT': 'category',
        'DESTINATION_AIRPORT': 'category',
        'SCHEDULED_DEPARTURE': 'int16',
        'DEPARTURE_TIME': 'float32',
        'DEPARTURE_DELAY': 'float32',
        'TAXI_OUT': 'float32',
        'WHEELS_OFF': 'float32',
        'SCHEDULED_TIME': 'float32',
        'ELAPSED_TIME': 'float32',
        'AIR_TIME': 'float32',
        'DISTANCE': 'int16',
        'WHEELS_ON': 'float32',
        'TAXI_IN': 'float32',
        'SCHEDULED_ARRIVAL': 'int16',
        'ARRIVAL_TIME': 'float32',
        'ARRIVAL_DELAY': 'float32',
        'DIVERTED': 'int8',
        'CANCELLED': 'int8',
    }
    
    flights_path = DATA_RAW / "flights.csv"
    
    print(f"Loading flights data from {flights_path}...")
    df = pd.read_csv(
        flights_path,
        dtype=dtype_dict,
        nrows=nrows,
        low_memory=False
    )
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df


def load_airlines() -> pd.DataFrame:
    """Load airlines reference data."""
    return pd.read_csv(DATA_RAW / "airlines.csv")


def load_airports() -> pd.DataFrame:
    """Load airports reference data."""
    return pd.read_csv(DATA_RAW / "airports.csv")


def sample_data(
    df: pd.DataFrame,
    n_samples: int = 50000,
    stratify_col: str = 'delayed',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform stratified sampling to reduce dataset size.
    
    Args:
        df: Input DataFrame (must have target column)
        n_samples: Number of samples to keep
        stratify_col: Column to stratify by
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    if len(df) <= n_samples:
        print(f"Dataset already smaller than {n_samples:,}, returning as-is")
        return df
    
    # Stratified sampling
    from sklearn.model_selection import train_test_split
    
    sampled, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    print(f"Sampled {len(sampled):,} rows from {len(df):,} (stratified by '{stratify_col}')")
    return sampled.reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str = "flights_processed.csv") -> Path:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / filename
    
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return output_path


def load_processed(filename: str = "flights_processed.csv") -> pd.DataFrame:
    """Load previously processed data."""
    return pd.read_csv(DATA_PROCESSED / filename)


if __name__ == "__main__":
    # Quick test
    df = load_raw_flights(nrows=1000)
    print(f"\nSample columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
