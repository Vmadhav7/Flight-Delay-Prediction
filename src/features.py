"""
Feature Engineering Module
==========================
Creates features for flight delay prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from SCHEDULED_DEPARTURE.
    
    SCHEDULED_DEPARTURE is in HHMM format (e.g., 1430 = 2:30 PM)
    
    Features created:
    - hour_of_day: Hour (0-23)
    - time_of_day: Categorical (Morning/Afternoon/Evening/Night)
    - is_weekend: Binary (1 if Saturday/Sunday)
    - is_holiday_season: Binary (November/December)
    """
    df = df.copy()
    
    # Extract hour from HHMM format
    df['hour_of_day'] = (df['SCHEDULED_DEPARTURE'] // 100).astype(int)
    df['hour_of_day'] = df['hour_of_day'].clip(0, 23)  # Ensure valid range
    
    # Time of day buckets
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)
    
    # Weekend indicator (DAY_OF_WEEK: 1=Monday, 7=Sunday)
    df['is_weekend'] = (df['DAY_OF_WEEK'] >= 6).astype(int)
    
    # Holiday season (November = 11, December = 12)
    if 'MONTH' in df.columns:
        df['is_holiday_season'] = df['MONTH'].isin([11, 12]).astype(int)
    
    print(f"Time features created: hour_of_day, time_of_day, is_weekend, is_holiday_season")
    
    return df


def create_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create airport congestion features.
    
    Features:
    - origin_flights_per_hour: Number of flights from origin airport per hour
    - dest_flights_per_hour: Number of flights to destination airport per hour
    """
    df = df.copy()
    
    # Origin airport congestion (flights per hour)
    origin_congestion = df.groupby(['ORIGIN_AIRPORT', 'hour_of_day']).size()
    origin_congestion = origin_congestion.reset_index(name='origin_flights_per_hour')
    df = df.merge(origin_congestion, on=['ORIGIN_AIRPORT', 'hour_of_day'], how='left')
    
    # Destination airport congestion
    dest_congestion = df.groupby(['DESTINATION_AIRPORT', 'hour_of_day']).size()
    dest_congestion = dest_congestion.reset_index(name='dest_flights_per_hour')
    df = df.merge(dest_congestion, on=['DESTINATION_AIRPORT', 'hour_of_day'], how='left')
    
    print(f"Congestion features created: origin_flights_per_hour, dest_flights_per_hour")
    
    return df


def create_airline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create airline-based features.
    
    Features:
    - airline_delay_rate: Historical delay rate for each airline
    """
    df = df.copy()
    
    # Calculate airline delay rates
    airline_delay_rate = df.groupby('AIRLINE')['delayed'].mean()
    airline_delay_rate = airline_delay_rate.reset_index(name='airline_delay_rate')
    df = df.merge(airline_delay_rate, on='AIRLINE', how='left')
    
    print(f"Airline features created: airline_delay_rate")
    
    return df


def create_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create route-based features.
    
    Features:
    - route: Combined origin-destination
    - route_popularity: Number of flights on this route
    """
    df = df.copy()
    
    # Create route identifier
    df['route'] = df['ORIGIN_AIRPORT'].astype(str) + '_' + df['DESTINATION_AIRPORT'].astype(str)
    
    # Route popularity
    route_popularity = df['route'].value_counts().reset_index()
    route_popularity.columns = ['route', 'route_popularity']
    df = df.merge(route_popularity, on='route', how='left')
    
    print(f"Route features created: route, route_popularity")
    
    return df


def prepare_features_for_modeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare final feature set for modeling.
    
    - Encodes categorical variables
    - Selects relevant features
    - Returns feature matrix and feature names
    
    Args:
        df: DataFrame with all engineered features
    
    Returns:
        Tuple of (feature DataFrame, feature names list)
    """
    df = df.copy()
    
    # Define features to use
    numeric_features = [
        'hour_of_day',
        'DAY_OF_WEEK',
        'MONTH',
        'is_weekend',
        'is_holiday_season',
        'SCHEDULED_TIME',
        'DISTANCE',
        'origin_flights_per_hour',
        'dest_flights_per_hour',
        'airline_delay_rate',
        'route_popularity'
    ]
    
    categorical_features = ['AIRLINE', 'time_of_day']
    
    # Filter to existing columns
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Get all feature columns (numeric + encoded categorical)
    feature_cols = numeric_features.copy()
    for cat in categorical_features:
        encoded_cols = [c for c in df_encoded.columns if c.startswith(cat + '_')]
        feature_cols.extend(encoded_cols)
    
    # Filter to valid features only
    feature_cols = [f for f in feature_cols if f in df_encoded.columns]
    
    print(f"\nFinal feature set: {len(feature_cols)} features")
    print(f"  Numeric: {len(numeric_features)}")
    print(f"  Categorical (encoded): {len(feature_cols) - len(numeric_features)}")
    
    return df_encoded, feature_cols


def feature_engineering_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run full feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame with target variable
    
    Returns:
        Tuple of (engineered DataFrame, feature names)
    """
    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    # Step 1: Time features
    df = extract_time_features(df)
    
    # Step 2: Congestion features
    df = create_congestion_features(df)
    
    # Step 3: Airline features
    df = create_airline_features(df)
    
    # Step 4: Route features
    df = create_route_features(df)
    
    # Step 5: Prepare for modeling
    df_final, feature_cols = prepare_features_for_modeling(df)
    
    print("=" * 50)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 50)
    
    return df_final, feature_cols


if __name__ == "__main__":
    # Quick test
    from data_loader import load_raw_flights
    from preprocessing import preprocess_pipeline
    
    df = load_raw_flights(nrows=10000)
    df = preprocess_pipeline(df)
    df, features = feature_engineering_pipeline(df)
    
    print(f"\nFeature columns: {features}")
