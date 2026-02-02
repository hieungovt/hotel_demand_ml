"""
Data Preprocessing Module
CRISP-DM Phase 3: Data Preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the hotel bookings dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values and outliers."""
    df = df.copy()
    
    # Handle missing values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Remove rows with 0 guests
    df = df[(df['adults'] + df['children'] + df['babies']) > 0]
    
    # Remove extreme outliers in ADR
    df = df[df['adr'] >= 0]
    df = df[df['adr'] < df['adr'].quantile(0.99)]
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features for modeling."""
    df = df.copy()
    
    # Total nights
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # Total guests
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    
    # Is weekend stay
    df['is_weekend_stay'] = (df['stays_in_weekend_nights'] > 0).astype(int)
    
    # Lead time buckets
    df['lead_time_bucket'] = pd.cut(
        df['lead_time'],
        bins=[0, 7, 30, 90, 180, 365, float('inf')],
        labels=['0-7', '8-30', '31-90', '91-180', '181-365', '365+']
    )
    
    # Season based on arrival month
    month_to_season = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_month_num'] = df['arrival_date_month'].map(month_mapping)
    df['season'] = df['arrival_month_num'].map(month_to_season)
    
    # Create date column for time series
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_month_num'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str),
        errors='coerce'
    )
    
    return df


def prepare_classification_data(df: pd.DataFrame) -> tuple:
    """Prepare data for classification model."""
    # Select features for classification
    feature_cols = [
        'lead_time', 'arrival_date_week_number', 'arrival_month_num',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
        'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
        'total_of_special_requests', 'total_nights', 'total_guests',
        'is_weekend_stay'
    ]
    
    # Categorical features to encode
    cat_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel',
                'deposit_type', 'customer_type', 'season']
    
    # Prepare feature matrix
    X = df[feature_cols].copy()
    
    # Encode categorical features
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
    
    # Target
    y = df['is_canceled']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols + cat_cols


def prepare_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data for time series forecasting."""
    df = df.copy()
    
    # Remove cancelled bookings for demand counting
    df_not_canceled = df[df['is_canceled'] == 0]
    
    # Aggregate daily bookings
    daily_bookings = df_not_canceled.groupby('arrival_date').agg({
        'hotel': 'count',
        'adr': 'mean'
    }).reset_index()
    
    daily_bookings.columns = ['ds', 'y', 'avg_adr']
    daily_bookings = daily_bookings.dropna()
    daily_bookings = daily_bookings.sort_values('ds')
    
    return daily_bookings


if __name__ == "__main__":
    # Test preprocessing
    df = load_data("../data/raw/hotel_bookings.csv")
    print(f"Loaded {len(df)} records")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df)} records")
    
    df = engineer_features(df)
    print(f"Features engineered")
    
    X_train, X_test, y_train, y_test, features = prepare_classification_data(df)
    print(f"Classification data: {X_train.shape}, {X_test.shape}")
    
    ts_data = prepare_time_series_data(df)
    print(f"Time series data: {len(ts_data)} days")
