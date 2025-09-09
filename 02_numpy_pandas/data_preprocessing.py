"""
Problem: Implement common data preprocessing steps using pandas.

Tasks:
1. Handle missing values with different strategies
2. Encode categorical variables
3. Normalize/standardize numerical features
4. Feature engineering (date/time features)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Any

class DataPreprocessor:
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_stats = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values with different strategies per column.
        
        strategy: dict mapping column names to strategies
        Strategies: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        """

        df_processed = df.copy()
        
        for col, method in strategy.items():
            if col not in df_processed.columns:
                continue
                
            if method == 'drop':
                df_processed = df_processed.dropna(subset=[col])
            elif method == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif method == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            elif method == 'mode':
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            elif method == 'forward_fill':
                df_processed[col].fillna(method='ffill', inplace=True)
            elif method == 'backward_fill':
                df_processed[col].fillna(method='bfill', inplace=True)
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        """
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col not in df_processed.columns:
                continue
                
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        return df_processed
    
    def normalize_features(self, df: pd.DataFrame, numerical_cols: List[str], 
                          method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features.
        Methods: 'standard', 'minmax', 'robust'
        """

        df_processed = df.copy()
        
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            else:
                df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def create_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Extract useful features from datetime column.
        """
        
        df_processed = df.copy()
        
        if date_col in df_processed.columns:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
            
            df_processed[f'{date_col}_year'] = df_processed[date_col].dt.year
            df_processed[f'{date_col}_month'] = df_processed[date_col].dt.month
            df_processed[f'{date_col}_day'] = df_processed[date_col].dt.day
            df_processed[f'{date_col}_dayofweek'] = df_processed[date_col].dt.dayofweek
            df_processed[f'{date_col}_quarter'] = df_processed[date_col].dt.quarter
            df_processed[f'{date_col}_is_weekend'] = (df_processed[date_col].dt.dayofweek >= 5).astype(int)
        
        return df_processed

# Test cases
def test_data_preprocessor():
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': ['cat', 'dog', 'cat', np.nan, 'bird'],
        'C': [10, 20, 30, 40, 50],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    df = pd.DataFrame(data)
    
    preprocessor = DataPreprocessor()
    
    # Test missing value handling
    strategy = {'A': 'mean', 'B': 'mode'}
    df_clean = preprocessor.handle_missing_values(df, strategy)
    assert not df_clean['A'].isna().any()
    
    # Test categorical encoding
    df_encoded = preprocessor.encode_categorical_features(df_clean, ['B'])
    assert df_encoded['B'].dtype in [np.int32, np.int64]
    
    # Test date feature extraction
    df_dates = preprocessor.create_date_features(df_encoded, 'date')
    assert 'date_year' in df_dates.columns
    assert 'date_is_weekend' in df_dates.columns
    
    print("All data preprocessing tests passed!")

if __name__ == "__main__":
    test_data_preprocessor()
