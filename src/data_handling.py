import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(file_path, time_col, data_col, date_format=None):
    """
    Load time series data from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    time_col : str
        Name of the column containing time/date information
    data_col : str
        Name of the column containing the target data to forecast
    date_format : str, optional
        Format of the date string if needed for parsing
        
    Returns:
    --------
    pd.DataFrame
        Loaded and processed data
    """
    # Determine file type from extension
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == 'json':
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Ensure the required columns exist
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in the data")
    if data_col not in df.columns:
        raise ValueError(f"Data column '{data_col}' not found in the data")
    
    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        if date_format:
            df[time_col] = pd.to_datetime(df[time_col], format=date_format)
        else:
            df[time_col] = pd.to_datetime(df[time_col])
    
    # Set time column as index
    df = df.set_index(time_col)
    
    # Sort index to ensure chronological order
    df = df.sort_index()
    
    # Extract only the data column we want to forecast
    if len(df.columns) > 1:
        df = df[[data_col]]
    
    return df

def prepare_features(df, target_col, lags=5, rolling_window=3):
    """
    Prepare features for machine learning models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with the time series data
    target_col : str
        Name of the target column
    lags : int, optional
        Number of lag features to create
    rolling_window : int, optional
        Size of the rolling window for statistics
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and target
    """
    df_features = df.copy()
    
    # Create lag features
    for lag in range(1, lags + 1):
        df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Create rolling window statistics
    df_features['rolling_mean'] = df_features[target_col].rolling(window=rolling_window).mean().shift(1)
    df_features['rolling_std'] = df_features[target_col].rolling(window=rolling_window).std().shift(1)
    
    # Add time-based features
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    
    # Drop rows with NaN values due to lag and rolling calculations
    df_features = df_features.dropna()
    
    return df_features

def split_data(df, target_col, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    target_col : str
        Name of the target column
    test_size : float, optional
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    # For time series, we typically want to split chronologically
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # For statistical models (ARIMA, SARIMA, etc.)
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # For ML models
    features = [col for col in df.columns if col != target_col]
    X_train = train_df[features]
    X_test = test_df[features]
    
    return X_train, X_test, y_train, y_test, train_df, test_df

def prepare_sequence_data(data, n_steps=3):
    """
    Prepare sequences for RNN/LSTM models.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    n_steps : int
        Number of time steps for input sequence
    
    Returns:
    --------
    tuple
        X (input sequences), y (target values)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)
