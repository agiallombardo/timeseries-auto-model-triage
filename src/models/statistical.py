import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

def run_arima(train_data, test_data, order=(5,1,0)):
    """Run ARIMA model."""
    logger.info(f"Training ARIMA{order} model...")
    
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))
    
    return predictions

def run_sarima(train_data, test_data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Run SARIMA model."""
    logger.info(f"Training SARIMA{order}x{seasonal_order} model...")
    
    model = SARIMAX(train_data, 
                   order=order, 
                   seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))
    
    return predictions

def run_moving_average(train_data, test_data, window=3):
    """Run Simple Moving Average model."""
    logger.info(f"Calculating Moving Average with window={window}...")
    
    # Calculate the moving average of the training data
    ma = train_data.rolling(window=window).mean()
    
    # Use the last value as the prediction for all test points
    last_ma = ma.iloc[-1]
    predictions = np.full(len(test_data), last_ma)
    
    return predictions

def run_exponential_smoothing(train_data, test_data, trend='add', seasonal='add', seasonal_periods=12):
    """Run Exponential Smoothing model."""
    logger.info(f"Training Exponential Smoothing model with {trend} trend and {seasonal} seasonality...")
    
    model = ExponentialSmoothing(train_data, 
                               trend=trend, 
                               seasonal=seasonal, 
                               seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))
    
    return predictions