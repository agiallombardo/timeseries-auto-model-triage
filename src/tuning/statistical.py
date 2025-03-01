import numpy as np
import pandas as pd
import logging
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

def grid_search_arima(train_data, test_data):
    """
    Grid search for ARIMA model hyperparameters.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
        
    Returns:
    --------
    tuple
        Best parameters, best predictions
    """
    logger.info("Performing grid search for ARIMA model...")
    
    # Define parameter grid
    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 5)
    
    # Create all possible combinations
    pdq = list(itertools.product(p_values, d_values, q_values))
    
    # Track the best model
    best_aic = float('inf')
    best_order = None
    results = []
    
    # Loop through all combinations
    for param in pdq:
        try:
            model = ARIMA(train_data, order=param)
            model_fit = model.fit()
            aic = model_fit.aic
            
            # Store results
            results.append({
                'order': param,
                'aic': aic
            })
            
            if aic < best_aic:
                best_aic = aic
                best_order = param
                
            logger.debug(f"ARIMA{param} - AIC: {aic:.4f}")
        except:
            continue
    
    logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.4f}")
    
    # Fit the best model
    best_model = ARIMA(train_data, order=best_order)
    best_model_fit = best_model.fit()
    
    # Make predictions
    best_predictions = best_model_fit.forecast(steps=len(test_data))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')
    results_df.to_csv('arima_grid_search_results.csv', index=False)
    
    return best_order, best_predictions

def grid_search_sarima(train_data, test_data, seasonal_periods=12):
    """
    Grid search for SARIMA model hyperparameters.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
    seasonal_periods : int
        Seasonal periods for the model
        
    Returns:
    --------
    tuple
        Best parameters, best predictions
    """
    logger.info("Performing grid search for SARIMA model...")
    
    # Define parameter grid (smaller grid due to computational complexity)
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    
    # Create all possible combinations
    pdq = list(itertools.product(p_values, d_values, q_values))
    seasonal_pdq = list(itertools.product(P_values, D_values, Q_values, [seasonal_periods]))
    
    # Track the best model
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    results = []
    
    # Loop through combinations
    max_combinations = 20  # Limit to avoid excessive computation
    combinations_tested = 0
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(train_data, 
                               order=param, 
                               seasonal_order=seasonal_param)
                model_fit = model.fit(disp=False)
                aic = model_fit.aic
                
                # Store results
                results.append({
                    'order': param,
                    'seasonal_order': seasonal_param,
                    'aic': aic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = param
                    best_seasonal_order = seasonal_param
                    
                logger.debug(f"SARIMA{param}x{seasonal_param} - AIC: {aic:.4f}")
                
                combinations_tested += 1
                if combinations_tested >= max_combinations:
                    break
            except:
                continue
        
        if combinations_tested >= max_combinations:
            break
    
    logger.info(f"Best SARIMA order: {best_order}x{best_seasonal_order} with AIC: {best_aic:.4f}")
    
    # Fit the best model
    best_model = SARIMAX(train_data, 
                        order=best_order, 
                        seasonal_order=best_seasonal_order)
    best_model_fit = best_model.fit(disp=False)
    
    # Make predictions
    best_predictions = best_model_fit.forecast(steps=len(test_data))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')
    results_df.to_csv('sarima_grid_search_results.csv', index=False)
    
    return (best_order, best_seasonal_order), best_predictions

def grid_search_moving_average(train_data, test_data):
    """
    Grid search for Moving Average window size.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
        
    Returns:
    --------
    tuple
        Best window, best predictions
    """
    logger.info("Performing grid search for Moving Average window size...")
    
    # Define window sizes to try
    window_sizes = range(2, min(30, len(train_data) // 2))
    
    # Validation set from training data
    val_size = min(len(test_data), len(train_data) // 4)
    train_val = train_data[:-val_size]
    val = train_data[-val_size:]
    
    best_rmse = float('inf')
    best_window = None
    results = []
    
    for window in window_sizes:
        ma = train_val.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        val_pred = np.full(len(val), last_ma)
        
        rmse = np.sqrt(mean_squared_error(val, val_pred))
        
        results.append({
            'window': window,
            'rmse': rmse
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window
            
        logger.debug(f"MA(window={window}) - RMSE: {rmse:.4f}")
    
    logger.info(f"Best Moving Average window: {best_window} with RMSE: {best_rmse:.4f}")
    
    # Calculate predictions with best window
    ma = train_data.rolling(window=best_window).mean()
    last_ma = ma.iloc[-1]
    best_predictions = np.full(len(test_data), last_ma)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df.to_csv('ma_grid_search_results.csv', index=False)
    
    return best_window, best_predictions

def grid_search_exponential_smoothing(train_data, test_data, seasonal_periods=12):
    """
    Grid search for Exponential Smoothing parameters.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
    seasonal_periods : int
        Seasonal periods for the model
        
    Returns:
    --------
    tuple
        Best parameters, best predictions
    """
    logger.info("Performing grid search for Exponential Smoothing model...")
    
    # Define parameter grid
    trend_types = ['add', 'mul', None]
    seasonal_types = ['add', 'mul', None]
    
    best_aic = float('inf')
    best_params = None
    results = []
    
    for trend in trend_types:
        for seasonal in seasonal_types:
            # Skip invalid combinations
            if seasonal is not None and len(train_data) < 2 * seasonal_periods:
                continue
                
            try:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None
                )
                model_fit = model.fit()
                aic = model_fit.aic
                
                results.append({
                    'trend': trend,
                    'seasonal': seasonal,
                    'aic': aic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (trend, seasonal)
                    
                logger.debug(f"ETS(trend={trend}, seasonal={seasonal}) - AIC: {aic:.4f}")
            except:
                continue
    
    logger.info(f"Best Exponential Smoothing parameters: trend={best_params[0]}, seasonal={best_params[1]} with AIC: {best_aic:.4f}")
    
    # Fit the best model
    best_model = ExponentialSmoothing(
        train_data,
        trend=best_params[0],
        seasonal=best_params[1],
        seasonal_periods=seasonal_periods if best_params[1] else None
    )
    best_model_fit = best_model.fit()
    
    # Make predictions
    best_predictions = best_model_fit.forecast(steps=len(test_data))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')
    results_df.to_csv('es_grid_search_results.csv', index=False)
    
    return best_params, best_predictions