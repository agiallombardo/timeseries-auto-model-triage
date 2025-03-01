import numpy as np
import pandas as pd
import logging
import itertools
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

def grid_search_prophet(train_data, test_data):
    """
    Grid search for Prophet hyperparameters.
    
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
    logger.info("Performing grid search for Prophet model...")
    
    # Prepare data in Prophet format
    train_df = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    # Generate all combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Validation set
    val_size = min(len(test_data), len(train_data) // 4)
    train_val = train_df.iloc[:-val_size]
    val_df = pd.DataFrame({'ds': train_data.index[-val_size:]})
    val = train_data[-val_size:]
    
    rmses = []
    
    for params in all_params:
        model = Prophet(**params)
        model.fit(train_val)
        forecast = model.predict(val_df)
        val_pred = forecast['yhat'].values
        
        rmse = np.sqrt(mean_squared_error(val, val_pred))
        rmses.append(rmse)
        
        logger.debug(f"Prophet({params}) - RMSE: {rmse:.4f}")
    
    # Find the best parameters
    best_idx = np.argmin(rmses)
    best_params = all_params[best_idx]
    best_rmse = rmses[best_idx]
    
    logger.info(f"Best Prophet parameters: {best_params} with RMSE: {best_rmse:.4f}")
    
    # Fit the best model on the full training data
    best_model = Prophet(**best_params)
    best_model.fit(train_df)
    
    # Make predictions
    future = pd.DataFrame({'ds': test_data.index})
    forecast = best_model.predict(future)
    best_predictions = forecast['yhat'].values
    
    # Save results
    results = []
    for params, rmse in zip(all_params, rmses):
        result = params.copy()
        result['rmse'] = rmse
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df.to_csv('prophet_grid_search_results.csv', index=False)
    
    return best_params, best_predictions