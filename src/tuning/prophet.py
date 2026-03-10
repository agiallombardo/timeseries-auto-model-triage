import itertools
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from prophet import Prophet

from ..model_config import TUNING_SETUP
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_prophet(train_data, test_data, results_dir=None, **kwargs):
    """
    Grid search for Prophet hyperparameters using train-only validation.
    Refits best model on full training data before producing test predictions.
    Returns (best_params, best_predictions).
    """
    logger.info("Performing grid search for Prophet model...")
    train_df = pd.DataFrame({
        "ds": train_data.index,
        "y": train_data.values,
    })
    n = len(train_df)
    val_frac = TUNING_SETUP.get("val_frac", 0.2)
    train_idx, val_idx = get_chronological_holdout_indices(n, val_frac)
    train_val = train_df.iloc[train_idx]
    val_df = train_df.iloc[val_idx][["ds"]]
    val = train_df.iloc[val_idx]["y"].values

    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    rmses = []
    
    for params in tqdm(all_params, desc="Prophet grid", unit="candidate", leave=False):
        model = Prophet(**params)
        model.fit(train_val)
        forecast = model.predict(val_df)
        val_pred = forecast["yhat"].values
        rmse = score_validation_rmse(val, val_pred)
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
    path = os.path.join(results_dir, 'prophet_grid_search_results.csv') if results_dir else 'prophet_grid_search_results.csv'
    results_df.to_csv(path, index=False)
    return best_params, best_predictions