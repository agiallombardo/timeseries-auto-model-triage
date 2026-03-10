import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..model_config import TUNING_SETUP
from ..models.moving_average import run_moving_average
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_moving_average(train_data, test_data, results_dir=None, **kwargs):
    """Grid search for Moving Average window using train-only validation. Returns (best_params, best_predictions)."""
    logger.info("Performing grid search for Moving Average window size...")
    train_arr = np.asarray(train_data).ravel()
    n = len(train_arr)
    val_frac = TUNING_SETUP.get("val_frac", 0.2)
    train_idx, val_idx = get_chronological_holdout_indices(n, val_frac)
    train_val = train_arr[train_idx]
    val = train_arr[val_idx]
    window_sizes = range(2, min(30, len(train_val) // 2))

    best_rmse = float('inf')
    best_window = None
    results = []

    for window in tqdm(window_sizes, desc="MA grid", unit="candidate", leave=False):
        ma = pd.Series(train_val).rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        val_pred = np.full(len(val), last_ma)
        rmse = score_validation_rmse(val, val_pred)
        results.append({'window': window, 'rmse': rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window
        logger.debug(f"MA(window={window}) - RMSE: {rmse:.4f}")

    logger.info(f"Best Moving Average window: {best_window} with RMSE: {best_rmse:.4f}")
    best_predictions = run_moving_average(train_data, test_data, window=best_window)

    if results:
        path = os.path.join(results_dir, 'ma_grid_search_results.csv') if results_dir else 'ma_grid_search_results.csv'
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)
    return {"window": best_window}, best_predictions
