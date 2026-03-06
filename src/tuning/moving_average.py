import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..models.moving_average import run_moving_average

logger = logging.getLogger(__name__)


def grid_search_moving_average(train_data, test_data):
    """Grid search for Moving Average window. Returns (best_window, best_predictions)."""
    logger.info("Performing grid search for Moving Average window size...")
    window_sizes = range(2, min(30, len(train_data) // 2))
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
        results.append({'window': window, 'rmse': rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window
        logger.debug(f"MA(window={window}) - RMSE: {rmse:.4f}")

    logger.info(f"Best Moving Average window: {best_window} with RMSE: {best_rmse:.4f}")
    best_predictions = run_moving_average(train_data, test_data, window=best_window)

    if results:
        pd.DataFrame(results).sort_values('rmse').to_csv(
            'ma_grid_search_results.csv', index=False
        )
    return {"window": best_window}, best_predictions
