import itertools
import logging
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..models.arima import run_arima

logger = logging.getLogger(__name__)


def grid_search_arima(train_data, test_data):
    """Grid search for ARIMA order. Returns (best_order, best_predictions)."""
    logger.info("Performing grid search for ARIMA model...")
    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 5)
    pdq = list(itertools.product(p_values, d_values, q_values))

    best_aic = float('inf')
    best_order = None
    results = []

    for param in pdq:
        try:
            model = ARIMA(train_data, order=param)
            model_fit = model.fit()
            aic = model_fit.aic
            results.append({'order': param, 'aic': aic})
            if aic < best_aic:
                best_aic = aic
                best_order = param
            logger.debug(f"ARIMA{param} - AIC: {aic:.4f}")
        except Exception:
            continue

    logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.4f}")
    best_predictions = run_arima(train_data, test_data, order=best_order)

    if results:
        pd.DataFrame(results).sort_values('aic').to_csv(
            'arima_grid_search_results.csv', index=False
        )
    return {"order": list(best_order)}, best_predictions
