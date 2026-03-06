import itertools
import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..models.sarima import run_sarima

logger = logging.getLogger(__name__)


def grid_search_sarima(train_data, test_data, seasonal_periods=12):
    """Grid search for SARIMA. Returns ((order, seasonal_order), best_predictions)."""
    logger.info("Performing grid search for SARIMA model...")
    p_values, d_values, q_values = range(0, 3), range(0, 2), range(0, 3)
    P_values, D_values, Q_values = range(0, 2), range(0, 2), range(0, 2)
    pdq = list(itertools.product(p_values, d_values, q_values))
    seasonal_pdq = list(itertools.product(P_values, D_values, Q_values, [seasonal_periods]))

    best_aic = float('inf')
    best_order, best_seasonal_order = None, None
    results = []
    max_combinations = 20
    combinations_tested = 0

    for param in pdq:
        for seasonal_param in seasonal_pdq:
            if combinations_tested >= max_combinations:
                break
            try:
                model = SARIMAX(train_data, order=param, seasonal_order=seasonal_param)
                model_fit = model.fit(disp=False)
                aic = model_fit.aic
                results.append({'order': param, 'seasonal_order': seasonal_param, 'aic': aic})
                if aic < best_aic:
                    best_aic = aic
                    best_order = param
                    best_seasonal_order = seasonal_param
                logger.debug(f"SARIMA{param}x{seasonal_param} - AIC: {aic:.4f}")
                combinations_tested += 1
            except Exception:
                continue
        if combinations_tested >= max_combinations:
            break

    logger.info(f"Best SARIMA order: {best_order}x{best_seasonal_order} with AIC: {best_aic:.4f}")
    best_predictions = run_sarima(
        train_data, test_data,
        order=best_order,
        seasonal_order=best_seasonal_order,
    )

    if results:
        pd.DataFrame(results).sort_values('aic').to_csv(
            'sarima_grid_search_results.csv', index=False
        )
    best_params = {
        "order": list(best_order),
        "seasonal_order": list(best_seasonal_order),
    }
    return best_params, best_predictions
