import logging
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..models.exponential_smoothing import run_exponential_smoothing

logger = logging.getLogger(__name__)


def grid_search_exponential_smoothing(train_data, test_data, seasonal_periods=12):
    """Grid search for Exponential Smoothing. Returns (best_params, best_predictions)."""
    logger.info("Performing grid search for Exponential Smoothing model...")
    trend_types = ['add', 'mul', None]
    seasonal_types = ['add', 'mul', None]

    best_aic = float('inf')
    best_params = None
    results = []

    for trend in trend_types:
        for seasonal in seasonal_types:
            if seasonal is not None and len(train_data) < 2 * seasonal_periods:
                continue
            try:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None,
                )
                model_fit = model.fit()
                aic = model_fit.aic
                results.append({'trend': trend, 'seasonal': seasonal, 'aic': aic})
                if aic < best_aic:
                    best_aic = aic
                    best_params = (trend, seasonal)
                logger.debug(f"ETS(trend={trend}, seasonal={seasonal}) - AIC: {aic:.4f}")
            except Exception:
                continue

    logger.info(f"Best Exponential Smoothing parameters: trend={best_params[0]}, seasonal={best_params[1]} with AIC: {best_aic:.4f}")
    best_predictions = run_exponential_smoothing(
        train_data, test_data,
        trend=best_params[0],
        seasonal=best_params[1],
        seasonal_periods=seasonal_periods if best_params[1] else None,
    )

    if results:
        pd.DataFrame(results).sort_values('aic').to_csv(
            'es_grid_search_results.csv', index=False
        )
    return {"trend": best_params[0], "seasonal": best_params[1]}, best_predictions
