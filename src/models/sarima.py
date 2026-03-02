import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


def run_sarima(train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Run SARIMA model."""
    logger.info(f"Training SARIMA{order}x{seasonal_order} model...")
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions
