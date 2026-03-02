import logging
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


def run_arima(train_data, test_data, order=(5, 1, 0)):
    """Run ARIMA model."""
    logger.info(f"Training ARIMA{order} model...")
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions
