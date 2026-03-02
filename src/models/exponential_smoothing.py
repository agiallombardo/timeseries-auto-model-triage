import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)


def run_exponential_smoothing(
    train_data, test_data,
    trend='add', seasonal='add', seasonal_periods=12,
):
    """Run Exponential Smoothing model."""
    logger.info(f"Training Exponential Smoothing model with {trend} trend and {seasonal} seasonality...")
    model = ExponentialSmoothing(
        train_data,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions
