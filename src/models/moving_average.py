import logging
import numpy as np

logger = logging.getLogger(__name__)


def run_moving_average(train_data, test_data, window=3):
    """Run Simple Moving Average model."""
    logger.info(f"Calculating Moving Average with window={window}...")
    ma = train_data.rolling(window=window).mean()
    last_ma = ma.iloc[-1]
    predictions = np.full(len(test_data), last_ma)
    return predictions
