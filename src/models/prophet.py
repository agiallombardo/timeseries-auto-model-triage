import numpy as np
import pandas as pd
import logging
from prophet import Prophet

logger = logging.getLogger(__name__)

# Prophet() constructor args only; avoid passing pipeline params (X_train, y_train, etc.)
PROPHET_INIT_KWARGS = (
    "changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode",
    "yearly_seasonality", "weekly_seasonality", "daily_seasonality",
    "growth", "interval_width", "uncertainty_samples",
)

def run_prophet(train_data, test_data, **kwargs):
    """Run Facebook Prophet model. Only Prophet constructor kwargs are passed to Prophet()."""
    logger.info("Training Prophet model...")
    prophet_kwargs = {k: v for k, v in kwargs.items() if k in PROPHET_INIT_KWARGS}

    # Prepare data in Prophet format
    train_df = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    # Initialize and fit Prophet model
    model = Prophet(**prophet_kwargs)
    model.fit(train_df)
    
    # Create future dataframe for forecasting
    future = pd.DataFrame({'ds': test_data.index})
    
    # Make predictions
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    return predictions