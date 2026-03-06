import numpy as np
import pandas as pd
import logging
from prophet import Prophet

logger = logging.getLogger(__name__)

def run_prophet(train_data, test_data, **kwargs):
    """Run Facebook Prophet model. kwargs are passed to Prophet()."""
    logger.info("Training Prophet model...")
    
    # Prepare data in Prophet format
    train_df = pd.DataFrame({
        'ds': train_data.index,
        'y': train_data.values
    })
    
    # Initialize and fit Prophet model
    model = Prophet(**kwargs)
    model.fit(train_df)
    
    # Create future dataframe for forecasting
    future = pd.DataFrame({'ds': test_data.index})
    
    # Make predictions
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    return predictions