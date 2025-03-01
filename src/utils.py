import os
import logging
import sys
from datetime import datetime

def setup_logging(output_dir, log_file='time_series_forecast.log'):
    """
    Configure logging to file and console.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the log file
    log_file : str, optional
        Name of the log file
    
    Returns:
    --------
    logging.Logger
        Configured logger object
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    log_path = os.path.join(output_dir, log_file)
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and return a named logger for the caller
    logger = logging.getLogger(__name__)
    return logger