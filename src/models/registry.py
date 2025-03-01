import logging

# Module-level logger
logger = logging.getLogger(__name__)

from .statistical import run_arima, run_sarima, run_moving_average, run_exponential_smoothing
from .ml import run_random_forest, run_svr, run_xgboost
from .deep_learning import run_rnn, run_lstm
from .prophet import run_prophet

from ..tuning.statistical import grid_search_arima, grid_search_sarima, grid_search_moving_average, grid_search_exponential_smoothing
from ..tuning.ml import grid_search_random_forest, grid_search_svr, grid_search_xgboost
from ..tuning.deep_learning import grid_search_rnn, grid_search_lstm
from ..tuning.prophet import grid_search_prophet

def get_available_models():
    """Get dictionary of available forecasting models."""
    return {
        'arima': run_arima_wrapper,
        'sarima': run_sarima_wrapper,
        'ma': run_ma_wrapper,
        'es': run_es_wrapper,
        'prophet': run_prophet_wrapper,
        'rf': run_rf_wrapper,
        'svr': run_svr_wrapper,
        'xgb': run_xgb_wrapper,
        'rnn': run_rnn_wrapper,
        'lstm': run_lstm_wrapper
    }

def get_tuning_functions():
    """Get dictionary of available tuning functions."""
    return {
        'arima': tune_arima_wrapper,
        'sarima': tune_sarima_wrapper,
        'ma': tune_ma_wrapper,
        'es': tune_es_wrapper,
        'prophet': tune_prophet_wrapper,
        'rf': tune_rf_wrapper,
        'svr': tune_svr_wrapper,
        'xgb': tune_xgb_wrapper,
        'rnn': tune_rnn_wrapper,
        'lstm': tune_lstm_wrapper
    }

# Model wrapper functions
def run_arima_wrapper(y_train, y_test, **kwargs):
    pred = run_arima(y_train, y_test)
    return pred, "ARIMA"

def run_sarima_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    pred = run_sarima(y_train, y_test, seasonal_order=(1,1,1,seasonal_periods))
    return pred, "SARIMA"

def run_ma_wrapper(y_train, y_test, ma_window=3, **kwargs):
    pred = run_moving_average(y_train, y_test, window=ma_window)
    return pred, f"Moving Average (w={ma_window})"

def run_es_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    pred = run_exponential_smoothing(y_train, y_test, seasonal_periods=seasonal_periods)
    return pred, "Exponential Smoothing"

def run_prophet_wrapper(y_train, y_test, **kwargs):
    pred = run_prophet(y_train, y_test)
    return pred, "Prophet"

def run_rf_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    pred = run_random_forest(X_train, X_test, y_train)
    return pred, "Random Forest"

def run_svr_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    pred, _ = run_svr(X_train, X_test, y_train)
    return pred, "SVR"

def run_xgb_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    pred = run_xgboost(X_train, X_test, y_train)
    return pred, "XGBoost"

def run_rnn_wrapper(y_train, y_test, n_steps=3, **kwargs):
    pred, _ = run_rnn(y_train, y_test, n_steps=n_steps)
    return pred, "RNN"

def run_lstm_wrapper(y_train, y_test, n_steps=3, **kwargs):
    pred, _ = run_lstm(y_train, y_test, n_steps=n_steps)
    return pred, "LSTM"

# Tuning wrapper functions 
def tune_arima_wrapper(y_train, y_test, **kwargs):
    best_params, pred = grid_search_arima(y_train, y_test)
    return pred, f"ARIMA{best_params} (Tuned)", best_params

def tune_sarima_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    best_params, pred = grid_search_sarima(y_train, y_test, seasonal_periods)
    return pred, f"SARIMA{best_params[0]}x{best_params[1]} (Tuned)", best_params

def tune_ma_wrapper(y_train, y_test, **kwargs):
    best_window, pred = grid_search_moving_average(y_train, y_test)
    return pred, f"Moving Average (w={best_window}, Tuned)", best_window

def tune_es_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    best_params, pred = grid_search_exponential_smoothing(y_train, y_test, seasonal_periods)
    return pred, f"Exponential Smoothing (trend={best_params[0]}, seasonal={best_params[1]}, Tuned)", best_params

def tune_prophet_wrapper(y_train, y_test, **kwargs):
    best_params, pred = grid_search_prophet(y_train, y_test)
    return pred, "Prophet (Tuned)", best_params

def tune_rf_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    best_params, pred = grid_search_random_forest(X_train, X_test, y_train, y_test)
    return pred, "Random Forest (Tuned)", best_params

def tune_svr_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    best_params, pred, scalers = grid_search_svr(X_train, X_test, y_train, y_test)
    return pred, "SVR (Tuned)", best_params

def tune_xgb_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    best_params, pred = grid_search_xgboost(X_train, X_test, y_train, y_test)
    return pred, "XGBoost (Tuned)", best_params

def tune_rnn_wrapper(y_train, y_test, n_steps=3, **kwargs):
    best_params, pred, _ = grid_search_rnn(y_train, y_test)
    return pred, "RNN (Tuned)", best_params

def tune_lstm_wrapper(y_train, y_test, n_steps=3, **kwargs):
    best_params, pred, _ = grid_search_lstm(y_train, y_test)
    return pred, "LSTM (Tuned)", best_params