"""
Registry: maps model keys to runner and tuning functions.
Each model lives in its own file; this module only routes and passes loss.
"""

import logging
from ..losses import LOSS_DISPLAY

from .arima import run_arima
from .sarima import run_sarima
from .moving_average import run_moving_average
from .exponential_smoothing import run_exponential_smoothing
from .prophet import run_prophet
from .random_forest import run_random_forest
from .svr import run_svr
from .xgboost import run_xgboost
from .linear_regression import run_linear_regression
from .rnn import run_rnn
from .lstm import run_lstm
from .mlp import run_mlp
from .lstm_feat import run_lstm_features
from .rnn_feat import run_rnn_features
from .cnn1d import run_cnn1d

from ..tuning.arima import grid_search_arima
from ..tuning.sarima import grid_search_sarima
from ..tuning.moving_average import grid_search_moving_average
from ..tuning.exponential_smoothing import grid_search_exponential_smoothing
from ..tuning.prophet import grid_search_prophet
from ..tuning.random_forest import grid_search_random_forest
from ..tuning.svr import grid_search_svr
from ..tuning.xgboost import grid_search_xgboost
from ..tuning.linear_regression import grid_search_linear_regression
from ..tuning.rnn import grid_search_rnn
from ..tuning.lstm import grid_search_lstm
from ..tuning.mlp import grid_search_mlp
from ..tuning.lstm_feat import grid_search_lstm_features
from ..tuning.rnn_feat import grid_search_rnn_features
from ..tuning.cnn1d import grid_search_cnn1d

logger = logging.getLogger(__name__)

# Base display names (loss suffix is added in main when applicable)
DISPLAY_NAMES = {
    'arima': 'ARIMA',
    'sarima': 'SARIMA',
    'ma': 'Moving Average',
    'es': 'Exponential Smoothing',
    'prophet': 'Prophet',
    'rf': 'Random Forest',
    'svr': 'SVR',
    'xgb': 'XGBoost',
    'lr': 'Linear Regression',
    'rnn': 'RNN',
    'lstm': 'LSTM',
    'mlp': 'MLP',
    'lstm_feat': 'LSTM-feat',
    'rnn_feat': 'RNN-feat',
    'cnn1d': 'CNN-1D',
}


def get_available_models():
    return {
        'arima': run_arima_wrapper,
        'sarima': run_sarima_wrapper,
        'ma': run_ma_wrapper,
        'es': run_es_wrapper,
        'prophet': run_prophet_wrapper,
        'rf': run_rf_wrapper,
        'svr': run_svr_wrapper,
        'xgb': run_xgb_wrapper,
        'lr': run_lr_wrapper,
        'rnn': run_rnn_wrapper,
        'lstm': run_lstm_wrapper,
        'mlp': run_mlp_wrapper,
        'lstm_feat': run_lstm_feat_wrapper,
        'rnn_feat': run_rnn_feat_wrapper,
        'cnn1d': run_cnn1d_wrapper,
    }


def get_tuning_functions():
    return {
        'arima': tune_arima_wrapper,
        'sarima': tune_sarima_wrapper,
        'ma': tune_ma_wrapper,
        'es': tune_es_wrapper,
        'prophet': tune_prophet_wrapper,
        'rf': tune_rf_wrapper,
        'svr': tune_svr_wrapper,
        'xgb': tune_xgb_wrapper,
        'lr': tune_lr_wrapper,
        'rnn': tune_rnn_wrapper,
        'lstm': tune_lstm_wrapper,
        'mlp': tune_mlp_wrapper,
        'lstm_feat': tune_lstm_feat_wrapper,
        'rnn_feat': tune_rnn_feat_wrapper,
        'cnn1d': tune_cnn1d_wrapper,
    }


def _name_with_loss(base_name, loss_key):
    if loss_key is None:
        return base_name
    return f"{base_name} ({LOSS_DISPLAY.get(loss_key, loss_key.upper())})"


# ---- Run wrappers (return (predictions, display_name)) ---------------------

def run_arima_wrapper(y_train, y_test, order=(5, 1, 0), **kwargs):
    order = tuple(order) if isinstance(order, list) else order
    pred = run_arima(y_train, y_test, order=order)
    return pred, f"ARIMA{order}"

def run_sarima_wrapper(y_train, y_test, seasonal_periods=12, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), **kwargs):
    order = tuple(order) if isinstance(order, list) else order
    so = tuple(seasonal_order) if isinstance(seasonal_order, list) else seasonal_order
    if len(so) == 4 and so[3] != seasonal_periods:
        so = (so[0], so[1], so[2], seasonal_periods)
    pred = run_sarima(y_train, y_test, order=order, seasonal_order=so)
    return pred, f"SARIMA{order}x{so}"

def run_ma_wrapper(y_train, y_test, ma_window=3, window=None, **kwargs):
    w = window if window is not None else ma_window
    pred = run_moving_average(y_train, y_test, window=w)
    return pred, f"{DISPLAY_NAMES['ma']} (w={w})"

def run_es_wrapper(y_train, y_test, seasonal_periods=12, trend='add', seasonal='add', **kwargs):
    pred = run_exponential_smoothing(
        y_train, y_test,
        trend=trend, seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
    )
    return pred, f"{DISPLAY_NAMES['es']} (trend={trend}, seasonal={seasonal})"

def run_prophet_wrapper(y_train, y_test, **kwargs):
    pred = run_prophet(y_train, y_test, **kwargs)
    return pred, DISPLAY_NAMES['prophet']

def run_rf_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    pred = run_random_forest(X_train, X_test, y_train, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['rf'], loss)

def run_svr_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    svr_kwargs = {k: v for k, v in kwargs.items() if k in ('C', 'kernel', 'gamma', 'epsilon')}
    pred, _ = run_svr(X_train, X_test, y_train, **svr_kwargs)
    return pred, DISPLAY_NAMES['svr']

def run_xgb_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    pred = run_xgboost(X_train, X_test, y_train, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['xgb'], loss)

def run_lr_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    pred = run_linear_regression(X_train, X_test, y_train, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['lr'], loss)

def run_rnn_wrapper(y_train, y_test, n_steps=3, loss='l2', **kwargs):
    pred, _ = run_rnn(y_train, y_test, n_steps=n_steps, loss=loss, **kwargs)
    return pred, _name_with_loss(DISPLAY_NAMES['rnn'], loss)

def run_lstm_wrapper(y_train, y_test, n_steps=3, loss='l2', **kwargs):
    pred, _ = run_lstm(y_train, y_test, n_steps=n_steps, loss=loss, **kwargs)
    return pred, _name_with_loss(DISPLAY_NAMES['lstm'], loss)

def run_mlp_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    pred = run_mlp(X_train, X_test, y_train, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['mlp'], loss)

def run_lstm_feat_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    pred = run_lstm_features(X_train, X_test, y_train, y_test, n_steps=n_steps, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['lstm_feat'], loss)

def run_rnn_feat_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    pred = run_rnn_features(X_train, X_test, y_train, y_test, n_steps=n_steps, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['rnn_feat'], loss)

def run_cnn1d_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    pred = run_cnn1d(X_train, X_test, y_train, y_test, n_steps=n_steps, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['cnn1d'], loss)


# ---- Tune wrappers (return (predictions, display_name, best_params)); best_params are dicts -------

def tune_arima_wrapper(y_train, y_test, **kwargs):
    best_params, pred = grid_search_arima(y_train, y_test)
    order = tuple(best_params["order"]) if isinstance(best_params.get("order"), list) else best_params.get("order")
    return pred, f"ARIMA{order} (Tuned)", best_params

def tune_sarima_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    best_params, pred = grid_search_sarima(y_train, y_test, seasonal_periods)
    o = best_params.get("order", [])
    s = best_params.get("seasonal_order", [])
    return pred, f"SARIMA{o}x{s} (Tuned)", best_params

def tune_ma_wrapper(y_train, y_test, **kwargs):
    best_params, pred = grid_search_moving_average(y_train, y_test)
    w = best_params.get("window", 3)
    return pred, f"{DISPLAY_NAMES['ma']} (w={w}, Tuned)", best_params

def tune_es_wrapper(y_train, y_test, seasonal_periods=12, **kwargs):
    best_params, pred = grid_search_exponential_smoothing(y_train, y_test, seasonal_periods)
    return pred, f"{DISPLAY_NAMES['es']} (Tuned)", best_params

def tune_prophet_wrapper(y_train, y_test, **kwargs):
    best_params, pred = grid_search_prophet(y_train, y_test)
    return pred, "Prophet (Tuned)", best_params

def tune_rf_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    best_params, pred = grid_search_random_forest(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['rf'], loss) + " (Tuned)", best_params

def tune_svr_wrapper(y_train, y_test, X_train=None, X_test=None, **kwargs):
    best_params, pred, _ = grid_search_svr(X_train, X_test, y_train, y_test)
    return pred, "SVR (Tuned)", best_params

def tune_xgb_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    best_params, pred = grid_search_xgboost(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['xgb'], loss) + " (Tuned)", best_params

def tune_lr_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    best_params, pred = grid_search_linear_regression(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['lr'], loss) + " (Tuned)", best_params

def tune_rnn_wrapper(y_train, y_test, n_steps=3, loss='l2', **kwargs):
    best_params, pred, _ = grid_search_rnn(y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['rnn'], loss) + " (Tuned)", best_params

def tune_lstm_wrapper(y_train, y_test, n_steps=3, loss='l2', **kwargs):
    best_params, pred, _ = grid_search_lstm(y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['lstm'], loss) + " (Tuned)", best_params

def tune_mlp_wrapper(y_train, y_test, X_train=None, X_test=None, loss='l2', **kwargs):
    best_params, pred = grid_search_mlp(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['mlp'], loss) + " (Tuned)", best_params

def tune_lstm_feat_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    best_params, pred = grid_search_lstm_features(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['lstm_feat'], loss) + " (Tuned)", best_params

def tune_rnn_feat_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    best_params, pred = grid_search_rnn_features(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['rnn_feat'], loss) + " (Tuned)", best_params

def tune_cnn1d_wrapper(y_train, y_test, X_train=None, X_test=None, n_steps=5, loss='l2', **kwargs):
    best_params, pred = grid_search_cnn1d(X_train, X_test, y_train, y_test, loss=loss)
    return pred, _name_with_loss(DISPLAY_NAMES['cnn1d'], loss) + " (Tuned)", best_params
