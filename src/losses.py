"""
Centralized loss / objective mapping for every framework used in the pipeline.

Each helper converts a canonical loss key ('l1', 'l2', 'huber', 'quantile')
into the framework-specific argument expected by that model.
"""

import tensorflow as tf
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.linear_model import QuantileRegressor

LOSS_KEYS = ['l1', 'l2', 'huber', 'quantile']

LOSS_DISPLAY = {
    'l1': 'L1',
    'l2': 'L2',
    'huber': 'Huber',
    'quantile': 'Quantile',
}

# -- Which losses each model category supports --------------------------------

LOSS_SUPPORT = {
    'keras':       ['l1', 'l2', 'huber', 'quantile'],
    'xgb':         ['l1', 'l2', 'huber', 'quantile'],
    'rf':          ['l1', 'l2'],
    'lr':          ['l1', 'l2', 'huber', 'quantile'],
    'svr':         ['l2'],
    'statistical': [],
}

# Model registry key -> list of loss keys to try (no CLI; always test all)
LOSS_SUPPORTED_MODELS = {
    'lr':        ['l1', 'l2', 'huber', 'quantile'],
    'mlp':       ['l1', 'l2', 'huber', 'quantile'],
    'lstm_feat': ['l1', 'l2', 'huber', 'quantile'],
    'rnn_feat':  ['l1', 'l2', 'huber', 'quantile'],
    'cnn1d':     ['l1', 'l2', 'huber', 'quantile'],
    'rnn':       ['l1', 'l2', 'huber', 'quantile'],
    'lstm':      ['l1', 'l2', 'huber', 'quantile'],
    'xgb':       ['l1', 'l2', 'huber', 'quantile'],
    'rf':        ['l1', 'l2'],
}

# -- Keras ---------------------------------------------------------------------

def _pinball_loss(quantile=0.5):
    """Keras-compatible pinball (quantile) loss."""
    def loss_fn(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * err, (quantile - 1) * err))
    loss_fn.__name__ = f'pinball_q{quantile}'
    return loss_fn


def get_keras_loss(loss_key, quantile=0.5):
    mapping = {
        'l2':       'mse',
        'l1':       'mae',
        'huber':    tf.keras.losses.Huber(delta=1.0),
        'quantile': _pinball_loss(quantile),
    }
    if loss_key not in mapping:
        raise ValueError(f"Unsupported Keras loss key: {loss_key}")
    return mapping[loss_key]


# -- XGBoost -------------------------------------------------------------------

def get_xgb_params(loss_key, quantile=0.5):
    """Return dict of extra XGBRegressor kwargs for the requested loss."""
    mapping = {
        'l2':       {'objective': 'reg:squarederror'},
        'l1':       {'objective': 'reg:absoluteerror'},
        'huber':    {'objective': 'reg:pseudohubererror'},
        'quantile': {'objective': 'reg:quantileerror',
                     'quantile_alpha': quantile},
    }
    if loss_key not in mapping:
        raise ValueError(f"Unsupported XGBoost loss key: {loss_key}")
    return mapping[loss_key]


# -- Random Forest -------------------------------------------------------------

def get_rf_criterion(loss_key):
    mapping = {
        'l2': 'squared_error',
        'l1': 'absolute_error',
    }
    if loss_key not in mapping:
        raise ValueError(
            f"Random Forest only supports l1/l2, got: {loss_key}"
        )
    return mapping[loss_key]


# -- Linear Regression ---------------------------------------------------------

def get_linear_model(loss_key, quantile=0.5, alpha=None):
    """Return an unfitted sklearn estimator matching the requested loss.

    alpha : float or None
        Regularization strength for Lasso (l1). When None the default 0.01 is used.
        Has no effect for l2, huber, or quantile.
    """
    if loss_key == 'l2':
        return LinearRegression()
    if loss_key == 'l1':
        _alpha = alpha if alpha is not None else 0.01
        return Lasso(alpha=_alpha, max_iter=10000)
    if loss_key == 'huber':
        return HuberRegressor(max_iter=2000)
    if loss_key == 'quantile':
        return QuantileRegressor(quantile=quantile, alpha=0.0,
                                 solver='highs')
    raise ValueError(f"Unsupported linear regression loss key: {loss_key}")
