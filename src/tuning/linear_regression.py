import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from ..models.linear_regression import run_linear_regression

logger = logging.getLogger(__name__)


def grid_search_linear_regression(X_train, X_test, y_train, y_test, loss='l2'):
    """Grid search for Linear Regression (Ridge/Lasso/Huber/Quantile). Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for Linear Regression ({loss.upper()}) model...")
    tscv = TimeSeriesSplit(n_splits=3)
    best_rmse = float('inf')
    best_params = None
    best_predictions = None

    if loss == 'l2':
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha)
            rmses = []
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(np.sqrt(mean_squared_error(y_v, model.predict(X_v))))
            rmse = np.mean(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'alpha': alpha}
        if best_params is not None:
            model = Ridge(**best_params)
            model.fit(X_train, y_train)
            best_predictions = model.predict(X_test)
    elif loss == 'l1':
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            model = Lasso(alpha=alpha, max_iter=10000)
            rmses = []
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(np.sqrt(mean_squared_error(y_v, model.predict(X_v))))
            rmse = np.mean(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'alpha': alpha}
        if best_params is not None:
            model = Lasso(**best_params, max_iter=10000)
            model.fit(X_train, y_train)
            best_predictions = model.predict(X_test)
    elif loss == 'huber':
        for epsilon in [1.0, 1.35, 2.0]:
            model = HuberRegressor(epsilon=epsilon, max_iter=200)
            rmses = []
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(np.sqrt(mean_squared_error(y_v, model.predict(X_v))))
            rmse = np.mean(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'epsilon': epsilon}
        if best_params is not None:
            model = HuberRegressor(**best_params, max_iter=200)
            model.fit(X_train, y_train)
            best_predictions = model.predict(X_test)
    else:
        best_predictions = run_linear_regression(X_train, X_test, y_train, loss=loss)
        best_params = {'quantile': 0.5}

    if best_predictions is None:
        best_predictions = run_linear_regression(X_train, X_test, y_train, loss=loss)
        best_params = {}
    logger.info(f"Best Linear Regression ({loss}) parameters: {best_params} with RMSE: {best_rmse:.4f}")
    return best_params, best_predictions
