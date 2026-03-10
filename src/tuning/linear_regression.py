import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from ..model_config import TUNING_SETUP
from ..models.linear_regression import run_linear_regression
from ._validation import get_time_series_splits, aggregate_fold_scores, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_linear_regression(X_train, X_test, y_train, y_test, loss='l2'):
    """Grid search for Linear Regression (Ridge/Lasso/Huber/Quantile). Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for Linear Regression ({loss.upper()}) model...")
    n_splits = TUNING_SETUP.get("n_splits", 3)
    splits = list(get_time_series_splits(X_train, n_splits=n_splits))
    best_rmse = float('inf')
    best_params = None
    best_predictions = None

    if loss == 'l2':
        for alpha in tqdm([0.01, 0.1, 1.0, 10.0], desc="LR (L2) grid", unit="candidate", leave=False):
            model = Ridge(alpha=alpha)
            rmses = []
            for train_idx, val_idx in splits:
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(score_validation_rmse(y_v, model.predict(X_v)))
            rmse = aggregate_fold_scores(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'alpha': alpha}
        if best_params is not None:
            model = Ridge(**best_params)
            model.fit(X_train, y_train)
            best_predictions = model.predict(X_test)
    elif loss == 'l1':
        for alpha in tqdm([0.001, 0.01, 0.1, 1.0], desc="LR (L1) grid", unit="candidate", leave=False):
            model = Lasso(alpha=alpha, max_iter=10000)
            rmses = []
            for train_idx, val_idx in splits:
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(score_validation_rmse(y_v, model.predict(X_v)))
            rmse = aggregate_fold_scores(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'alpha': alpha}
        if best_params is not None:
            model = Lasso(**best_params, max_iter=10000)
            model.fit(X_train, y_train)
            best_predictions = model.predict(X_test)
    elif loss == 'huber':
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_train_sc = pd.DataFrame(X_train_sc, index=X_train.index, columns=X_train.columns)
        X_test_sc = pd.DataFrame(X_test_sc, index=X_test.index, columns=X_test.columns)
        for epsilon in tqdm([1.0, 1.35, 2.0], desc="LR (Huber) grid", unit="candidate", leave=False):
            model = HuberRegressor(epsilon=epsilon, max_iter=2000)
            rmses = []
            for train_idx, val_idx in splits:
                X_t, X_v = X_train_sc.iloc[train_idx], X_train_sc.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_t, y_t)
                rmses.append(score_validation_rmse(y_v, model.predict(X_v)))
            rmse = aggregate_fold_scores(rmses)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'epsilon': epsilon}
        if best_params is not None:
            model = HuberRegressor(**best_params, max_iter=2000)
            model.fit(X_train_sc, y_train)
            best_predictions = model.predict(X_test_sc)
    else:
        best_predictions = run_linear_regression(X_train, X_test, y_train, loss=loss)
        best_params = {'quantile': 0.5}

    if best_predictions is None:
        best_predictions = run_linear_regression(X_train, X_test, y_train, loss=loss)
        best_params = {}
    logger.info(f"Best Linear Regression ({loss}) parameters: {best_params} with RMSE: {best_rmse:.4f}")
    return best_params, best_predictions
