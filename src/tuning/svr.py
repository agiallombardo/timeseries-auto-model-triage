import logging
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from ..model_config import TUNING_SETUP

logger = logging.getLogger(__name__)


def grid_search_svr(X_train, X_test, y_train, y_test, results_dir=None, **kwargs):
    """Grid search for SVR. Returns (best_params, best_predictions, scalers)."""
    logger.info("Performing grid search for SVR model...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_X.transform(X_test)

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1, 0.2],
    }
    n_splits = TUNING_SETUP.get("n_splits", 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = list(ParameterGrid(param_grid))
    cv_results = []
    best_score = float("-inf")
    best_params = None

    for params in tqdm(candidates, desc="SVR grid", unit="candidate", leave=False):
        model = SVR(**params)
        scores = cross_val_score(
            model, X_train_scaled, y_train_scaled, cv=tscv,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        mean_score = scores.mean()
        cv_results.append({**params, "mean_test_score": -mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        del model

    best_rmse = -best_score
    logger.info(f"Best SVR parameters: {best_params} with RMSE: {best_rmse:.4f}")

    best_model = SVR(**best_params)
    best_model.fit(X_train_scaled, y_train_scaled)
    predictions_scaled = best_model.predict(X_test_scaled)
    best_predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    path = os.path.join(results_dir, 'svr_grid_search_results.csv') if results_dir else 'svr_grid_search_results.csv'
    pd.DataFrame(cv_results).sort_values('mean_test_score').to_csv(path, index=False)
    return best_params, best_predictions, (scaler_X, scaler_y)
