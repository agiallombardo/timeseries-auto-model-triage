import logging
import os
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit, cross_val_score

from ..model_config import TUNING_SETUP
from ..losses import get_rf_criterion

logger = logging.getLogger(__name__)


def grid_search_random_forest(X_train, X_test, y_train, y_test, loss='l2', results_dir=None, **kwargs):
    """Grid search for Random Forest. Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for Random Forest ({loss.upper()}) model...")
    criterion = get_rf_criterion(loss)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    n_splits = TUNING_SETUP.get("n_splits", 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = list(ParameterGrid(param_grid))
    cv_results = []
    best_score = float("-inf")
    best_params = None

    for params in tqdm(candidates, desc=f"RF ({loss}) grid", unit="candidate", leave=False):
        model = RandomForestRegressor(random_state=42, criterion=criterion, **params)
        scores = cross_val_score(
            model, X_train, y_train, cv=tscv,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        mean_score = scores.mean()
        cv_results.append({**params, "mean_test_score": -mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        del model

    best_rmse = -best_score
    logger.info(f"Best Random Forest parameters: {best_params} with RMSE: {best_rmse:.4f}")

    best_model = RandomForestRegressor(random_state=42, criterion=criterion, **best_params)
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)

    if results_dir:
        grid_path = os.path.join(results_dir, 'rf_grid_search_results.csv')
        fi_path = os.path.join(results_dir, 'rf_feature_importance.csv')
    else:
        grid_path, fi_path = 'rf_grid_search_results.csv', 'rf_feature_importance.csv'
    pd.DataFrame(cv_results).sort_values('mean_test_score').to_csv(grid_path, index=False)
    pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_,
    }).sort_values('importance', ascending=False).to_csv(fi_path, index=False)
    return best_params, best_predictions
