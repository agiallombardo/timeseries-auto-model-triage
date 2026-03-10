import logging
import os
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit

from ..model_config import TUNING_SETUP
from ..models.random_forest import run_random_forest
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
    rf = RandomForestRegressor(random_state=42, criterion=criterion)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0,
    )
    n_candidates = len(list(ParameterGrid(param_grid)))
    with tqdm(total=n_candidates, desc=f"RF ({loss}) grid", unit="candidate", leave=False) as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(n_candidates)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    logger.info(f"Best Random Forest parameters: {best_params} with RMSE: {best_score:.4f}")

    best_model = RandomForestRegressor(random_state=42, criterion=criterion, **best_params)
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)

    if results_dir:
        grid_path = os.path.join(results_dir, 'rf_grid_search_results.csv')
        fi_path = os.path.join(results_dir, 'rf_feature_importance.csv')
    else:
        grid_path, fi_path = 'rf_grid_search_results.csv', 'rf_feature_importance.csv'
    pd.DataFrame(grid_search.cv_results_).assign(
        mean_test_score=lambda df: -df['mean_test_score']
    ).sort_values('mean_test_score').to_csv(grid_path, index=False)
    pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_,
    }).sort_values('importance', ascending=False).to_csv(fi_path, index=False)
    return best_params, best_predictions
