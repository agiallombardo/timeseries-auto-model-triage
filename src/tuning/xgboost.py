import logging
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from ..model_config import TUNING_SETUP
from ..losses import get_xgb_params

logger = logging.getLogger(__name__)


def grid_search_xgboost(X_train, X_test, y_train, y_test, loss='l2', results_dir=None, **kwargs):
    """Grid search for XGBoost. Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for XGBoost ({loss.upper()}) model...")
    xgb_extra = get_xgb_params(loss)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    n_splits = TUNING_SETUP.get("n_splits", 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb = XGBRegressor(random_state=42, **xgb_extra)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    logger.info(f"Best XGBoost parameters: {best_params} with RMSE: {best_score:.4f}")

    best_model = XGBRegressor(random_state=42, **xgb_extra, **best_params)
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)

    if results_dir:
        grid_path = os.path.join(results_dir, 'xgb_grid_search_results.csv')
        fi_path = os.path.join(results_dir, 'xgb_feature_importance.csv')
    else:
        grid_path, fi_path = 'xgb_grid_search_results.csv', 'xgb_feature_importance.csv'
    pd.DataFrame(grid_search.cv_results_).assign(
        mean_test_score=lambda df: -df['mean_test_score']
    ).sort_values('mean_test_score').to_csv(grid_path, index=False)
    pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_,
    }).sort_values('importance', ascending=False).to_csv(fi_path, index=False)
    return best_params, best_predictions
