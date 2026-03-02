import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def grid_search_svr(X_train, X_test, y_train, y_test):
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
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_scaled, y_train_scaled)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    logger.info(f"Best SVR parameters: {best_params} with RMSE: {best_score:.4f}")

    best_model = SVR(**best_params)
    best_model.fit(X_train_scaled, y_train_scaled)
    predictions_scaled = best_model.predict(X_test_scaled)
    best_predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    pd.DataFrame(grid_search.cv_results_).assign(
        mean_test_score=lambda df: -df['mean_test_score']
    ).sort_values('mean_test_score').to_csv('svr_grid_search_results.csv', index=False)
    return best_params, best_predictions, (scaler_X, scaler_y)
