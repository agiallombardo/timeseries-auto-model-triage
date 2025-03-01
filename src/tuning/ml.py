import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import itertools
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

def grid_search_random_forest(X_train, X_test, y_train, y_test):
    """
    Grid search for Random Forest hyperparameters.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target values
    y_test : Series
        Test target values
        
    Returns:
    --------
    tuple
        Best parameters, best predictions
    """
    logger.info("Performing grid search for Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create time series split for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create grid search model
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the grid search model
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to RMSE
    
    logger.info(f"Best Random Forest parameters: {best_params} with RMSE: {best_score:.4f}")
    
    # Train the model with best parameters
    best_model = RandomForestRegressor(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions
    best_predictions = best_model.predict(X_test)
    
    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    # Convert negative RMSE scores back to positive
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df.sort_values('mean_test_score')
    results_df.to_csv('rf_grid_search_results.csv', index=False)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('rf_feature_importance.csv', index=False)
    
    return best_params, best_predictions

def grid_search_svr(X_train, X_test, y_train, y_test):
    """
    Grid search for SVR hyperparameters.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target values
    y_test : Series
        Test target values
        
    Returns:
    --------
    tuple
        (best_params, best_predictions, scalers)
    """
    logger.info("Performing grid search for SVR model...")
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_X.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    # Create time series split for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create grid search model
    svr = SVR()
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search model
    grid_search.fit(X_train_scaled, y_train_scaled)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to RMSE
    
    logger.info(f"Best SVR parameters: {best_params} with RMSE: {best_score:.4f}")
    
    # Train the model with best parameters
    best_model = SVR(**best_params)
    best_model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    predictions_scaled = best_model.predict(X_test_scaled)
    best_predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
    
    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    # Convert negative RMSE scores back to positive
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df.sort_values('mean_test_score')
    results_df.to_csv('svr_grid_search_results.csv', index=False)
    
    # Return tuple of (best_params, predictions, scalers)
    scalers = (scaler_X, scaler_y)
    return best_params, best_predictions, scalers
    
def grid_search_xgboost(X_train, X_test, y_train, y_test):
    """
    Grid search for XGBoost hyperparameters.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    y_train : Series
        Training target values
    y_test : Series
        Test target values
        
    Returns:
    --------
    tuple
        Best parameters, best predictions
    """
    logger.info("Performing grid search for XGBoost model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create time series split for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create grid search model
    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the grid search model
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to RMSE
    
    logger.info(f"Best XGBoost parameters: {best_params} with RMSE: {best_score:.4f}")
    
    # Train the model with best parameters
    best_model = XGBRegressor(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions
    best_predictions = best_model.predict(X_test)
    
    # Save results
    results_df = pd.DataFrame(grid_search.cv_results_)
    # Convert negative RMSE scores back to positive
    results_df['mean_test_score'] = -results_df['mean_test_score']
    results_df = results_df.sort_values('mean_test_score')
    results_df.to_csv('xgb_grid_search_results.csv', index=False)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('xgb_feature_importance.csv', index=False)
    
    return best_params, best_predictions

