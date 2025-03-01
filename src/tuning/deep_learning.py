import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


def grid_search_rnn(train_data, test_data):
    """
    Grid search for RNN hyperparameters.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
        
    Returns:
    --------
    tuple
        Best parameters, best predictions, scaler
    """
    logger.info("Performing grid search for RNN model...")
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Define hyperparameters to search
    n_steps_list = [3, 5, 7]
    units_list = [32, 50, 100]
    learning_rates = [0.001, 0.01, 0.0001]
    batch_sizes = [16, 32, 64]
    
    # Validation set
    val_size = int(len(train_scaled) * 0.2)
    train_set = train_scaled[:-val_size]
    val_set = train_scaled[-val_size:]
    
    best_rmse = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Loop through parameter combinations
    for n_steps in n_steps_list:
        for units in units_list:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    # Prepare sequence data
                    X_train, y_train = prepare_sequence_data(train_set, n_steps)
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    
                    X_val, y_val = prepare_sequence_data(val_set, n_steps)
                    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                    
                    # Skip if not enough data
                    if len(X_train) < batch_size or len(X_val) == 0:
                        continue
                    
                    # Build model
                    model = create_rnn_model((n_steps, 1), units, lr)
                    
                    # Train model
                    try:
                        model.fit(
                            X_train, y_train,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Evaluate on validation set
                        val_pred = model.predict(X_val, verbose=0).ravel()
                        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                        
                        results.append({
                            'n_steps': n_steps,
                            'units': units,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'rmse': rmse
                        })
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'n_steps': n_steps,
                                'units': units,
                                'learning_rate': lr,
                                'batch_size': batch_size
                            }
                            best_model = model
                            
                        logger.debug(f"RNN(n_steps={n_steps}, units={units}, lr={lr}, batch_size={batch_size}) - RMSE: {rmse:.4f}")
                    except Exception as e:
                        logger.warning(f"Error training RNN: {e}")
                        continue
    
    logger.info(f"Best RNN parameters: {best_params} with RMSE: {best_rmse:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df.to_csv('rnn_grid_search_results.csv', index=False)
    
    # Generate predictions with best model
    if best_model is not None:
        # Retrain on full training data
        X_train_full, y_train_full = prepare_sequence_data(train_scaled, best_params['n_steps'])
        X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1], 1)
        
        best_model.fit(
            X_train_full, y_train_full,
            epochs=100,
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate predictions for test data
        predictions = []
        last_sequence = train_scaled[-best_params['n_steps']:].reshape(1, best_params['n_steps'], 1)
        
        for _ in range(len(test_data)):
            next_value = best_model.predict(last_sequence, verbose=0)
            predictions.append(next_value[0, 0])
            
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_value
        
        # Inverse transform predictions
        best_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    else:
        # Fallback to default parameters if no valid model was found
        logger.warning("No valid RNN model found during grid search. Using default parameters.")
        best_predictions, _ = run_rnn(train_data, test_data)
    
    return best_params, best_predictions, scaler


def grid_search_lstm(train_data, test_data):
    """
    Grid search for LSTM hyperparameters.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
        
    Returns:
    --------
    tuple
        Best parameters, best predictions, scaler
    """
    logger.info("Performing grid search for LSTM model...")
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Define hyperparameters to search
    n_steps_list = [3, 5, 7]
    units_list = [32, 50, 100]
    layers_list = [1, 2]
    dropout_rates = [0.0, 0.2]
    learning_rates = [0.001, 0.01]
    batch_sizes = [16, 32]
    
    # Validation set
    val_size = int(len(train_scaled) * 0.2)
    train_set = train_scaled[:-val_size]
    val_set = train_scaled[-val_size:]
    
    best_rmse = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Limit number of combinations to avoid excessive computation
    max_combinations = 20
    combinations_tested = 0
    
    # Loop through parameter combinations
    for n_steps in n_steps_list:
        for units in units_list:
            for layers in layers_list:
                for dropout_rate in dropout_rates:
                    for lr in learning_rates:
                        for batch_size in batch_sizes:
                            # Check if we've reached the maximum number of combinations
                            if combinations_tested >= max_combinations:
                                break
                                
                            # Prepare sequence data
                            X_train, y_train = prepare_sequence_data(train_set, n_steps)
                            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                            
                            X_val, y_val = prepare_sequence_data(val_set, n_steps)
                            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                            
                            # Skip if not enough data
                            if len(X_train) < batch_size or len(X_val) == 0:
                                continue
                            
                            # Build model
                            model = create_lstm_model(
                                (n_steps, 1),
                                units=units,
                                layers=layers,
                                dropout_rate=dropout_rate,
                                learning_rate=lr
                            )
                            
                            # Train model
                            try:
                                model.fit(
                                    X_train, y_train,
                                    epochs=50,
                                    batch_size=batch_size,
                                    validation_data=(X_val, y_val),
                                    callbacks=[early_stopping],
                                    verbose=0
                                )
                                
                                # Evaluate on validation set
                                val_pred = model.predict(X_val, verbose=0).ravel()
                                rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                                
                                results.append({
                                    'n_steps': n_steps,
                                    'units': units,
                                    'layers': layers,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'rmse': rmse
                                })
                                
                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_params = {
                                        'n_steps': n_steps,
                                        'units': units,
                                        'layers': layers,
                                        'dropout_rate': dropout_rate,
                                        'learning_rate': lr,
                                        'batch_size': batch_size
                                    }
                                    best_model = model
                                    
                                logger.debug(f"LSTM(n_steps={n_steps}, units={units}, layers={layers}, dropout={dropout_rate}, lr={lr}, batch={batch_size}) - RMSE: {rmse:.4f}")
                                
                                combinations_tested += 1
                            except Exception as e:
                                logger.warning(f"Error training LSTM: {e}")
                                continue
                        
                        if combinations_tested >= max_combinations:
                            break
                    if combinations_tested >= max_combinations:
                        break
                if combinations_tested >= max_combinations:
                    break
            if combinations_tested >= max_combinations:
                break
        if combinations_tested >= max_combinations:
            break
    
    logger.info(f"Best LSTM parameters: {best_params} with RMSE: {best_rmse:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    results_df.to_csv('lstm_grid_search_results.csv', index=False)
    
    # Generate predictions with best model
    if best_model is not None:
        # Retrain on full training data
        X_train_full, y_train_full = prepare_sequence_data(train_scaled, best_params['n_steps'])
        X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1], 1)
        
        best_model.fit(
            X_train_full, y_train_full,
            epochs=100,
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate predictions for test data
        predictions = []
        last_sequence = train_scaled[-best_params['n_steps']:].reshape(1, best_params['n_steps'], 1)
        
        for _ in range(len(test_data)):
            next_value = best_model.predict(last_sequence, verbose=0)
            predictions.append(next_value[0, 0])
            
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_value
        
        # Inverse transform predictions
        best_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    else:
        # Fallback to default parameters if no valid model was found
        logger.warning("No valid LSTM model found during grid search. Using default parameters.")
        best_predictions, _ = run_lstm(train_data, test_data)
    
    return best_params, best_predictions, scaler