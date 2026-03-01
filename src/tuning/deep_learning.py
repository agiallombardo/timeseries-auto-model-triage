import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from ..models.deep_learning import (
    prepare_sequence_data, create_rnn_model, create_lstm_model,
    run_rnn, run_lstm, prepare_feature_sequences,
)

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


def grid_search_mlp(X_train, X_test, y_train, y_test):
    """
    Grid search for MLP hyperparameters using the same feature matrices
    as the tree-based models.

    Returns:
    --------
    tuple
        (best_params dict, predictions array)
    """
    logger.info("Performing grid search for MLP model...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    val_size = int(len(X_train_sc) * 0.2)
    X_tr, X_val = X_train_sc[:-val_size], X_train_sc[-val_size:]
    y_tr = y_train.values[:-val_size] if hasattr(y_train, 'values') else y_train[:-val_size]
    y_val = y_train.values[-val_size:] if hasattr(y_train, 'values') else y_train[-val_size:]

    hidden_units_list = [32, 64, 128]
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.01]

    best_rmse = float('inf')
    best_params = None
    results = []

    max_combinations = 18
    combinations_tested = 0

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    for units in hidden_units_list:
        for dr in dropout_rates:
            for lr in learning_rates:
                if combinations_tested >= max_combinations:
                    break

                model = Sequential([
                    Input(shape=(X_tr.shape[1],)),
                    Dense(units, activation='relu'),
                    Dropout(dr),
                    Dense(units // 2, activation='relu'),
                    Dropout(dr),
                    Dense(1),
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

                try:
                    model.fit(
                        X_tr, y_tr,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0,
                    )

                    val_pred = model.predict(X_val, verbose=0).ravel()
                    rmse = np.sqrt(mean_squared_error(y_val, val_pred))

                    results.append({
                        'units': units, 'dropout': dr,
                        'learning_rate': lr, 'rmse': rmse,
                    })

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            'units': units, 'dropout': dr,
                            'learning_rate': lr,
                        }

                    logger.debug(
                        f"MLP(units={units}, dropout={dr}, lr={lr}) "
                        f"- RMSE: {rmse:.4f}"
                    )
                    combinations_tested += 1
                except Exception as e:
                    logger.warning(f"Error training MLP: {e}")
                    continue

            if combinations_tested >= max_combinations:
                break
        if combinations_tested >= max_combinations:
            break

    logger.info(f"Best MLP parameters: {best_params} with RMSE: {best_rmse:.4f}")

    if results:
        pd.DataFrame(results).sort_values('rmse').to_csv(
            'mlp_grid_search_results.csv', index=False
        )

    if best_params is not None:
        u = best_params['units']
        model = Sequential([
            Input(shape=(X_train_sc.shape[1],)),
            Dense(u, activation='relu'),
            Dropout(best_params['dropout']),
            Dense(u // 2, activation='relu'),
            Dropout(best_params['dropout']),
            Dense(1),
        ])
        model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='mse',
        )
        model.fit(
            X_train_sc, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
        best_predictions = model.predict(X_test_sc, verbose=0).ravel()
    else:
        logger.warning("No valid MLP found during grid search. Using defaults.")
        from ..models.deep_learning import run_mlp
        best_predictions = run_mlp(X_train, X_test, y_train)

    return best_params, best_predictions


def grid_search_lstm_features(X_train, X_test, y_train, y_test):
    """
    Grid search for the many-to-one LSTM that uses feature matrices.

    Returns:
    --------
    tuple
        (best_params dict, predictions array)
    """
    logger.info("Performing grid search for LSTM (many-to-one) model...")

    n_steps_list = [3, 5, 7]
    units_list = [32, 64]
    dropout_rates = [0.0, 0.2]
    learning_rates = [0.001, 0.01]

    best_rmse = float('inf')
    best_params = None
    results = []

    max_combinations = 24
    combinations_tested = 0

    for n_steps in n_steps_list:
        X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = prepare_feature_sequences(
            X_train, X_test, y_train, y_test, n_steps
        )
        if len(X_tr_seq) == 0:
            continue

        n_features = X_tr_seq.shape[2]

        seq_scaler = StandardScaler()
        seq_scaler.fit(X_tr_seq.reshape(-1, n_features))
        X_tr_sc = seq_scaler.transform(
            X_tr_seq.reshape(-1, n_features)
        ).reshape(X_tr_seq.shape)

        val_size = max(1, int(len(X_tr_sc) * 0.2))
        X_tr_s, X_val_s = X_tr_sc[:-val_size], X_tr_sc[-val_size:]
        y_tr_s, y_val_s = y_tr_seq[:-val_size], y_tr_seq[-val_size:]

        for units in units_list:
            for dr in dropout_rates:
                for lr in learning_rates:
                    if combinations_tested >= max_combinations:
                        break

                    model = Sequential()
                    model.add(Input(shape=(n_steps, n_features)))
                    model.add(LSTM(units, activation='relu'))
                    if dr > 0:
                        model.add(Dropout(dr))
                    model.add(Dense(1))
                    model.compile(
                        optimizer=Adam(learning_rate=lr), loss='mse'
                    )

                    es = EarlyStopping(
                        monitor='val_loss', patience=10,
                        restore_best_weights=True,
                    )

                    try:
                        model.fit(
                            X_tr_s, y_tr_s,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val_s, y_val_s),
                            callbacks=[es],
                            verbose=0,
                        )

                        val_pred = model.predict(X_val_s, verbose=0).ravel()
                        rmse = np.sqrt(mean_squared_error(y_val_s, val_pred))

                        results.append({
                            'n_steps': n_steps, 'units': units,
                            'dropout': dr, 'learning_rate': lr,
                            'rmse': rmse,
                        })

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'n_steps': n_steps, 'units': units,
                                'dropout': dr, 'learning_rate': lr,
                            }

                        logger.debug(
                            f"LSTM-feat(n_steps={n_steps}, units={units}, "
                            f"dropout={dr}, lr={lr}) - RMSE: {rmse:.4f}"
                        )
                        combinations_tested += 1
                    except Exception as e:
                        logger.warning(f"Error training LSTM-feat: {e}")
                        continue

                if combinations_tested >= max_combinations:
                    break
            if combinations_tested >= max_combinations:
                break
        if combinations_tested >= max_combinations:
            break

    logger.info(
        f"Best LSTM-feat parameters: {best_params} with RMSE: {best_rmse:.4f}"
    )

    if results:
        pd.DataFrame(results).sort_values('rmse').to_csv(
            'lstm_feat_grid_search_results.csv', index=False
        )

    if best_params is not None:
        ns = best_params['n_steps']
        X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
            X_train, X_test, y_train, y_test, ns
        )
        n_features = X_tr_seq.shape[2]

        seq_scaler = StandardScaler()
        seq_scaler.fit(X_tr_seq.reshape(-1, n_features))
        X_tr_sc = seq_scaler.transform(
            X_tr_seq.reshape(-1, n_features)
        ).reshape(X_tr_seq.shape)
        X_te_sc = seq_scaler.transform(
            X_te_seq.reshape(-1, n_features)
        ).reshape(X_te_seq.shape)

        model = Sequential()
        model.add(Input(shape=(ns, n_features)))
        model.add(LSTM(best_params['units'], activation='relu'))
        if best_params['dropout'] > 0:
            model.add(Dropout(best_params['dropout']))
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='mse',
        )

        es = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        model.fit(
            X_tr_sc, y_tr_seq,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[es],
            verbose=0,
        )
        best_predictions = model.predict(X_te_sc, verbose=0).ravel()
    else:
        logger.warning(
            "No valid LSTM-feat found during grid search. Using defaults."
        )
        from ..models.deep_learning import run_lstm_features
        best_predictions = run_lstm_features(X_train, X_test, y_train, y_test)

    return best_params, best_predictions