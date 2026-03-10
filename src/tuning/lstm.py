import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ..model_config import TUNING_SETUP
from ..models._dl_utils import prepare_sequence_data
from ..models.lstm import create_lstm_model, run_lstm
from ..losses import get_keras_loss
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_lstm(train_data, test_data, loss='l2', results_dir=None, **kwargs):
    """Grid search for LSTM. Returns (best_params, best_predictions, scaler)."""
    logger.info(f"Performing grid search for LSTM ({loss.upper()}) model...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    val_frac = TUNING_SETUP.get("val_frac", 0.2)
    train_idx, val_idx = get_chronological_holdout_indices(len(train_scaled), val_frac)
    train_set = train_scaled[train_idx]
    val_set = train_scaled[val_idx]

    n_steps_list = [3, 5]
    units_list = [32, 50]
    layers_list = [1]
    dropout_rates = [0.0, 0.2]
    learning_rates = [0.001, 0.01]
    batch_sizes = [16, 32]

    best_rmse = float('inf')
    best_params = None
    best_model = None
    results = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    max_combinations = 16
    combos = list(product(
        n_steps_list, units_list, layers_list,
        dropout_rates, learning_rates, batch_sizes
    ))[:max_combinations]

    for (n_steps, units, layers, dropout_rate, lr, batch_size) in tqdm(
        combos, desc=f"LSTM ({loss}) grid", unit="candidate", leave=False
    ):
        X_train, y_train = prepare_sequence_data(train_set, n_steps)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val, y_val = prepare_sequence_data(val_set, n_steps)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        if len(X_train) < 16 or len(X_val) == 0:
            continue
        try:
            model = create_lstm_model(
                (n_steps, 1), units=units, layers=layers,
                dropout_rate=dropout_rate, learning_rate=lr, loss=loss,
            )
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0,
            )
            val_pred = model.predict(X_val, verbose=0).ravel()
            rmse = score_validation_rmse(y_val, val_pred)
            results.append({
                'n_steps': n_steps, 'units': units, 'layers': layers,
                'dropout_rate': dropout_rate, 'learning_rate': lr,
                'batch_size': batch_size, 'rmse': rmse,
            })
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    'n_steps': n_steps, 'units': units, 'layers': layers,
                    'dropout_rate': dropout_rate, 'learning_rate': lr,
                    'batch_size': batch_size,
                }
                best_model = model
        except Exception as e:
            logger.warning(f"Error training LSTM: {e}")
            continue

    logger.info(f"Best LSTM parameters: {best_params} with RMSE: {best_rmse:.4f}")
    if results:
        path = os.path.join(results_dir, 'lstm_grid_search_results.csv') if results_dir else 'lstm_grid_search_results.csv'
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)

    if best_model is not None and best_params is not None:
        X_train_full, y_train_full = prepare_sequence_data(train_scaled, best_params['n_steps'])
        X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1], 1)
        best_model.fit(X_train_full, y_train_full, epochs=100, batch_size=best_params['batch_size'], callbacks=[early_stopping], verbose=0)
        predictions = []
        last_sequence = train_scaled[-best_params['n_steps']:].reshape(1, best_params['n_steps'], 1)
        for _ in range(len(test_data)):
            next_out = best_model.predict(last_sequence, verbose=0)
            next_val = float(np.asarray(next_out).ravel()[0])
            predictions.append(next_val)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_val
        best_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    else:
        best_predictions, _ = run_lstm(train_data, test_data, loss=loss)
    return best_params, best_predictions, scaler
