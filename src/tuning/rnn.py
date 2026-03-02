import logging
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ..models._dl_utils import prepare_sequence_data
from ..models.rnn import create_rnn_model, run_rnn
from ..losses import get_keras_loss

logger = logging.getLogger(__name__)


def grid_search_rnn(train_data, test_data, loss='l2'):
    """Grid search for RNN. Returns (best_params, best_predictions, scaler)."""
    logger.info(f"Performing grid search for RNN ({loss.upper()}) model...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    val_size = int(len(train_scaled) * 0.2)
    train_set = train_scaled[:-val_size]
    val_set = train_scaled[-val_size:]

    n_steps_list = [3, 5, 7]
    units_list = [32, 50]
    learning_rates = [0.001, 0.01]
    batch_sizes = [16, 32]

    best_rmse = float('inf')
    best_params = None
    best_model = None
    results = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for n_steps in n_steps_list:
        X_train, y_train = prepare_sequence_data(train_set, n_steps)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val, y_val = prepare_sequence_data(val_set, n_steps)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        if len(X_train) < 16 or len(X_val) == 0:
            continue
        for units in units_list:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    try:
                        model = create_rnn_model((n_steps, 1), units=units, learning_rate=lr, loss=loss)
                        model.fit(
                            X_train, y_train,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=0,
                        )
                        val_pred = model.predict(X_val, verbose=0).ravel()
                        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                        results.append({'n_steps': n_steps, 'units': units, 'learning_rate': lr, 'batch_size': batch_size, 'rmse': rmse})
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'n_steps': n_steps, 'units': units, 'learning_rate': lr, 'batch_size': batch_size}
                            best_model = model
                    except Exception as e:
                        logger.warning(f"Error training RNN: {e}")
                        continue

    logger.info(f"Best RNN parameters: {best_params} with RMSE: {best_rmse:.4f}")
    if results:
        pd.DataFrame(results).sort_values('rmse').to_csv('rnn_grid_search_results.csv', index=False)

    if best_model is not None and best_params is not None:
        X_train_full, y_train_full = prepare_sequence_data(train_scaled, best_params['n_steps'])
        X_train_full = X_train_full.reshape(X_train_full.shape[0], X_train_full.shape[1], 1)
        best_model.fit(X_train_full, y_train_full, epochs=100, batch_size=best_params['batch_size'], callbacks=[early_stopping], verbose=0)
        predictions = []
        last_sequence = train_scaled[-best_params['n_steps']:].reshape(1, best_params['n_steps'], 1)
        for _ in range(len(test_data)):
            next_value = best_model.predict(last_sequence, verbose=0)
            predictions.append(next_value[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_value
        best_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    else:
        best_predictions, _ = run_rnn(train_data, test_data, loss=loss)
    return best_params, best_predictions, scaler
