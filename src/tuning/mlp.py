import logging
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ..models.mlp import run_mlp
from ..losses import get_keras_loss

logger = logging.getLogger(__name__)


def grid_search_mlp(X_train, X_test, y_train, y_test, loss='l2', results_dir=None, **kwargs):
    """Grid search for MLP. Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for MLP ({loss.upper()}) model...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    val_size = int(len(X_train_sc) * 0.2)
    X_tr, X_val = X_train_sc[:-val_size], X_train_sc[-val_size:]
    y_tr = y_train.values[:-val_size] if hasattr(y_train, 'values') else y_train[:-val_size]
    y_val = y_train.values[-val_size:] if hasattr(y_train, 'values') else y_train[-val_size:]

    units_list = [32, 64]
    dropout_rates = [0.1, 0.2]
    learning_rates = [0.001, 0.01]
    best_rmse = float('inf')
    best_params = None
    results = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for units in units_list:
        for dr in dropout_rates:
            for lr in learning_rates:
                try:
                    model = Sequential([
                        Input(shape=(X_tr.shape[1],)),
                        Dense(units, activation='relu'),
                        Dropout(dr),
                        Dense(units // 2, activation='relu'),
                        Dropout(dr),
                        Dense(1),
                    ])
                    model.compile(optimizer=Adam(learning_rate=lr), loss=get_keras_loss(loss))
                    model.fit(X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
                    val_pred = model.predict(X_val, verbose=0).ravel()
                    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                    results.append({'units': units, 'dropout': dr, 'learning_rate': lr, 'rmse': rmse})
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {'units': units, 'dropout': dr, 'learning_rate': lr}
                except Exception as e:
                    logger.warning(f"Error training MLP: {e}")
                    continue

    logger.info(f"Best MLP parameters: {best_params} with RMSE: {best_rmse:.4f}")
    if results:
        path = os.path.join(results_dir, 'mlp_grid_search_results.csv') if results_dir else 'mlp_grid_search_results.csv'
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)

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
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=get_keras_loss(loss))
        model.fit(X_train_sc, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        best_predictions = model.predict(X_test_sc, verbose=0).ravel()
    else:
        best_predictions = run_mlp(X_train, X_test, y_train, loss=loss)
    return best_params, best_predictions
