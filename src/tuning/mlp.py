import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..model_config import TUNING_SETUP
from ..config import get_dl_overrides
from ..models.mlp import run_mlp
from ..losses import get_keras_loss
from ..preprocessing import get_scaler
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_mlp(X_train, X_test, y_train, y_test, loss='l2', results_dir=None,
                    scaler='standard', feature_range=(0, 1), **kwargs):
    """Grid search for MLP. Returns (best_params, best_predictions).

    Searches: units, dropout, learning_rate, activation.
    Scaler is fixed to the variation's scaler (not searched) so that the
    variation-level scaler choice is respected.
    """
    logger.info(f"Performing grid search for MLP ({loss.upper()}, scaler={scaler}) model...")
    scaler_obj = get_scaler(scaler, feature_range=feature_range)
    if scaler_obj is not None:
        X_train_sc = scaler_obj.fit_transform(X_train)
        X_test_sc = scaler_obj.transform(X_test)
    else:
        X_train_sc = np.asarray(X_train)
        X_test_sc = np.asarray(X_test)

    val_frac = TUNING_SETUP.get("val_frac", 0.2)
    train_idx, val_idx = get_chronological_holdout_indices(len(X_train_sc), val_frac)
    X_tr = X_train_sc[train_idx]
    X_val = X_train_sc[val_idx]
    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    y_tr = y_arr[train_idx]
    y_val = y_arr[val_idx]

    dl = get_dl_overrides()
    epochs_grid = dl.get("epochs_grid", 100)
    epochs_refit = dl.get("epochs_refit", 200)
    patience = dl.get("patience", 10)

    if TUNING_SETUP.get("tuning_fast"):
        units_list = [64]
        dropout_rates = [0.1]
        learning_rates = [0.001]
        activations = ['relu']
    else:
        units_list = [32, 64]
        dropout_rates = [0.1, 0.2]
        learning_rates = [0.001, 0.01]
        activations = ['relu', 'tanh']
    best_rmse = float('inf')
    best_params = None
    results = []
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    combos = list(product(activations, units_list, dropout_rates, learning_rates))

    for (act, units, dr, lr) in tqdm(
        combos, desc=f"MLP ({loss}) grid", unit="candidate", leave=False
    ):
        try:
            model = Sequential([
                Input(shape=(X_tr.shape[1],)),
                Dense(units, activation=act),
                Dropout(dr),
                Dense(units // 2, activation=act),
                Dropout(dr),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=lr), loss=get_keras_loss(loss))
            model.fit(X_tr, y_tr, epochs=epochs_grid, batch_size=32,
                      validation_data=(X_val, y_val),
                      callbacks=[early_stopping], verbose=0)
            val_pred = model.predict(X_val, verbose=0).ravel()
            rmse = score_validation_rmse(y_val, val_pred)
            results.append({'activation': act, 'units': units, 'dropout': dr,
                            'learning_rate': lr, 'rmse': rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'activation': act, 'units': units,
                               'dropout': dr, 'learning_rate': lr}
        except Exception as e:
            logger.warning("Error training MLP: %s", e)
            continue

    logger.info("Best MLP parameters: %s with RMSE: %.4f", best_params, best_rmse)
    if results:
        path = (os.path.join(results_dir, 'mlp_grid_search_results.csv')
                if results_dir else 'mlp_grid_search_results.csv')
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)

    if best_params is not None:
        u = best_params['units']
        act = best_params['activation']
        model = Sequential([
            Input(shape=(X_train_sc.shape[1],)),
            Dense(u, activation=act),
            Dropout(best_params['dropout']),
            Dense(u // 2, activation=act),
            Dropout(best_params['dropout']),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                      loss=get_keras_loss(loss))
        model.fit(X_train_sc, y_train, epochs=epochs_refit, batch_size=32,
                  validation_split=0.2, callbacks=[early_stopping], verbose=0)
        best_predictions = model.predict(X_test_sc, verbose=0).ravel()
    else:
        best_predictions = run_mlp(X_train, X_test, y_train, loss=loss,
                                   scaler=scaler, feature_range=feature_range)
    return best_params, best_predictions
