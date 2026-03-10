import logging
import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..model_config import TUNING_SETUP
from ..models._dl_utils import prepare_feature_sequences
from ..models.cnn1d import run_cnn1d
from ..losses import get_keras_loss
from ..preprocessing import get_scaler
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_cnn1d(X_train, X_test, y_train, y_test, loss='l2', results_dir=None,
                      scaler='standard', feature_range=(0, 1), **kwargs):
    """Grid search for CNN-1D. Returns (best_params, best_predictions).

    Scaler is inherited from the calling variation so it matches the non-tuned path.
    Searches: n_steps, filters, kernel_size, learning_rate.
    """
    logger.info(f"Performing grid search for CNN-1D ({loss.upper()}, scaler={scaler}) model...")
    n_steps_list = [3, 5, 7]
    filters_list = [32, 64]
    kernel_sizes = [2, 3]
    learning_rates = [0.001, 0.01]

    best_rmse = float('inf')
    best_params = None
    results = []
    max_combinations = 16
    combos = []
    for n_steps in n_steps_list:
        for combo in product(filters_list, kernel_sizes, learning_rates):
            combos.append((n_steps,) + combo)
            if len(combos) >= max_combinations:
                break
        if len(combos) >= max_combinations:
            break
    combos = combos[:max_combinations]
    n_steps_cache = {}

    for (n_steps, f1, ks, lr) in tqdm(
        combos, desc=f"CNN-1D ({loss}) grid", unit="candidate", leave=False
    ):
        if n_steps not in n_steps_cache:
            X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
                X_train, X_test, y_train, y_test, n_steps)
            if len(X_tr_seq) == 0:
                n_steps_cache[n_steps] = None
                continue
            n_features = X_tr_seq.shape[2]
            scaler_obj = get_scaler(scaler, feature_range=feature_range)
            if scaler_obj is not None:
                scaler_obj.fit(X_tr_seq.reshape(-1, n_features))
                X_tr_sc = scaler_obj.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
            else:
                X_tr_sc = X_tr_seq.copy()
            val_frac = TUNING_SETUP.get("val_frac", 0.2)
            train_idx, val_idx = get_chronological_holdout_indices(len(X_tr_sc), val_frac)
            n_steps_cache[n_steps] = (
                X_tr_sc[train_idx], X_tr_sc[val_idx],
                y_tr_seq[train_idx], y_tr_seq[val_idx],
                X_tr_seq.shape[2],
            )
        cached = n_steps_cache.get(n_steps)
        if cached is None:
            continue
        X_tr_s, X_val_s, y_tr_s, y_val_s = cached[:4]
        n_features = cached[4]
        try:
            k1 = min(ks, n_steps)
            out1 = n_steps - k1 + 1
            k2 = min(ks, out1)
            if out1 < 1 or k2 < 1:
                continue
            model = Sequential([
                Input(shape=(n_steps, n_features)),
                Conv1D(f1, k1, activation='relu'),
                Conv1D(f1 // 2, k2, activation='relu'),
                GlobalAveragePooling1D(),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=lr), loss=get_keras_loss(loss))
            es = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
            model.fit(X_tr_s, y_tr_s, epochs=50, batch_size=32,
                      validation_data=(X_val_s, y_val_s),
                      callbacks=[es], verbose=0)
            val_pred = model.predict(X_val_s, verbose=0).ravel()
            rmse = score_validation_rmse(y_val_s, val_pred)
            results.append({'n_steps': n_steps, 'filters': f1, 'kernel_size': ks,
                            'learning_rate': lr, 'rmse': rmse})
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'n_steps': n_steps, 'filters': f1,
                               'kernel_size': ks, 'learning_rate': lr}
        except Exception as e:
            logger.warning("Error training CNN-1D: %s", e)
            continue

    logger.info("Best CNN-1D parameters: %s with RMSE: %.4f", best_params, best_rmse)
    if results:
        path = (os.path.join(results_dir, 'cnn1d_grid_search_results.csv')
                if results_dir else 'cnn1d_grid_search_results.csv')
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)

    if best_params is not None:
        ns = best_params['n_steps']
        f1, ks = best_params['filters'], best_params['kernel_size']
        X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
            X_train, X_test, y_train, y_test, ns)
        n_features = X_tr_seq.shape[2]
        scaler_obj = get_scaler(scaler, feature_range=feature_range)
        if scaler_obj is not None:
            scaler_obj.fit(X_tr_seq.reshape(-1, n_features))
            X_tr_sc = scaler_obj.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
            X_te_sc = scaler_obj.transform(X_te_seq.reshape(-1, n_features)).reshape(X_te_seq.shape)
        else:
            X_tr_sc, X_te_sc = X_tr_seq.copy(), X_te_seq.copy()
        k1 = min(ks, ns)
        out1 = ns - k1 + 1
        k2 = min(ks, out1)
        model = Sequential([
            Input(shape=(ns, n_features)),
            Conv1D(f1, k1, activation='relu'),
            Conv1D(f1 // 2, k2, activation='relu'),
            GlobalAveragePooling1D(),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                      loss=get_keras_loss(loss))
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_tr_sc, y_tr_seq, epochs=200, batch_size=32,
                  validation_split=0.2, callbacks=[es], verbose=0)
        best_predictions = model.predict(X_te_sc, verbose=0).ravel()
    else:
        best_predictions = run_cnn1d(X_train, X_test, y_train, y_test, loss=loss,
                                     scaler=scaler, feature_range=feature_range)
    return best_params, best_predictions
