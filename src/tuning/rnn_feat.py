import logging
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ..model_config import TUNING_SETUP
from ..models._dl_utils import prepare_feature_sequences
from ..models.rnn_feat import run_rnn_features
from ..losses import get_keras_loss
from ._validation import get_chronological_holdout_indices, score_validation_rmse

logger = logging.getLogger(__name__)


def grid_search_rnn_features(X_train, X_test, y_train, y_test, loss='l2', results_dir=None, **kwargs):
    """Grid search for RNN-feat. Returns (best_params, best_predictions)."""
    logger.info(f"Performing grid search for RNN-feat ({loss.upper()}) model...")
    n_steps_list = [3, 5, 7]
    units_list = [32, 64]
    dropout_rates = [0.0, 0.2]
    learning_rates = [0.001, 0.01]

    best_rmse = float('inf')
    best_params = None
    results = []
    max_combinations = 20
    combinations_tested = 0

    for n_steps in n_steps_list:
        X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(X_train, X_test, y_train, y_test, n_steps)
        if len(X_tr_seq) == 0:
            continue
        n_features = X_tr_seq.shape[2]
        seq_scaler = StandardScaler()
        seq_scaler.fit(X_tr_seq.reshape(-1, n_features))
        X_tr_sc = seq_scaler.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
        val_frac = TUNING_SETUP.get("val_frac", 0.2)
        train_idx, val_idx = get_chronological_holdout_indices(len(X_tr_sc), val_frac)
        X_tr_s, X_val_s = X_tr_sc[train_idx], X_tr_sc[val_idx]
        y_tr_s, y_val_s = y_tr_seq[train_idx], y_tr_seq[val_idx]

        for units in units_list:
            for dr in dropout_rates:
                for lr in learning_rates:
                    if combinations_tested >= max_combinations:
                        break
                    try:
                        model = Sequential()
                        model.add(Input(shape=(n_steps, n_features)))
                        model.add(SimpleRNN(units, activation='relu'))
                        if dr > 0:
                            model.add(Dropout(dr))
                        model.add(Dense(1))
                        model.compile(optimizer=Adam(learning_rate=lr), loss=get_keras_loss(loss))
                        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        model.fit(X_tr_s, y_tr_s, epochs=50, batch_size=32, validation_data=(X_val_s, y_val_s), callbacks=[es], verbose=0)
                        val_pred = model.predict(X_val_s, verbose=0).ravel()
                        rmse = score_validation_rmse(y_val_s, val_pred)
                        results.append({'n_steps': n_steps, 'units': units, 'dropout': dr, 'learning_rate': lr, 'rmse': rmse})
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'n_steps': n_steps, 'units': units, 'dropout': dr, 'learning_rate': lr}
                        combinations_tested += 1
                    except Exception as e:
                        logger.warning(f"Error training RNN-feat: {e}")
                        continue
            if combinations_tested >= max_combinations:
                break
        if combinations_tested >= max_combinations:
            break

    logger.info(f"Best RNN-feat parameters: {best_params} with RMSE: {best_rmse:.4f}")
    if results:
        path = os.path.join(results_dir, 'rnn_feat_grid_search_results.csv') if results_dir else 'rnn_feat_grid_search_results.csv'
        pd.DataFrame(results).sort_values('rmse').to_csv(path, index=False)

    if best_params is not None:
        ns = best_params['n_steps']
        X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(X_train, X_test, y_train, y_test, ns)
        n_features = X_tr_seq.shape[2]
        seq_scaler = StandardScaler()
        seq_scaler.fit(X_tr_seq.reshape(-1, n_features))
        X_tr_sc = seq_scaler.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
        X_te_sc = seq_scaler.transform(X_te_seq.reshape(-1, n_features)).reshape(X_te_seq.shape)
        model = Sequential()
        model.add(Input(shape=(ns, n_features)))
        model.add(SimpleRNN(best_params['units'], activation='relu'))
        if best_params['dropout'] > 0:
            model.add(Dropout(best_params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=get_keras_loss(loss))
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_tr_sc, y_tr_seq, epochs=200, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
        best_predictions = model.predict(X_te_sc, verbose=0).ravel()
    else:
        best_predictions = run_rnn_features(X_train, X_test, y_train, y_test, loss=loss)
    return best_params, best_predictions
