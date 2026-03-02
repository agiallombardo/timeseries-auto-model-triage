"""
Shared sequence-building utilities for RNN, LSTM, LSTM-feat, RNN-feat, and CNN-1D.
"""

import numpy as np


def prepare_sequence_data(data, n_steps=3):
    """
    Prepare sequences for univariate RNN/LSTM (raw target series only).

    Parameters:
    -----------
    data : array-like
        Time series data (e.g. scaled target)
    n_steps : int
        Number of time steps for input sequence

    Returns:
    --------
    tuple
        X (input sequences), y (target values)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def prepare_feature_sequences(X_train, X_test, y_train, y_test, n_steps):
    """
    Build 3-D input sequences from 2-D feature matrices (lags, rolling, etc.)
    for LSTM-feat, RNN-feat, CNN-1D.

    For index i the input is X[i-n_steps:i], target y[i]. Train/test boundary
    is handled so the first test sequence uses the last n_steps training rows.

    Returns:
    --------
    tuple
        (X_train_seq, X_test_seq, y_train_seq, y_test_seq)
    """
    X_all = np.vstack([
        X_train.values if hasattr(X_train, 'values') else X_train,
        X_test.values if hasattr(X_test, 'values') else X_test,
    ])
    y_all = np.concatenate([
        y_train.values if hasattr(y_train, 'values') else y_train,
        y_test.values if hasattr(y_test, 'values') else y_test,
    ])
    train_end = len(X_train)
    X_seq, y_seq, is_train = [], [], []
    for i in range(n_steps, len(X_all)):
        X_seq.append(X_all[i - n_steps:i])
        y_seq.append(y_all[i])
        is_train.append(i < train_end)
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    is_train = np.array(is_train)
    return (
        X_seq[is_train], X_seq[~is_train],
        y_seq[is_train], y_seq[~is_train],
    )
