import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def prepare_sequence_data(data, n_steps=3):
    """
    Prepare sequences for RNN/LSTM models.
    
    Parameters:
    -----------
    data : array-like
        Time series data
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

def create_rnn_model(input_shape, units=50, learning_rate=0.001):
    """
    Create a RNN model with given parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    units : int
        Number of RNN units
    learning_rate : float
        Learning rate for Adam optimizer
        
    Returns:
    --------
    keras.models.Sequential
        Compiled RNN model
    """
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def run_rnn(train_data, test_data, n_steps=3):
    """
    Run RNN model for time series forecasting.
    
    Parameters:
    -----------
    train_data : Series
        Training data
    test_data : Series
        Test data
    n_steps : int
        Number of time steps for input sequence
        
    Returns:
    --------
    tuple
        (predictions, scaler)
    """
    logger.info("Training RNN model...")
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Prepare sequence data
    X_train, y_train = prepare_sequence_data(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build RNN model using the helper function
    model = create_rnn_model(input_shape=(n_steps, 1))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Generate predictions for test data
    predictions = []
    last_sequence = train_scaled[-n_steps:].reshape(1, n_steps, 1)
    
    for _ in range(len(test_data)):
        # Predict next value
        next_value = model.predict(last_sequence, verbose=0)
        next_scalar = float(np.squeeze(next_value))
        predictions.append(next_scalar)
        
        # Update sequence with the predicted value (must assign scalar, not array)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_scalar
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    
    return predictions, scaler


def create_lstm_model(input_shape, units=50, layers=1, dropout_rate=0.0, learning_rate=0.001):
    """
    Create an LSTM model with given parameters.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_steps, features)
    units : int
        Number of LSTM units
    layers : int
        Number of LSTM layers
    dropout_rate : float
        Dropout rate after each LSTM layer
    learning_rate : float
        Learning rate for Adam optimizer
        
    Returns:
    --------
    keras.models.Sequential
        Compiled LSTM model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    # Add first LSTM layer with return_sequences=True if there are multiple layers
    model.add(LSTM(
        units, 
        activation='relu',
        return_sequences=(layers > 1)
    ))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Add additional LSTM layers if requested
    for i in range(1, layers):
        model.add(LSTM(
            units,
            activation='relu',
            return_sequences=(i < layers - 1)
        ))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

def run_lstm(train_data, test_data, n_steps=3):
    """Run LSTM model."""
    logger.info("Training LSTM model...")
    
    # Normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Prepare sequence data
    X_train, y_train = prepare_sequence_data(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Build LSTM model
    model = Sequential([
        Input(shape=(n_steps, 1)),
        LSTM(50, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Generate predictions for test data
    predictions = []
    last_sequence = train_scaled[-n_steps:].reshape(1, n_steps, 1)
    
    for _ in range(len(test_data)):
        # Predict next value
        next_value = model.predict(last_sequence, verbose=0)
        next_scalar = float(np.squeeze(next_value))
        predictions.append(next_scalar)
        
        # Update sequence with the predicted value (must assign scalar, not array)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_scalar
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    
    return predictions, scaler


def run_mlp(X_train, X_test, y_train):
    """
    Run MLP model on the same lag/rolling features used by tree-based models.

    Parameters:
    -----------
    X_train : DataFrame
        Training features (lags, rolling stats, calendar, etc.)
    X_test : DataFrame
        Test features
    y_train : Series
        Training target values

    Returns:
    --------
    np.ndarray
        Predictions for the test set
    """
    logger.info("Training MLP model...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_train_sc.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_sc, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = model.predict(X_test_sc, verbose=0).ravel()
    return predictions


def prepare_feature_sequences(X_train, X_test, y_train, y_test, n_steps):
    """
    Build 3-D input sequences from the 2-D feature matrices so that an LSTM
    can consume the same lag/rolling features used by tree-based models.

    For time index *i* the input is X[i-n_steps : i] (shape ``(n_steps, n_features)``)
    and the target is y[i].  Sequences that straddle the train/test boundary
    are handled correctly: the first test sequence uses the last ``n_steps``
    training rows as context.

    Returns:
    --------
    tuple
        (X_train_seq, X_test_seq, y_train_seq, y_test_seq)
    """
    X_all = np.vstack([X_train.values if hasattr(X_train, 'values') else X_train,
                       X_test.values if hasattr(X_test, 'values') else X_test])
    y_all = np.concatenate([y_train.values if hasattr(y_train, 'values') else y_train,
                            y_test.values if hasattr(y_test, 'values') else y_test])

    train_end = len(X_train)

    X_seq, y_seq, is_train = [], [], []
    for i in range(n_steps, len(X_all)):
        X_seq.append(X_all[i - n_steps:i])
        y_seq.append(y_all[i])
        is_train.append(i < train_end)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    is_train = np.array(is_train)

    return (X_seq[is_train], X_seq[~is_train],
            y_seq[is_train], y_seq[~is_train])


def run_lstm_features(X_train, X_test, y_train, y_test, n_steps=5):
    """
    Many-to-one LSTM that operates on the same feature matrices as the
    tree-based models.  One forward pass per test sample (no recursive
    rollout).

    Parameters:
    -----------
    X_train, X_test : DataFrame
        Feature matrices (lags, rolling stats, calendar, etc.)
    y_train, y_test : Series
        Target values
    n_steps : int
        Number of consecutive feature rows fed to the LSTM

    Returns:
    --------
    np.ndarray
        Predictions aligned with y_test
    """
    logger.info("Training LSTM (many-to-one) model...")

    X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
        X_train, X_test, y_train, y_test, n_steps
    )

    n_features = X_tr_seq.shape[2]

    scaler = StandardScaler()
    X_tr_flat = X_tr_seq.reshape(-1, n_features)
    scaler.fit(X_tr_flat)

    X_tr_seq = scaler.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
    X_te_seq = scaler.transform(X_te_seq.reshape(-1, n_features)).reshape(X_te_seq.shape)

    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = model.predict(X_te_seq, verbose=0).ravel()
    return predictions