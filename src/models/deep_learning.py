import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
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
    model = Sequential()
    model.add(SimpleRNN(units, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
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
        predictions.append(next_value[0, 0])
        
        # Update sequence with the predicted value
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_value
    
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
    
    # Add first LSTM layer with return_sequences=True if there are multiple layers
    model.add(LSTM(
        units, 
        activation='relu',
        input_shape=input_shape,
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
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
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
        predictions.append(next_value[0, 0])
        
        # Update sequence with the predicted value
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_value
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    
    return predictions, scaler