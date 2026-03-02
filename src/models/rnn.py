import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from ._dl_utils import prepare_sequence_data
from ..losses import get_keras_loss

logger = logging.getLogger(__name__)


def create_rnn_model(input_shape, units=50, learning_rate=0.001, loss='l2'):
    """Build and compile RNN (used by tuning)."""
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units, activation='relu'),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=get_keras_loss(loss),
    )
    return model


def run_rnn(train_data, test_data, n_steps=3, loss='l2'):
    """Run RNN model for time series forecasting (univariate, recursive rollout)."""
    logger.info(f"Training RNN ({loss.upper()}) model...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    X_train, y_train = prepare_sequence_data(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = create_rnn_model(input_shape=(n_steps, 1), loss=loss)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = []
    last_sequence = train_scaled[-n_steps:].reshape(1, n_steps, 1)
    for _ in range(len(test_data)):
        next_value = model.predict(last_sequence, verbose=0)
        next_scalar = float(np.squeeze(next_value))
        predictions.append(next_scalar)
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_scalar

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    return predictions, scaler
