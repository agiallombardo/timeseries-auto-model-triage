import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from ._dl_utils import prepare_feature_sequences
from ..losses import get_keras_loss

logger = logging.getLogger(__name__)


def run_lstm_features(X_train, X_test, y_train, y_test, n_steps=5, loss='l2'):
    """Many-to-one LSTM on feature matrices; one forward pass per test sample."""
    logger.info(f"Training LSTM-feat ({loss.upper()}) model...")
    X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
        X_train, X_test, y_train, y_test, n_steps
    )
    n_features = X_tr_seq.shape[2]

    scaler = StandardScaler()
    scaler.fit(X_tr_seq.reshape(-1, n_features))
    X_tr_seq = scaler.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
    X_te_seq = scaler.transform(X_te_seq.reshape(-1, n_features)).reshape(X_te_seq.shape)

    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=get_keras_loss(loss))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model.predict(X_te_seq, verbose=0).ravel()
