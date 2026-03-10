import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ._dl_utils import prepare_feature_sequences
from ..losses import get_keras_loss
from ..preprocessing import get_scaler

logger = logging.getLogger(__name__)


def run_cnn1d(X_train, X_test, y_train, y_test, n_steps=5, loss='l2',
              scaler='standard', feature_range=(0, 1),
              learning_rate=0.001, **kwargs):
    """1D CNN on feature sequences; one forward pass per test sample."""
    logger.info(f"Training CNN-1D ({loss.upper()}, scaler={scaler}) model...")
    X_tr_seq, X_te_seq, y_tr_seq, _ = prepare_feature_sequences(
        X_train, X_test, y_train, y_test, n_steps
    )
    n_features = X_tr_seq.shape[2]

    scaler_obj = get_scaler(scaler, feature_range=feature_range)
    if scaler_obj is not None:
        scaler_obj.fit(X_tr_seq.reshape(-1, n_features))
        X_tr_seq = scaler_obj.transform(X_tr_seq.reshape(-1, n_features)).reshape(X_tr_seq.shape)
        X_te_seq = scaler_obj.transform(X_te_seq.reshape(-1, n_features)).reshape(X_te_seq.shape)

    # Kernel sizes chosen so two Conv1D layers never produce negative length:
    # out1 = n_steps - k1 + 1, out2 = out1 - k2 + 1 >= 1
    k1 = min(2, n_steps)
    out1 = n_steps - k1 + 1
    k2 = min(2, out1)
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Conv1D(64, k1, activation='relu'),
        Conv1D(32, k2, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=get_keras_loss(loss))
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
