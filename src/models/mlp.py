import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from ..losses import get_keras_loss

logger = logging.getLogger(__name__)


def run_mlp(X_train, X_test, y_train, loss='l2'):
    """Run MLP on the same lag/rolling features as tree-based models."""
    logger.info(f"Training MLP ({loss.upper()}) model...")
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss=get_keras_loss(loss))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train_sc, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model.predict(X_test_sc, verbose=0).ravel()
