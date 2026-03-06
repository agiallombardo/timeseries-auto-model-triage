import logging
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_svr(X_train, X_test, y_train, **kwargs):
    """Run Support Vector Regression model. kwargs are passed to SVR() (defaults: kernel='rbf', C=100, gamma=0.1, epsilon=0.1)."""
    logger.info("Training SVR model...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_X.transform(X_test)
    defaults = {"kernel": "rbf", "C": 100, "gamma": 0.1, "epsilon": 0.1}
    defaults.update(kwargs)
    model = SVR(**defaults)
    model.fit(X_train_scaled, y_train_scaled)
    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
    return predictions, (scaler_X, scaler_y)
