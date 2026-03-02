import logging
from ..losses import get_linear_model

logger = logging.getLogger(__name__)


def run_linear_regression(X_train, X_test, y_train, loss='l2'):
    """Run Linear Regression baseline with configurable loss."""
    logger.info(f"Training Linear Regression ({loss.upper()}) model...")
    model = get_linear_model(loss)
    model.fit(X_train, y_train)
    return model.predict(X_test)
