import logging
from ..losses import get_linear_model

logger = logging.getLogger(__name__)


def run_linear_regression(X_train, X_test, y_train, loss='l2', alpha=None, **kwargs):
    """Run Linear Regression baseline with configurable loss and optional regularization strength."""
    alpha_info = f", alpha={alpha}" if alpha is not None else ""
    logger.info(f"Training Linear Regression ({loss.upper()}{alpha_info}) model...")
    model = get_linear_model(loss, alpha=alpha)
    model.fit(X_train, y_train)
    return model.predict(X_test)
