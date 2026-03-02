import logging
from sklearn.ensemble import RandomForestRegressor

from ..losses import get_rf_criterion

logger = logging.getLogger(__name__)


def run_random_forest(X_train, X_test, y_train, loss='l2'):
    """Run Random Forest model."""
    logger.info(f"Training Random Forest ({loss.upper()}) model...")
    criterion = get_rf_criterion(loss)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        criterion=criterion,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
