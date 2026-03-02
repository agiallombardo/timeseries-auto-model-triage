import logging
from xgboost import XGBRegressor

from ..losses import get_xgb_params

logger = logging.getLogger(__name__)


def run_xgboost(X_train, X_test, y_train, loss='l2'):
    """Run XGBoost model."""
    logger.info(f"Training XGBoost ({loss.upper()}) model...")
    xgb_extra = get_xgb_params(loss)
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        **xgb_extra,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
