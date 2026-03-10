import itertools
import logging
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..models.arima import run_arima
from ._validation import (
    get_chronological_holdout_indices,
    score_validation_rmse,
    write_tuning_artifacts,
    DEFAULT_VAL_FRAC,
)

logger = logging.getLogger(__name__)


def grid_search_arima(train_data, test_data, results_dir=None, val_frac=None, **kwargs):
    """
    Grid search for ARIMA order using train-only validation (out-of-sample RMSE).
    Refits best model on full training data before producing test predictions.
    Returns (best_params, best_predictions).
    """
    logger.info("Performing grid search for ARIMA model (validation RMSE)...")
    val_frac = val_frac or DEFAULT_VAL_FRAC
    train_arr = np.asarray(train_data).ravel()
    n = len(train_arr)
    train_idx, val_idx = get_chronological_holdout_indices(n, val_frac)
    train_part = train_arr[train_idx]
    val_part = train_arr[val_idx]
    n_val = len(val_part)

    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 5)
    pdq = list(itertools.product(p_values, d_values, q_values))

    best_rmse = float("inf")
    best_order = None
    results = []

    for param in pdq:
        try:
            model = ARIMA(train_part, order=param)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=n_val)
            rmse = score_validation_rmse(val_part, forecast)
            results.append({"order": param, "validation_rmse": rmse, "params": {"order": list(param)}})
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = param
            logger.debug(f"ARIMA{param} - validation RMSE: {rmse:.4f}")
        except Exception:
            continue

    logger.info(f"Best ARIMA order: {best_order} with validation RMSE: {best_rmse:.4f}")
    best_predictions = run_arima(train_data, test_data, order=best_order)

    if results:
        path = os.path.join(results_dir, "arima_grid_search_results.csv") if results_dir else "arima_grid_search_results.csv"
        pd.DataFrame([{**r, "aggregate_score": r["validation_rmse"]} for r in results]).sort_values("validation_rmse").to_csv(path, index=False)
        write_tuning_artifacts(
            results_dir,
            "arima",
            [{"params": r["params"], "validation_rmse": r["validation_rmse"], "aggregate_score": r["validation_rmse"]} for r in results],
            {"order": list(best_order)},
            {"aggregate_score": best_rmse, "metric": "rmse", "fold_scores": [best_rmse]},
            filename_suffix="tuning_results",
        )
    return {"order": list(best_order)}, best_predictions
