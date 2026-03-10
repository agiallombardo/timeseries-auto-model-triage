import itertools
import logging
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..models.sarima import run_sarima
from ._validation import (
    get_chronological_holdout_indices,
    score_validation_rmse,
    write_tuning_artifacts,
    DEFAULT_VAL_FRAC,
)

logger = logging.getLogger(__name__)


def grid_search_sarima(train_data, test_data, seasonal_periods=12, results_dir=None, val_frac=None, **kwargs):
    """
    Grid search for SARIMA using train-only validation (out-of-sample RMSE).
    Refits best model on full training data before producing test predictions.
    Returns (best_params, best_predictions).
    """
    logger.info("Performing grid search for SARIMA model (validation RMSE)...")
    val_frac = val_frac or DEFAULT_VAL_FRAC
    train_arr = np.asarray(train_data).ravel()
    n = len(train_arr)
    train_idx, val_idx = get_chronological_holdout_indices(n, val_frac)
    train_part = train_arr[train_idx]
    val_part = train_arr[val_idx]
    n_val = len(val_part)

    p_values, d_values, q_values = range(0, 3), range(0, 2), range(0, 3)
    P_values, D_values, Q_values = range(0, 2), range(0, 2), range(0, 2)
    pdq = list(itertools.product(p_values, d_values, q_values))
    seasonal_pdq = list(itertools.product(P_values, D_values, Q_values, [seasonal_periods]))

    best_rmse = float("inf")
    best_order, best_seasonal_order = None, None
    results = []
    max_combinations = 20
    combinations_tested = 0

    for param in pdq:
        for seasonal_param in seasonal_pdq:
            if combinations_tested >= max_combinations:
                break
            try:
                model = SARIMAX(train_part, order=param, seasonal_order=seasonal_param)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=n_val)
                rmse = score_validation_rmse(val_part, forecast)
                results.append({
                    "order": param,
                    "seasonal_order": seasonal_param,
                    "validation_rmse": rmse,
                    "params": {"order": list(param), "seasonal_order": list(seasonal_param)},
                })
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = param
                    best_seasonal_order = seasonal_param
                logger.debug(f"SARIMA{param}x{seasonal_param} - validation RMSE: {rmse:.4f}")
                combinations_tested += 1
            except Exception:
                continue
        if combinations_tested >= max_combinations:
            break

    logger.info(f"Best SARIMA order: {best_order}x{best_seasonal_order} with validation RMSE: {best_rmse:.4f}")
    best_predictions = run_sarima(
        train_data, test_data,
        order=best_order,
        seasonal_order=best_seasonal_order,
    )

    if results:
        path = os.path.join(results_dir, "sarima_grid_search_results.csv") if results_dir else "sarima_grid_search_results.csv"
        pd.DataFrame(results).sort_values("validation_rmse").to_csv(path, index=False)
        write_tuning_artifacts(
            results_dir,
            "sarima",
            [{"params": r["params"], "validation_rmse": r["validation_rmse"], "aggregate_score": r["validation_rmse"]} for r in results],
            {"order": list(best_order), "seasonal_order": list(best_seasonal_order)},
            {"aggregate_score": best_rmse, "metric": "rmse", "fold_scores": [best_rmse]},
            filename_suffix="tuning_results",
        )
    best_params = {
        "order": list(best_order),
        "seasonal_order": list(best_seasonal_order),
    }
    return best_params, best_predictions
