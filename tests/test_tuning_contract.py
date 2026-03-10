"""
Model-level and regression tests: tuners use train-only validation,
return consistent shapes, and test metrics are produced after refit on full train.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.tuning.arima import grid_search_arima
from src.tuning.moving_average import grid_search_moving_average
from src.tuning.linear_regression import grid_search_linear_regression


class TestArimaTuner:
    """ARIMA: selection by validation RMSE; refit on full train; test predictions only for test period."""

    def test_arima_returns_best_params_and_predictions_for_test_only(self):
        np.random.seed(42)
        n_train, n_test = 60, 20
        train = np.cumsum(np.random.randn(n_train)) + 100
        test = np.cumsum(np.random.randn(n_test)) + train[-1]
        best_params, predictions = grid_search_arima(train, test)
        assert isinstance(best_params, dict)
        assert "order" in best_params
        assert len(predictions) == n_test
        assert not np.any(np.isnan(predictions))

    def test_arima_uses_train_only_for_selection(self):
        # If test were used, we could detect by passing different test and checking selection changed.
        # Here we only check that with fixed train, we get deterministic best_params (same seed).
        np.random.seed(123)
        train = np.cumsum(np.random.randn(50)) + 100
        test_a = np.ones(10)
        test_b = np.ones(10) * 2
        params_a, _ = grid_search_arima(train, test_a)
        params_b, _ = grid_search_arima(train, test_b)
        # Selection is on train-only validation; same train -> same best order
        assert params_a["order"] == params_b["order"]


class TestMovingAverageTuner:
    """MA: train-only holdout; refit on full train; test predictions for test period."""

    def test_ma_returns_best_params_and_predictions(self):
        np.random.seed(42)
        n_train, n_test = 50, 15
        train = pd.Series(np.cumsum(np.random.randn(n_train)) + 100)
        test = pd.Series(np.cumsum(np.random.randn(n_test)) + float(train.iloc[-1]))
        best_params, predictions = grid_search_moving_average(train, test)
        assert isinstance(best_params, dict)
        assert "window" in best_params
        assert len(predictions) == n_test


class TestLinearRegressionTuner:
    """LR: TimeSeriesSplit on train only; refit on full train; test predictions."""

    def test_lr_returns_best_params_and_predictions(self):
        np.random.seed(42)
        n_train, n_test = 50, 15
        X_train = pd.DataFrame({"x": np.arange(n_train)}, index=range(n_train))
        y_train = pd.Series(np.cumsum(np.random.randn(n_train)) + 100, index=range(n_train))
        X_test = pd.DataFrame({"x": np.arange(n_test)}, index=range(n_train, n_train + n_test))
        y_test = pd.Series(np.zeros(n_test), index=range(n_train, n_train + n_test))
        best_params, predictions = grid_search_linear_regression(
            X_train, X_test, y_train, y_test, loss="l2"
        )
        assert isinstance(best_params, dict)
        assert len(predictions) == n_test


class TestResultSummaryContract:
    """Chosen params come from validation; result artifacts have consistent fields."""

    def test_validation_summary_json_when_tuning_used(self):
        # ARIMA tuner writes validation summary when results_dir is set
        with tempfile.TemporaryDirectory() as tmp:
            np.random.seed(42)
            train = np.cumsum(np.random.randn(40)) + 100
            test = np.ones(10)
            grid_search_arima(train, test, results_dir=tmp)
            path = os.path.join(tmp, "arima_validation_summary.json")
            assert os.path.isfile(path)
            with open(path) as f:
                summary = json.load(f)
            assert "best_params" in summary
            assert summary.get("selection_metric") == "rmse"
            assert summary.get("refit_on_full_train") is True
