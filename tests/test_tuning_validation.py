"""
Unit tests for shared time-series validation helpers.

Contract: no fold uses future data in training; no tuner uses y_test/X_test
during parameter selection; fold boundaries behave on short series.
"""

import numpy as np
import pandas as pd
import pytest

from src.tuning._validation import (
    get_time_series_splits,
    get_chronological_holdout,
    get_chronological_holdout_indices,
    score_validation_rmse,
    aggregate_fold_scores,
    DEFAULT_N_SPLITS,
    DEFAULT_VAL_FRAC,
)


class TestTimeSeriesSplits:
    """No fold uses future data: train indices strictly before val indices."""

    def test_splits_no_future_data(self):
        n = 100
        X = np.arange(n).reshape(-1, 1)
        for train_idx, val_idx in get_time_series_splits(X, n_splits=3):
            assert train_idx.max() < val_idx.min(), "training must not use future data"
            assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_splits_chronological_order(self):
        n = 50
        X = pd.DataFrame({"x": np.arange(n)})
        splits = list(get_time_series_splits(X, n_splits=3))
        assert len(splits) == 3
        for train_idx, val_idx in splits:
            assert np.all(train_idx < val_idx.max())

    def test_short_series_fold_count(self):
        # TimeSeriesSplit with n_splits=3 needs enough samples; with 20 points we get 3 folds
        X = np.arange(20).reshape(-1, 1)
        splits = list(get_time_series_splits(X, n_splits=3))
        assert len(splits) == 3

    def test_very_short_series(self):
        # With 10 points, n_splits=3 still yields 3 folds (sklearn allows minimal train size)
        X = np.arange(10).reshape(-1, 1)
        splits = list(get_time_series_splits(X, n_splits=3))
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()


class TestChronologicalHoldout:
    """Single holdout: last val_frac of train is validation; no future data in train."""

    def test_holdout_no_future_data(self):
        for len_train in (20, 50, 100):
            train_idx, val_idx = get_chronological_holdout_indices(len_train, val_frac=0.2)
            assert train_idx.max() < val_idx.min()
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_holdout_sizes(self):
        len_train = 100
        train_idx, val_idx = get_chronological_holdout_indices(len_train, val_frac=0.2)
        assert len(train_idx) + len(val_idx) == len_train
        assert len(val_idx) >= 1
        assert len(train_idx) >= 1

    def test_short_series_holdout(self):
        # len_train=10, val_frac=0.2 -> val_size at least 1, train at least 1
        train_idx, val_idx = get_chronological_holdout_indices(10, val_frac=0.2)
        assert len(val_idx) >= 1
        assert len(train_idx) >= 1
        assert train_idx.max() < val_idx.min()

    def test_slices_match_indices(self):
        len_train = 25
        train_sl, val_sl = get_chronological_holdout(len_train, val_frac=0.2)
        train_idx, val_idx = get_chronological_holdout_indices(len_train, val_frac=0.2)
        assert np.array_equal(np.arange(len_train)[train_sl], train_idx)
        assert np.array_equal(np.arange(len_train)[val_sl], val_idx)


class TestScoring:
    def test_score_validation_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.0, 2.9])
        rmse = score_validation_rmse(y, pred)
        assert rmse >= 0
        assert np.isclose(rmse, np.sqrt(((0.1**2) + (0**2) + (0.1**2)) / 3))

    def test_score_validation_rmse_empty(self):
        assert score_validation_rmse(np.array([]), np.array([])) == float("inf")

    def test_aggregate_fold_scores(self):
        scores = [1.0, 2.0, 3.0]
        assert aggregate_fold_scores(scores) == 2.0
        assert aggregate_fold_scores(scores, method="median") == 2.0

    def test_aggregate_fold_scores_empty(self):
        assert aggregate_fold_scores([]) == float("inf")
