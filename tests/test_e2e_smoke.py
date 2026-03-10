"""
End-to-end smoke test: run pipeline on a small dataset with tuning,
assert ranking completes and tuning output is comparable (validation-only selection).
"""

import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

# Run from repo root so src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module


def _small_csv(path: str, n: int = 80) -> None:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    y = np.cumsum(np.random.randn(n)) + 100
    pd.DataFrame({"date": dates, "sales": y}).to_csv(path, index=False)


@pytest.mark.slow
def test_e2e_ranking_completes_with_tuning():
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "small.csv")
        _small_csv(data_path, n=80)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        sys.argv = [
            "main",
            "--file", data_path,
            "--time_col", "date",
            "--data_col", "sales",
            "--models", "arima", "ma",
            "--tune_top", "1",
            "--output_dir", out_dir,
        ]
        main_module.main()
        dataset_name = "small"
        # Results are under output_dir/dataset_name/YYYY-MM-DD
        results_dir = os.path.join(out_dir, dataset_name)
        date_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        assert date_dirs, "expected a date subdir under results"
        summary_path = os.path.join(results_dir, date_dirs[0], "results_summary.json")
        assert os.path.isfile(summary_path), "results_summary.json should be written"
        with open(summary_path) as f:
            summary = json.load(f)
        assert "results" in summary
        assert "dataset" in summary
        # When we tuned at least one model, parameter_selection should be present
        if summary.get("tuned_best_params"):
            assert "parameter_selection" in summary
            assert summary["parameter_selection"].get("metric") == "rmse"
        # Each result entry should have hyperparameters_source
        for r in summary["results"]:
            assert "hyperparameters_source" in r
            assert r["hyperparameters_source"] in ("validation", "default_variation")
