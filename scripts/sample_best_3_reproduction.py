#!/usr/bin/env python3
"""
Generate sample code to reproduce the best 3 models using saved tune/hyperparameter
results and scaler (normalization) setup from a run.

Usage:
  python scripts/sample_best_3_reproduction.py results/dataset_name/YYYY-MM-DD
  python scripts/sample_best_3_reproduction.py results/dataset_name  # uses latest date subdir

Output: prints sample Python code for the best 3 configs (hyperparameters + normalization).
"""

import argparse
import json
import os
import sys

import pandas as pd


def find_results_dir(base_path: str) -> str:
    """Return path to the dir containing model_configs.json and results (summary or CSV)."""
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Not a directory: {base_path}")
    def has_results(p):
        return (
            os.path.isfile(os.path.join(p, "model_configs.json"))
            and (
                os.path.isfile(os.path.join(p, "results_summary.json"))
                or os.path.isfile(os.path.join(p, "model_comparison_results.csv"))
            )
        )
    if has_results(base_path):
        return base_path
    date_dirs = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))],
        reverse=True,
    )
    for d in date_dirs:
        candidate = os.path.join(base_path, d)
        if has_results(candidate):
            return candidate
    raise FileNotFoundError(
        f"No model_configs.json with results_summary.json or model_comparison_results.csv under {base_path}"
    )


def load_best_3(results_dir: str):
    """Load results summary and model configs; return (best 3 results, best 3 configs)."""
    configs_path = os.path.join(results_dir, "model_configs.json")
    if not os.path.isfile(configs_path):
        raise FileNotFoundError(configs_path)
    with open(configs_path) as f:
        all_configs = json.load(f)
    summary_path = os.path.join(results_dir, "results_summary.json")
    csv_path = os.path.join(results_dir, "model_comparison_results.csv")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        results = summary.get("results", [])[:3]
        configs = all_configs[:3]
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        results = df.head(3).to_dict("records")
        # Configs may be one-per-model (same order) or one-per-variation; take first 3
        configs = all_configs[:3]
    else:
        raise FileNotFoundError(
            f"Need either {summary_path} or {csv_path} (and {configs_path})"
        )
    return results, configs


def generate_sample_code(results, configs, results_dir: str) -> str:
    """Generate sample Python code for the best 3 using hyperparameters and scaler setup."""
    res_dir_slash = results_dir.replace(os.sep, "/")
    lines = [
        "# Generated sample: reproduce best 3 models using saved tune/hyperparameter results",
        "# and scaler (normalization) setup. Handles StandardScaler, MinMaxScaler, RobustScaler.",
        "# Training settings (epochs, batch_size, patience) are not persisted; defaults below may differ from run.",
        "",
        "import json",
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
        "",
        "# Optional: use project's get_scaler for standard/minmax/robust/none",
        "# from src.preprocessing import get_scaler",
        "",
        f'RESULTS_DIR = "{res_dir_slash}"',
        "",
        "# Training defaults used in pipeline (not saved in config; your run may differ)",
        "TRAINING_DEFAULTS = {",
        "    \"epochs_search\": 50,   # grid search / validation folds",
        "    \"epochs_refit\": 200,   # final refit on full train (MLP, LSTM, RNN, etc.)",
        "    \"batch_size\": 32,     # or from hp when tuned (e.g. LSTM/RNN: 16 or 32)",
        "    \"patience\": 10,       # EarlyStopping",
        "}",
        "",
        "# Load saved configs (best 3); hyperparameters may come from grid search when tuned=True",
        'with open(f"{RESULTS_DIR}/model_configs.json") as f:',
        "    configs = json.load(f)[:3]",
        "",
        "# Load best 3 results (composite_score to 4 decimal places)",
        "try:",
        '    with open(f"{RESULTS_DIR}/results_summary.json") as f:',
        "        summary = json.load(f)",
        '    results = summary["results"][:3]',
        "except FileNotFoundError:",
        '    df = pd.read_csv(f"{RESULTS_DIR}/model_comparison_results.csv")',
        '    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)',
        "    results = df.head(3).to_dict('records')",
        "",
        "for rank, (res, cfg) in enumerate(zip(results, configs), start=1):",
        "    model_name = res.get('model', cfg.get('display_name', '?'))",
        "    composite_score = float(res.get('composite_score', 0))",
        "    hp = cfg.get('hyperparameters', {})",
        "    norm = cfg.get('normalization', {})",
        "    time_steps = cfg.get('time_steps')",
        "    setup = cfg.get('setup', {})",
        "    tuned = cfg.get('tuned', False)",
        "    selection_metric = cfg.get('selection_metric', 'rmse')",
        "",
        "    print(f'--- Best {rank}: {model_name} (composite_score={composite_score:.4f}) ---')",
        "    print('Hyperparameters:', json.dumps(hp, indent=2))",
        "    print('Normalization (scaler setup):', json.dumps(norm, indent=2))",
        "    if time_steps:",
        "        print('Time steps (e.g. n_steps for sequences):', time_steps)",
        "    if tuned:",
        "        print('(Params from grid search; selection_metric=', selection_metric, ', validation-only)')",
        "    print()",
        "",
        "# Scaler reproduction: choose type from norm.type or hp.get('scaler')",
        "#   norm.type / hp['scaler'] can be: StandardScaler, MinMaxScaler, RobustScaler, none",
        "#   For MinMaxScaler use feature_range=norm.get('feature_range') or hp.get('feature_range') or (0, 1)",
        "# Example:",
        "#   scaler_type = (norm.get('type') or hp.get('scaler') or 'standard').lower()",
        "#   if scaler_type in ('none', None): scaler = None",
        "#   elif scaler_type == 'minmax':",
        "#       fr = tuple(norm.get('feature_range') or hp.get('feature_range') or (0, 1))",
        "#       scaler = MinMaxScaler(feature_range=fr)",
        "#   elif scaler_type == 'robust': scaler = RobustScaler()",
        "#   else: scaler = StandardScaler()",
        "#   if scaler: scaler.fit(X_train); X_train_sc = scaler.transform(X_train); X_test_sc = scaler.transform(X_test)",
        "",
    ]
    return "\n".join(lines)


def print_best_3_summary(results, configs):
    """Print a compact table of best 3 with composite_score (4 decimals), hp, and scaler setup."""
    print("\nBest 3 models (composite_score to 4 decimal places)\n")
    for rank, (res, cfg) in enumerate(zip(results, configs), start=1):
        name = res.get("model", "?")
        composite = res.get("composite_score", 0)
        hp = cfg.get("hyperparameters", {})
        norm = cfg.get("normalization", {})
        tuned = cfg.get("tuned", False)
        print(f"  {rank}. {name}")
        print(f"     composite_score = {composite:.4f}")
        print(f"     hyperparameters = {json.dumps(hp)}")
        norm_str = json.dumps(norm)
        print(f"     normalization   = {norm_str}")
        if norm.get("type") and norm.get("type") != "none":
            if norm.get("feature_range"):
                print(f"     scaler          = {norm['type']} (feature_range={norm['feature_range']})")
            else:
                print(f"     scaler          = {norm['type']}")
        if tuned:
            print(f"     (params from grid search; selection_metric = {cfg.get('selection_metric', 'rmse')})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample code for the best 3 models from a run (tune/hyperparameter results + scaler setup)."
    )
    parser.add_argument(
        "results_path",
        help="Path to dataset results dir (e.g. results/my_dataset/2025-03-09 or results/my_dataset)",
    )
    parser.add_argument(
        "--code",
        action="store_true",
        help="Print generated sample Python code to stdout.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write generated code to this file instead of stdout.",
    )
    args = parser.parse_args()
    results_path = os.path.abspath(args.results_path)
    try:
        results_dir = find_results_dir(results_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    results, configs = load_best_3(results_dir)
    print_best_3_summary(results, configs)
    code = generate_sample_code(results, configs, results_dir)
    if args.out:
        with open(args.out, "w") as f:
            f.write(code)
        print(f"Sample code written to {args.out}")
    else:
        if args.code:
            print("\n" + "=" * 60 + "\nGenerated sample code\n" + "=" * 60)
            print(code)
        else:
            print("Use --code to print sample code, or --out FILE to write it to a file.")


if __name__ == "__main__":
    main()
