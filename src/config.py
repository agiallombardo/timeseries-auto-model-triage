"""
Load configuration from environment (e.g. after load_dotenv()).
Used by main to apply .env overrides with precedence: code defaults <- .env <- CLI.
"""
import os

# Default models when neither CLI nor .env set --models: ML and DL families only (no statistical).
DEFAULT_MODELS = [
    "rf", "svr", "xgb", "lr",
    "rnn", "lstm", "mlp", "lstm_feat", "rnn_feat", "cnn1d",
]


def _env(key: str) -> str | None:
    v = os.environ.get(key)
    return v.strip() if (v and isinstance(v, str)) else None


def get_env_str(key: str) -> str | None:
    return _env(key)


def get_env_int(key: str) -> int | None:
    v = _env(key)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def get_env_bool(key: str) -> bool | None:
    v = _env(key)
    if v is None:
        return None
    return v.lower() in ("1", "true", "yes")


def get_env_list(key: str, sep: str = ",") -> list[str] | None:
    v = _env(key)
    if v is None:
        return None
    return [s.strip() for s in v.split(sep) if s.strip()]


def resolve_data_args(args) -> None:
    """Fill file, time_col, data_col, date_format from env if not set on args."""
    if getattr(args, "file", None) is None:
        args.file = get_env_str("DATA_FILE")
    if getattr(args, "time_col", None) is None:
        args.time_col = get_env_str("TIME_COL")
    if getattr(args, "data_col", None) is None:
        args.data_col = get_env_str("DATA_COL")
    if getattr(args, "date_format", None) is None:
        args.date_format = get_env_str("DATE_FORMAT")


def resolve_run_args(args, default_setup: dict) -> None:
    """Fill run/tuning/output args from env when not set on CLI. Mutates args and default_setup."""
    if getattr(args, "output_dir", None) is None or args.output_dir == "results":
        o = get_env_str("OUTPUT_DIR")
        if o is not None:
            args.output_dir = o
    if args.output_dir is None:
        args.output_dir = "results"

    if getattr(args, "models", None) is None or (len(args.models) == 1 and args.models[0] == "all"):
        m = get_env_list("MODELS")
        if m is not None:
            args.models = m
    if args.models is None:
        args.models = list(DEFAULT_MODELS)

    if getattr(args, "losses", None) is None:
        l = get_env_list("LOSSES")
        if l is not None:
            args.losses = l

    if getattr(args, "tune_top", None) is None:
        t = get_env_int("TUNE_TOP")
        if t is not None:
            args.tune_top = t
    if args.tune_top is None:
        args.tune_top = 3

    tune_all_env = get_env_bool("TUNE_ALL")
    if tune_all_env is not None and not getattr(args, "tune_all", False):
        args.tune_all = tune_all_env

    if getattr(args, "jobs", None) is None:
        j = get_env_int("JOBS")
        if j is not None:
            args.jobs = j
    if args.jobs is None:
        args.jobs = 1

    if getattr(args, "n_runs", None) is None:
        n = get_env_int("N_RUNS")
        if n is not None:
            args.n_runs = n
        else:
            args.n_runs = default_setup.get("n_runs", 3)
    if args.n_runs is None:
        args.n_runs = default_setup.get("n_runs", 3)

    # Override default_setup for n_runs so downstream code sees it
    n = get_env_int("N_RUNS")
    if n is not None:
        default_setup["n_runs"] = n

    if get_env_bool("MINIMAL_OUTPUT") is True:
        args.minimal_output = True
    if get_env_bool("NO_CHARTS") is True:
        args.no_charts = True


def get_dl_overrides() -> dict:
    """Return DL tuning overrides from env (epochs_grid, epochs_refit, patience). Unset keys are absent."""
    out = {}
    n = get_env_int("DL_EPOCHS_GRID")
    if n is not None:
        out["epochs_grid"] = n
    n = get_env_int("DL_EPOCHS_REFIT")
    if n is not None:
        out["epochs_refit"] = n
    n = get_env_int("DL_PATIENCE")
    if n is not None:
        out["patience"] = n
    return out


def apply_tuning_setup_from_env() -> None:
    """Apply TUNING_N_SPLITS and TUNING_FAST to model_config.TUNING_SETUP so tuning modules see them."""
    from src import model_config
    n = get_env_int("TUNING_N_SPLITS")
    if n is not None:
        model_config.TUNING_SETUP["n_splits"] = n
    b = get_env_bool("TUNING_FAST")
    if b is not None:
        model_config.TUNING_SETUP["tuning_fast"] = b
