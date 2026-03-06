"""
Scaler factory and options for variation testing.
Supports: standard (StandardScaler), minmax (MinMaxScaler), robust (RobustScaler), none.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def get_scaler(name="standard", feature_range=(0, 1), **kwargs):
    """
    Return a fresh scaler instance by name for fit/transform use.

    Parameters
    ----------
    name : str
        One of "standard", "minmax", "robust", "none".
    feature_range : tuple
        Used only for MinMaxScaler: (min, max) for scaled values.
    **kwargs
        Passed to the scaler constructor (e.g. RobustScaler quantile_range).

    Returns
    -------
    scaler instance or None
        None for name="none" (no scaling).
    """
    name = (name or "standard").lower().strip()
    if name == "none":
        return None
    if name == "standard":
        return StandardScaler(**kwargs)
    if name == "minmax":
        return MinMaxScaler(feature_range=feature_range, **kwargs)
    if name == "robust":
        return RobustScaler(**kwargs)
    raise ValueError(f"Unknown scaler: {name}. Use one of: standard, minmax, robust, none.")
