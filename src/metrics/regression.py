import numpy as np


def _as_float_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
