"""Sample-weighted route metrics in physical energy units."""
import numpy as np


def metrics(y, p):
    (y, p) = (np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64"))
    err = p - y
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.square(err).mean())),
        "MAPE%": float(np.abs(err / (y + 1e-08)).mean() * 100),
        "R2": float(1 - np.square(err).sum() / np.square(y - y.mean()).sum()),
    }
