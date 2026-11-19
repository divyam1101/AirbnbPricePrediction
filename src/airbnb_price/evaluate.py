"""Metrics reported on both the log scale (what the model optimises) and in euros."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_log_true, y_log_pred) -> dict[str, float]:
    y_log_true = np.asarray(y_log_true, dtype=float)
    y_log_pred = np.asarray(y_log_pred, dtype=float)
    price_true = np.exp(y_log_true)
    price_pred = np.exp(y_log_pred)
    ape = np.abs(price_pred - price_true) / price_true
    return {
        "r2_log": float(r2_score(y_log_true, y_log_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_log_true, y_log_pred))),
        "mae_eur": float(mean_absolute_error(price_true, price_pred)),
        "median_ape": float(np.median(ape)),
        "n": int(len(y_log_true)),
    }
